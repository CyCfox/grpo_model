#!/usr/bin/env python
"""
Memory-R1 GRPO Training Entry Point.
Trains a Qwen3.5-0.8B Memory Manager with GRPO on the LoCoMo dataset.

Usage:
    python run_train.py                  # Train from scratch
    python run_train.py --resume STEP    # Resume from checkpoint
"""
# CRITICAL: Unsloth must be imported before transformers, peft, etc.
import unsloth  # noqa: F401

import sys
import os
import time
import argparse
import json
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_r1.config import (
    LOCOMO_DATA_PATH, CHECKPOINT_DIR, LOG_DIR, EXTRACTED_FACTS_DIR,
    CORE_MEMORY_LOG, TRAIN_CONV_INDICES, LEARNING_RATE,
)
from memory_r1.data_utils import load_locomo_data, split_conversations, get_conversation_sessions
from memory_r1.memory_store import get_embedding_model, MemoryStore
from memory_r1.model_utils import load_model_and_tokenizer, set_inference_mode, load_checkpoint
from memory_r1.grpo_trainer import GRPOTrainer


def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [CHECKPOINT_DIR, LOG_DIR, EXTRACTED_FACTS_DIR]:
        os.makedirs(d, exist_ok=True)


def write_core_memory_header():
    """Initialize the core_memory.md log file."""
    header = f"""# Memory-R1 GRPO Training Log

## Project Overview
- **Task**: GRPO fine-tuning of Qwen3.5-0.8B for Memory Management
- **Framework**: Unsloth + LoRA 4-bit (rank=16)
- **Dataset**: LoCoMo locomo10.json ({len(TRAIN_CONV_INDICES)} conversations for training)
- **Base Paper**: Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via RL

## Configuration
- Model: Qwen3.5-0.8B (4-bit QLoRA, rank=16)
- Embedding: Qwen3-Embedding-0.6B (CPU, 1024-dim)
- Answer Generation: DeepSeek v4-flash API
- Fact Extraction: DeepSeek v4-flash API
- GRPO: G=4, e=0.2, lr=5e-6
- Update/Save: Every 5 sessions
- Memory Retrieval: Top-10 via cosine similarity

## Training Started
- Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

---

"""
    with open(CORE_MEMORY_LOG, "w", encoding="utf-8") as f:
        f.write(header)


def main():
    parser = argparse.ArgumentParser(description="Memory-R1 GRPO Training")
    parser.add_argument(
        "--resume", "--resume-from",
        type=str, default=None, metavar="CHECKPOINT_PATH",
        help="Resume training from a checkpoint (e.g., checkpoints/step_0055)",
    )
    parser.add_argument(
        "--start-conv", type=int, default=None,
        help="Override: start from this conversation index (0-based)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Memory-R1 GRPO Training")
    print("=" * 60)

    # Ensure directories
    ensure_dirs()

    # 1. Load data
    print("\n[1/4] Loading LoCoMo dataset...")
    data = load_locomo_data(LOCOMO_DATA_PATH)
    print(f"  Loaded {len(data)} conversations")

    train_conv, val_conv, test_conv = split_conversations(data)
    print(f"  Train: {len(train_conv)}, Val: {len(val_conv)}, Test: {len(test_conv)}")

    # 2. Load embedding model (CPU)
    print("\n[2/4] Loading embedding model (CPU)...")
    embedding_model = get_embedding_model()
    print(f"  Embedding model loaded: {embedding_model}")

    # 3. Load main model (GPU, 4-bit LoRA)
    print("\n[3/4] Loading Qwen3.5-0.8B with 4-bit LoRA...")
    model, tokenizer = load_model_and_tokenizer()

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 4. Initialize trainer
    print("\n[4/4] Preparing trainer...")

    # Determine resume parameters
    start_conv_idx = 0
    start_session_idx = 0
    start_global_session = 0

    if args.resume:
        checkpoint_path = args.resume
        print(f"\n  Resuming from checkpoint: {checkpoint_path}")

        # Load checkpoint info first (without model) to get metadata
        meta_path = os.path.join(checkpoint_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            print(f"  Checkpoint metadata: {json.dumps(metadata, indent=2)}")

        resume_path = os.path.join(checkpoint_path, "resume_state.json")
        resume_state = {}
        if os.path.exists(resume_path):
            with open(resume_path, "r") as f:
                resume_state = json.load(f)
            print(f"  Resume state: {json.dumps(resume_state, indent=2)}")

        # Determine where to resume from
        ckpt_conv = metadata.get("conv_idx", resume_state.get("conv_idx", 0))
        ckpt_session_num = metadata.get("session_num", resume_state.get("session_num", 0))
        loaded_memory_store = None

        if args.start_conv is not None:
            # User explicitly specified
            start_conv_idx = args.start_conv
            start_session_idx = 0
            print(f"  User specified: start from conv {start_conv_idx}")
        else:
            # Auto-detect: check if there are remaining sessions in the checkpoint's conversation
            if ckpt_conv < len(train_conv):
                conv_sessions = get_conversation_sessions(train_conv[ckpt_conv])
                total_in_conv = len(conv_sessions)
                # ckpt_session_num is the LAST completed session (at checkpoint time)
                next_session_idx = ckpt_session_num  # session indexing is 0-based in enumerate()
                if next_session_idx < total_in_conv:
                    # Resume within same conversation
                    start_conv_idx = ckpt_conv
                    start_session_idx = next_session_idx
                    print(f"  Auto-resume: conv {ckpt_conv} has {total_in_conv} sessions, "
                          f"{ckpt_session_num} completed → resume at session {ckpt_session_num + 1}")
                else:
                    # This conversation done, move to next
                    start_conv_idx = ckpt_conv + 1
                    start_session_idx = 0
                    print(f"  Auto-resume: conv {ckpt_conv} fully completed, starting at conv {start_conv_idx}")
            else:
                start_conv_idx = ckpt_conv + 1
                start_session_idx = 0
                print(f"  Auto-resume: skipping to conv {start_conv_idx}")

        start_global_session = metadata.get("global_session", resume_state.get("global_session", 0))

        # Create optimizer first (same structure as trainer would)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
        )

        # Load model weights + optimizer state
        result = load_checkpoint(
            model, checkpoint_path,
            load_optimizer=True, optimizer=optimizer,
        )
        print(f"  Loaded weights from: {checkpoint_path}")
        if result["memory_bank_path"] and start_conv_idx == ckpt_conv:
            # Resuming mid-conversation — load the memory bank
            loaded_memory_store = MemoryStore.load_state(
                result["memory_bank_path"], embedding_model=embedding_model,
            )
            print(f"  Memory bank loaded: {len(loaded_memory_store)} entries")
        elif result["memory_bank_path"]:
            print(f"  Memory bank found but not loaded (starting new conversation)")

        # Create trainer with loaded optimizer
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            optimizer=optimizer,
        )

        print(f"\n  Resume summary:")
        print(f"    Start conv idx:  {start_conv_idx} ({'conv ' + str(start_conv_idx) if start_conv_idx < len(train_conv) else 'no more'})")
        print(f"    Start session:   {start_session_idx}")
        print(f"    Global session:  {start_global_session}")

        if start_conv_idx >= len(train_conv):
            print(f"\n  ERROR: start_conv_idx ({start_conv_idx}) >= total train convs ({len(train_conv)}).")
            print(f"  All conversations already trained!")
            return

        trainer.train(
            train_conv,
            start_conv_idx=start_conv_idx,
            start_session_idx=start_session_idx,
            start_global_session=start_global_session,
            initial_memory_store=loaded_memory_store,
        )
    else:
        # Train from scratch
        write_core_memory_header()

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
        )

        trainer.train(train_conv)

    print("\nDone!")


if __name__ == "__main__":
    main()
