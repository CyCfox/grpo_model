"""
GRPO Trainer for Memory-R1.
Implements the full GRPO training loop with:
- Session-level trajectory sampling (G=4)
- Token-level PPO-style clipped loss with group-normalized advantages
- Checkpointing every 5 sessions
- Old policy update every 5 sessions
"""
import os
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizer

from .config import (
    G, EPSILON, LEARNING_RATE, SESSION_UPDATE_FREQ, CHECKPOINT_FREQ,
    MAX_GRAD_NORM, DEVICE, TOP_K_RETRIEVAL, TEMPERATURE_TRAIN, MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH, CHECKPOINT_DIR, LOG_DIR,
)
from .data_utils import (
    get_conversation_sessions, get_session_datetime, build_session_qa_map,
    get_available_qas,
)
from .fact_extraction import extract_session_facts
from .memory_store import MemoryStore
from .reward import compute_step_reward, compute_trajectory_rewards, apply_memory_operation
from .answer_gen import evaluate_trajectory_answers
from .model_utils import (
    set_inference_mode, set_training_mode, save_checkpoint,
    generate_action, compute_step_loss,
)
from .prompts import MEMORY_MANAGER_SYSTEM, MEMORY_MANAGER_USER


# ===== Data Structures =====

@dataclass
class StepData:
    """Data for one step (dialog turn) in one trajectory."""
    prompt_text: str
    action_text: str          # raw generated text
    action_ids: List[int]     # token IDs of the action
    old_log_probs: List[float]  # log probs under old policy
    r2: float                 # format correctness (0 or 1)


@dataclass
class TrajectoryData:
    """Data for a complete trajectory through a session."""
    steps: List[StepData] = field(default_factory=list)
    r1: float = 0.0
    memory_store: Optional[MemoryStore] = None


@dataclass
class SessionData:
    """Data for a complete session with G=4 trajectories."""
    session_num: int
    trajectories: List[TrajectoryData] = field(default_factory=list)
    # Best trajectory index (for memory propagation)
    best_traj_idx: int = 0


# ===== Prompt Construction =====

def build_memory_manager_prompt(
    fact_text: str,
    retrieved_memories: List[Dict[str, str]],
    tokenizer,
    top_k: int = TOP_K_RETRIEVAL,
) -> str:
    """
    Build the full prompt for the Memory Manager model.
    Uses the Qwen3 chat template via tokenizer.apply_chat_template().
    """
    # Format retrieved memories
    if retrieved_memories:
        mem_lines = []
        for mem in retrieved_memories:
            mem_lines.append(
                f'{{"id": "{mem["id"]}", "text": "{mem["text"]}"}}'
            )
        mem_text = "\n".join(mem_lines)
    else:
        mem_text = "(No existing memories)"

    user_content = MEMORY_MANAGER_USER.format(
        fact_text=fact_text,
        top_k=top_k,
        retrieved_memories=mem_text,
    )

    messages = [
        {"role": "system", "content": MEMORY_MANAGER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return prompt


# ===== GRPO Trainer =====

class GRPOTrainer:
    """
    GRPO trainer for Memory Manager.

    Training loop:
    1. For each conversation, process sessions sequentially
    2. For each session, process each dialog turn, sampling G=4 trajectories
    3. After each session, evaluate QA to compute r1
    4. Every SESSION_UPDATE_FREQ sessions, compute GRPO loss and update
    """

    def __init__(
        self,
        model: FastLanguageModel,
        tokenizer: PreTrainedTokenizer,
        embedding_model,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.optimizer = optimizer or torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
        )

        # Training state
        self.global_session_count = 0
        self.total_steps = 0
        # Accumulated trajectory data for GRPO update.
        # Each entry: {"session_data": SessionData, "conv_idx": int,
        #              "session_num": int, "memory_size": int}
        self._accumulated_sessions: List[Dict] = []

        # Log file (one line per session)
        os.makedirs(LOG_DIR, exist_ok=True)
        self.train_log_path = os.path.join(LOG_DIR, "train_log.jsonl")

    def _log(self, msg: str):
        """Log a message with timestamp to console."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg, flush=True)

    def _write_train_log(self, session_data: SessionData, conv_idx: int,
                          session_num: int, global_session: int,
                          memory_size: int, loss: float):
        """Write one session record to train_log.jsonl."""
        r1_values = [t.r1 for t in session_data.trajectories]
        traj_r2_means = []
        all_r2s = []
        for t in session_data.trajectories:
            t_r2s = [step.r2 for step in t.steps]
            all_r2s.extend(t_r2s)
            traj_r2_means.append(np.mean(t_r2s) if t_r2s else 0.0)
        avg_r2 = np.mean(all_r2s) if all_r2s else 0.0
        avg_r1 = np.mean(r1_values)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_session": global_session,
            "conv_idx": conv_idx,
            "session_num": session_num,
            "memory_size": memory_size,
            "traj_r1": [round(v, 4) for v in r1_values],
            "avg_r1": round(avg_r1, 4),
            "traj_r2": [round(v, 4) for v in traj_r2_means],
            "avg_r2": round(avg_r2, 4),
            "reward": round(avg_r1 + avg_r2, 4),
            "loss": round(loss, 6),
        }
        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ===== Phase 1: Rollout =====

    def _run_session_rollout(
        self,
        session_num: int,
        session_dialogs: List[Dict],
        date_time: str,
        conv_idx: int,
        current_memory: MemoryStore,
    ) -> Tuple[SessionData, MemoryStore]:
        """
        Run a full session rollout with G=4 trajectories.

        Steps:
        1. For each dialog turn, extract facts and sample operations
        2. Each trajectory maintains its own memory store copy
        3. After all turns, evaluate QA for each trajectory

        Returns:
            (session_data, best_memory_for_next_session)
        """
        T = len(session_dialogs)  # number of steps

        # Extract facts for all dialogs (with caching)
        facts_list = extract_session_facts(
            conv_idx=conv_idx,
            session_num=session_num,
            dialogs=session_dialogs,
            date_time=date_time,
        )

        # Initialize G=4 trajectories
        trajectories = [
            TrajectoryData(memory_store=current_memory.copy())
            for _ in range(G)
        ]

        # Process each dialog turn
        for t, (dialog, fact_entry) in enumerate(zip(session_dialogs, facts_list)):
            fact_text = fact_entry.get("fact", "NONE")

            # Skip turns with no substantive fact
            if fact_text == "NONE" or not fact_text.strip():
                # Still record a placeholder step (no operation needed)
                for i in range(G):
                    trajectories[i].steps.append(StepData(
                        prompt_text="",
                        action_text="",
                        action_ids=[],
                        old_log_probs=[],
                        r2=1.0,  # No operation needed, format is "correct" by default
                    ))
                continue

            # For each trajectory, sample an action
            for i in range(G):
                mem_store = trajectories[i].memory_store
                retrieved = mem_store.search(fact_text, top_k=TOP_K_RETRIEVAL)
                existing_ids = {m["id"] for m in mem_store.get_all_memories()}

                # Build prompt
                prompt = build_memory_manager_prompt(
                    fact_text, retrieved, self.tokenizer,
                )

                # Generate action from model
                action_text, action_ids, old_log_probs = generate_action(
                    self.model, self.tokenizer, prompt,
                    temperature=TEMPERATURE_TRAIN,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                # Compute r2 (format correctness)
                r2 = compute_step_reward(action_text, existing_ids)

                # Try to apply operation (best effort)
                applied = apply_memory_operation(mem_store, action_text)

                # Record step data
                trajectories[i].steps.append(StepData(
                    prompt_text=prompt,
                    action_text=action_text,
                    action_ids=action_ids if action_ids else [],
                    old_log_probs=old_log_probs if old_log_probs else [],
                    r2=r2,
                ))

        # ---- Phase 2: QA Evaluation ----
        # Get available QAs for this session
        # Build session QA map on the fly
        self._qa_map = self._qa_map or {}
        qa_list, self._used_qa_indices = get_available_qas(
            self._qa_map, session_num,
            used_qa_indices=self._used_qa_indices,
        )

        best_r1 = -1.0
        best_traj_idx = 0

        for i in range(G):
            if qa_list:
                mem_store = trajectories[i].memory_store
                _, r1 = evaluate_trajectory_answers(
                    qa_list, mem_store, top_k=TOP_K_RETRIEVAL,
                )
                trajectories[i].r1 = r1
            else:
                # No QAs available for this session
                trajectories[i].r1 = 0.0

            if trajectories[i].r1 > best_r1:
                best_r1 = trajectories[i].r1
                best_traj_idx = i

        # Create session data
        session_data = SessionData(
            session_num=session_num,
            trajectories=trajectories,
            best_traj_idx=best_traj_idx,
        )

        # Best trajectory's memory becomes next session's starting memory
        best_memory = trajectories[best_traj_idx].memory_store

        return session_data, best_memory

    # ===== Phase 3: GRPO Update =====

    def _grpo_update_session(self, session_data: SessionData) -> dict:
        """
        Perform one GRPO update using a single session's trajectory data.

        Gradient accumulation across all (step, trajectory) pairs within the session,
        then a single optimizer.step().

        GRPO objective (per session):
          J = (1/T)·Σ_t (1/G)·Σ_i (1/|a|)·Σ_j min(ρ·A, clip(ρ)·A)

        Key: called sequentially for each accumulated session. Between calls,
        the model weights have been updated, so ρ = π_new / π_old ≠ 1 for
        sessions after the first — the GRPO clipped objective actually matters.

        Returns metrics dict with "loss" key.
        """
        set_training_mode(self.model)
        self.optimizer.zero_grad()

        T = len(session_data.trajectories[0].steps)
        total_tokens = 0
        total_loss_val = 0.0
        steps_processed = 0

        for t in range(T):
            # Collect step data for all G trajectories at this step
            step_trajs = []
            for i in range(G):
                if t < len(session_data.trajectories[i].steps):
                    step_trajs.append((i, session_data.trajectories[i].steps[t]))

            if len(step_trajs) < G:
                continue

            # Compute rewards and group-normalized advantages
            r_values = []
            for i, step in step_trajs:
                r1_i = session_data.trajectories[i].r1
                r_ti = r1_i + step.r2
                r_values.append(r_ti)

            mean_r = np.mean(r_values)
            std_r = np.std(r_values) + 1e-8
            advantages = [(r - mean_r) / std_r for r in r_values]

            for (i, step), advantage in zip(step_trajs, advantages):
                if not step.action_ids or not step.old_log_probs:
                    continue

                try:
                    token_loss = compute_step_loss(
                        self.model, self.tokenizer,
                        step.prompt_text, step.action_ids,
                        step.old_log_probs, advantage,
                    )
                    num_tokens = min(len(step.action_ids), len(step.old_log_probs))
                    if num_tokens > 0 and token_loss.grad_fn is not None:
                        # J = (1/T)·(1/G)·(1/|a|)·Σ_j min(ρ·A, clip(ρ)·A)
                        scaled_loss = token_loss / (num_tokens * T * G)
                        scaled_loss.backward()
                        total_loss_val += token_loss.item()
                        total_tokens += num_tokens
                        steps_processed += 1
                except Exception:
                    continue

            if t % 10 == 0:
                torch.cuda.empty_cache()

        if total_tokens == 0:
            self._log("  No valid tokens, skipping session update.")
            set_inference_mode(self.model)
            return {"loss": 0.0}

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            MAX_GRAD_NORM,
        )
        self.optimizer.step()

        avg_loss = total_loss_val / total_tokens if total_tokens > 0 else 0.0

        torch.cuda.empty_cache()
        set_inference_mode(self.model)

        return {"loss": avg_loss}

    # ===== Main Training Loop =====

    def train(
        self,
        train_conversations: List[Dict],
        start_conv_idx: int = 0,
        start_session_idx: int = 0,
        start_global_session: int = 0,
        initial_memory_store: Optional[MemoryStore] = None,
    ):
        """
        Main training loop over training conversations.

        Args:
            train_conversations: List of conversation dicts from locomo10.json
            start_conv_idx: First conversation index to process (for resume)
            start_session_idx: First session index within start_conv_idx (for resume)
            start_global_session: Global session counter value at resume point
        """
        training_from_scratch = (start_conv_idx == 0 and start_session_idx == 0)

        self._log("=" * 60)
        if training_from_scratch:
            self._log("Memory-R1 GRPO Training Started")
        else:
            self._log("Memory-R1 GRPO Training Resumed")
            self._log(f"  From conv {start_conv_idx}, session {start_session_idx}, "
                      f"global_session={start_global_session}")
        self._log(f"  Model: Qwen3.5-0.8B (4-bit LoRA, rank={16})")
        self._log(f"  G={G}, ε={EPSILON}, lr={LEARNING_RATE}")
        self._log(f"  Update every {SESSION_UPDATE_FREQ} sessions")
        self._log(f"  Training conversations: {len(train_conversations)}")
        self._log("=" * 60)

        # Start in inference mode for rollouts
        set_inference_mode(self.model)
        self.global_session_count = start_global_session

        for conv_idx, conversation in enumerate(train_conversations):
            # Skip completed conversations
            if conv_idx < start_conv_idx:
                self._log(f"  Skipping conv {conv_idx} (already completed)")
                continue

            sample_id = conversation.get("sample_id", f"conv_{conv_idx}")
            self._log(f"\n{'='*40}")
            self._log(f"Conversation {conv_idx+1}/{len(train_conversations)}: {sample_id}")
            self._log(f"{'='*40}")

            # Reset memory bank and QA tracking per conversation
            if initial_memory_store is not None and conv_idx == start_conv_idx:
                current_memory = initial_memory_store
                self._log(f"  Using provided memory bank: {len(current_memory)} entries")
            else:
                current_memory = MemoryStore(embedding_model=self.embedding_model)
            self._qa_map = build_session_qa_map(conversation)
            self._used_qa_indices: Set[str] = set()

            # Get sessions
            sessions = get_conversation_sessions(conversation)
            self._log(f"  Total sessions: {len(sessions)}")

            for sess_idx, (session_num, session_key, session_dialogs) in enumerate(sessions):
                # Skip completed sessions within the resume conversation
                if conv_idx == start_conv_idx and sess_idx < start_session_idx:
                    self._log(f"  Skipping session {session_num} (already completed)")
                    continue

                session_start = time.time()
                date_time = get_session_datetime(conversation, session_num)

                self._log(
                    f"  Session {session_num} ({sess_idx+1}/{len(sessions)}): "
                    f"{len(session_dialogs)} dialogs"
                )

                # Run rollout
                session_data, current_memory = self._run_session_rollout(
                    session_num=session_num,
                    session_dialogs=session_dialogs,
                    date_time=date_time,
                    conv_idx=conv_idx,
                    current_memory=current_memory,
                )

                # ---- Console log session metrics ----
                r1_values = [t.r1 for t in session_data.trajectories]
                all_r2s = []
                traj_r2_means = []
                for t in session_data.trajectories:
                    t_r2s = [step.r2 for step in t.steps]
                    all_r2s.extend(t_r2s)
                    traj_r2_means.append(np.mean(t_r2s) if t_r2s else 0.0)
                avg_r2 = np.mean(all_r2s) if all_r2s else 0.0
                avg_r1 = np.mean(r1_values)

                self._log(
                    f"    R1: {[f'{r:.3f}' for r in r1_values]}, "
                    f"avg={avg_r1:.3f}, best={r1_values[session_data.best_traj_idx]:.3f} "
                    f"(traj {session_data.best_traj_idx})"
                )
                self._log(
                    f"    traj_r2: {[f'{r:.3f}' for r in traj_r2_means]}, "
                    f"avg_r2={avg_r2:.3f} | "
                    f"avg_r1+r2={avg_r1 + avg_r2:.3f}"
                )

                # Accumulate for GRPO update (store metadata alongside session data)
                self._accumulated_sessions.append({
                    "session_data": session_data,
                    "conv_idx": conv_idx,
                    "session_num": session_num,
                    "memory_size": len(current_memory),
                })
                self.global_session_count += 1

                session_time = time.time() - session_start
                self._log(f"    Time: {session_time:.1f}s")

                # ---- Checkpoint & Update every SESSION_UPDATE_FREQ sessions ----
                if self.global_session_count % SESSION_UPDATE_FREQ == 0:
                    # GRPO update: one session at a time.
                    # π changes between sessions → ρ ≠ 1 after the first.
                    S = len(self._accumulated_sessions)
                    total_loss = 0.0
                    for sess_idx, entry in enumerate(self._accumulated_sessions):
                        sess_data = entry["session_data"]
                        sess_global = self.global_session_count - S + sess_idx + 1

                        sess_metrics = self._grpo_update_session(sess_data)
                        loss = sess_metrics.get("loss", 0.0)
                        total_loss += loss

                        # Write one record per session to train_log.jsonl
                        self._write_train_log(
                            session_data=sess_data,
                            conv_idx=entry["conv_idx"],
                            session_num=entry["session_num"],
                            global_session=sess_global,
                            memory_size=entry["memory_size"],
                            loss=loss,
                        )

                        r1_vals = [t.r1 for t in sess_data.trajectories]
                        self._log(
                            f"    Update {sess_idx+1}/{S} "
                            f"(session {sess_global}): "
                            f"loss={loss:.6f}, avg_r1={np.mean(r1_vals):.3f}"
                        )
                    avg_update_loss = total_loss / S
                    self._accumulated_sessions = []

                    # Save checkpoint
                    meta = {
                        "global_session": self.global_session_count,
                        "conv_idx": conv_idx,
                        "session_num": session_num,
                        "loss": avg_update_loss,
                        "lr": LEARNING_RATE,
                    }
                    ckpt_path = save_checkpoint(
                        self.model, self.tokenizer, self.optimizer,
                        self.global_session_count, CHECKPOINT_DIR,
                        extra_info=meta,
                        memory_store=current_memory,
                    )
                    self._log(f"    Checkpoint saved: {ckpt_path}")

        # Final update for any remaining accumulated data
        if self._accumulated_sessions:
            self._log("\nFinal GRPO updates...")
            S = len(self._accumulated_sessions)
            total_loss = 0.0
            for sess_idx, entry in enumerate(self._accumulated_sessions):
                sess_data = entry["session_data"]
                sess_global = self.global_session_count - S + sess_idx + 1

                sess_metrics = self._grpo_update_session(sess_data)
                loss = sess_metrics.get("loss", 0.0)
                total_loss += loss

                self._write_train_log(
                    session_data=sess_data,
                    conv_idx=entry["conv_idx"],
                    session_num=entry["session_num"],
                    global_session=sess_global,
                    memory_size=entry["memory_size"],
                    loss=loss,
                )

                r1_vals = [t.r1 for t in sess_data.trajectories]
                self._log(
                    f"  Update {sess_idx+1}/{S} "
                    f"(session {sess_global}): "
                    f"loss={loss:.6f}, avg_r1={np.mean(r1_vals):.3f}"
                )
            avg_update_loss = total_loss / S
            self._accumulated_sessions = []

            meta = {
                "global_session": self.global_session_count,
                "conv_idx": len(train_conversations) - 1,
                "session_num": "final",
                "loss": avg_update_loss,
                "lr": LEARNING_RATE,
                "is_final": True,
            }
            ckpt_path = save_checkpoint(
                self.model, self.tokenizer, self.optimizer,
                self.global_session_count, CHECKPOINT_DIR,
                extra_info=meta,
                memory_store=current_memory,
            )
            self._log(f"Final checkpoint saved: {ckpt_path}")

        self._log("\n" + "=" * 60)
        self._log("Training Complete!")
        self._log(f"  Total sessions processed: {self.global_session_count}")
        self._log(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
        self._log("=" * 60)
