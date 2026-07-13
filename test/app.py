#!/usr/bin/env python
"""
Memory-R1 Inference Demo — Flask backend.

Three-column visual demo:
  Left:   Chat with DeepSeek API
  Middle: Memory operations from trained model
  Right:  Current memory bank

Usage:
    python test/app.py
    Then open http://localhost:5000
"""
# CRITICAL: Unsloth must be imported before transformers, peft, torch
import unsloth  # noqa: F401

import sys
import os
import re
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# Add project root so memory_r1 imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_r1.config import (
    MODEL_PATH, CHECKPOINT_DIR, LOG_DIR,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEEPSEEK_MAX_RETRIES, DEEPSEEK_TIMEOUT,
    TOP_K_RETRIEVAL, MAX_NEW_TOKENS, MAX_SEQ_LENGTH, DEVICE,
    EMBEDDING_MODEL_PATH, EMBEDDING_DEVICE,
)
from memory_r1.model_utils import (
    load_model_and_tokenizer, set_inference_mode, generate_action,
)
from memory_r1.memory_store import MemoryStore, get_embedding_model
from memory_r1.reward import extract_json_from_text, apply_memory_operation, compute_step_reward
from memory_r1.prompts import (
    FACT_EXTRACTION_SYSTEM, FACT_EXTRACTION_USER,
    MEMORY_MANAGER_SYSTEM, MEMORY_MANAGER_USER,
)
from memory_r1.grpo_trainer import build_memory_manager_prompt

# ===== Paths =====
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TEST_DIR, "data")
DIALOGS_PATH = os.path.join(DATA_DIR, "dialogs.jsonl")
OPERATES_PATH = os.path.join(DATA_DIR, "operates.jsonl")
MEMORY_JSON_PATH = os.path.join(DATA_DIR, "memory.json")
MEMORY_STORE_DIR = os.path.join(DATA_DIR, "memory_store")

os.makedirs(DATA_DIR, exist_ok=True)

# ===== Flask App =====
app = Flask(__name__)

# ===== DeepSeek Client =====
_ds_client: Optional[OpenAI] = None

def get_ds_client() -> OpenAI:
    global _ds_client
    if _ds_client is None:
        _ds_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=DEEPSEEK_TIMEOUT,
        )
    return _ds_client

# ===== Chat System Prompt =====
CHAT_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant engaged in a natural conversation. "
    "Respond concisely and naturally. Keep answers to 2-4 sentences unless "
    "the user asks for more detail."
)

# ===== Global State (initialized in startup) =====
_model = None
_tokenizer = None
_memory_store: Optional[MemoryStore] = None
_conversation_history: List[Dict] = []  # [{role, content, timestamp}, ...]

# ===== Helpers =====

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _save_dialogs():
    """Append latest turn to dialogs.jsonl (locomo10-like format)."""
    with open(DIALOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(_conversation_history, ensure_ascii=False) + "\n")

def _save_operates(entry: dict):
    """Append one operation record to operates.jsonl."""
    with open(OPERATES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def _save_memory():
    """Save memory bank to disk (full state for vector search + meta for frontend)."""
    if _memory_store is not None:
        _memory_store.save(MEMORY_STORE_DIR)
        # Also save meta-only JSON for frontend display
        with open(MEMORY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(_memory_store.get_all_memories(), f, ensure_ascii=False, indent=2)

def _load_memory():
    """Load memory bank from disk, or create fresh."""
    global _memory_store
    if os.path.isdir(MEMORY_STORE_DIR):
        emb = get_embedding_model()
        _memory_store = MemoryStore.load_state(MEMORY_STORE_DIR, embedding_model=emb)
        print(f"[{_now()}] Loaded memory bank: {len(_memory_store)} entries")
    else:
        emb = get_embedding_model()
        _memory_store = MemoryStore(embedding_model=emb)
        print(f"[{_now()}] Created fresh memory bank")

def _load_dialogs():
    """Load conversation history from dialogs.jsonl on startup."""
    global _conversation_history
    if os.path.exists(DIALOGS_PATH):
        with open(DIALOGS_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines:
            # Last line is the full conversation array (most recent state)
            try:
                _conversation_history = json.loads(lines[-1].strip())
                print(f"[{_now()}] Loaded {len(_conversation_history)} previous messages")
            except json.JSONDecodeError:
                _conversation_history = []

def _get_chat_history_for_api(max_turns: int = 20) -> List[Dict]:
    """Return recent conversation messages for DeepSeek API context."""
    recent = _conversation_history[-max_turns * 2:]  # user+assistant pairs
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return messages

def _call_deepseek_chat(user_message: str) -> str:
    """Send user message to DeepSeek and return the assistant's reply."""
    client = get_ds_client()
    messages = _get_chat_history_for_api()
    messages.append({"role": "user", "content": user_message})

    for attempt in range(DEEPSEEK_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < DEEPSEEK_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"DeepSeek chat failed: {e}")

def _extract_fact(speaker: str, text: str, date_time: str) -> str:
    """Extract a fact from a single dialog turn using DeepSeek."""
    client = get_ds_client()
    user_prompt = FACT_EXTRACTION_USER.format(
        date_time=date_time, speaker=speaker, text=text,
    )

    for attempt in range(DEEPSEEK_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": FACT_EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            result = response.choices[0].message.content.strip()
            return result if result else "NONE"
        except Exception as e:
            if attempt < DEEPSEEK_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[{_now()}] Fact extraction failed: {e}")
                return "NONE"

def _run_memory_operation(fact_text: str) -> dict:
    """
    Feed a fact to the trained Memory Manager model.
    Returns {raw_text, parsed_json, event, applied, r2}.
    """
    global _model, _tokenizer, _memory_store

    if fact_text == "NONE" or not fact_text.strip():
        return {
            "raw_text": "",
            "parsed_json": None,
            "event": "NONE",
            "applied": True,
            "r2": 1.0,
        }

    existing_ids = {m["id"] for m in _memory_store.get_all_memories()}

    # Retrieve top-K similar memories
    retrieved = _memory_store.search(fact_text, top_k=TOP_K_RETRIEVAL)

    # Build prompt
    prompt = build_memory_manager_prompt(fact_text, retrieved, _tokenizer)

    # Generate single action (greedy — inference, not training)
    action_text, action_ids, _ = generate_action(
        _model, _tokenizer, prompt,
        temperature=0.0,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # Try to parse the JSON
    json_str = extract_json_from_text(action_text)
    parsed = None
    event = "NONE"
    try:
        if json_str:
            parsed = json.loads(json_str)
            event = parsed.get("event", "NONE").strip().upper()
    except (json.JSONDecodeError, AttributeError):
        pass

    # Apply operation
    applied = apply_memory_operation(_memory_store, action_text)

    # Compute r2 for this operation
    r2 = compute_step_reward(action_text, existing_ids)

    return {
        "raw_text": action_text,
        "parsed_json": parsed,
        "event": event,
        "applied": applied,
        "r2": r2,
    }

# ===== API Endpoints =====

@app.route("/")
def index():
    """Serve the frontend page."""
    templates_dir = os.path.join(TEST_DIR, "templates")
    return send_from_directory(templates_dir, "index.html")

@app.route("/memory", methods=["GET"])
def get_memory():
    """Return current memory bank as JSON."""
    global _memory_store
    if _memory_store is None:
        return jsonify({"memory": []})
    return jsonify({
        "memory": _memory_store.get_all_memories(),
        "count": len(_memory_store),
    })

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main endpoint: receive user message, return chat reply + fact + operation + memory.
    Body: {"message": "..."}
    """
    global _conversation_history, _memory_store

    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    timestamp = _now()

    # Step 1: Get chat reply from DeepSeek
    try:
        reply = _call_deepseek_chat(user_message)
        print(f"[{timestamp}] User: {user_message[:50]}...")
        print(f"[{timestamp}] AI:   {reply[:50]}...")
    except Exception as e:
        return jsonify({"error": f"Chat API failed: {e}"}), 500

    # Step 2: Append to conversation history
    _conversation_history.append({
        "role": "user", "content": user_message, "timestamp": timestamp,
    })
    _conversation_history.append({
        "role": "assistant", "content": reply, "timestamp": _now(),
    })
    _save_dialogs()

    # Step 3: Extract fact from user message
    date_time = timestamp
    try:
        fact = _extract_fact("User", user_message, date_time)
    except Exception:
        fact = "NONE"
    print(f"[{timestamp}] Fact: {fact[:80] if fact != 'NONE' else 'NONE'}")

    # Step 4: Run memory operation via trained model
    operation = _run_memory_operation(fact)

    print(f"[{timestamp}] Op: {operation['event']} r2={operation['r2']} "
          f"applied={operation['applied']}")

    # Step 5: Save memory state
    _save_memory()

    # Step 6: Log operation
    op_log = {
        "timestamp": timestamp,
        "user_message": user_message,
        "fact": fact,
        "operation": operation,
        "memory_size": len(_memory_store),
    }
    _save_operates(op_log)

    return jsonify({
        "reply": reply,
        "fact": fact,
        "operation": operation,
        "memory": _memory_store.get_all_memories(),
        "memory_size": len(_memory_store),
    })

@app.route("/reset", methods=["POST"])
def reset():
    """Clear conversation history and memory bank."""
    global _conversation_history, _memory_store

    _conversation_history = []
    _memory_store = MemoryStore(embedding_model=get_embedding_model())
    _save_memory()
    _save_dialogs()

    # Clear operates log
    if os.path.exists(OPERATES_PATH):
        os.remove(OPERATES_PATH)

    print(f"[{_now()}] Memory bank and history reset.")
    return jsonify({"status": "ok", "memory": []})

# ===== Startup =====

def _list_checkpoints() -> List[str]:
    """List available checkpoint step numbers."""
    ckpts = []
    if not os.path.isdir(CHECKPOINT_DIR):
        return ckpts
    for name in os.listdir(CHECKPOINT_DIR):
        m = re.match(r"step_(\d{4})$", name)
        if m:
            adapter = os.path.join(CHECKPOINT_DIR, name, "adapter_model")
            if os.path.isdir(adapter):
                ckpts.append(str(int(m.group(1))))
    ckpts.sort(key=int)
    return ckpts

def resolve_checkpoint_path(checkpoint_spec: str) -> Optional[str]:
    """
    Resolve a checkpoint specification to an adapter_model path.
    Accepts: step number ("128"), full path, "latest", "base"/"none", or basename ("step_0128").
    Returns the adapter path, or None for base model.
    """
    spec = checkpoint_spec.strip()
    available = _list_checkpoints()

    if spec.lower() in ("base", "none"):
        return None
    if spec.lower() == "latest":
        if not available:
            print(f"[{_now()}] WARNING: no checkpoints found, using base model")
            return None
        spec = available[-1]

    # Try as bare step number
    m = re.match(r"^(\d+)$", spec)
    if m:
        ckpt_dir = os.path.join(CHECKPOINT_DIR, f"step_{int(m.group(1)):04d}")
        adapter = os.path.join(ckpt_dir, "adapter_model")
        if os.path.isdir(adapter):
            return adapter
        raise FileNotFoundError(f"Checkpoint step {spec} not found. Available: {available}")

    # Try as step_NNNN basename
    m = re.match(r"^step_(\d{4})$", spec)
    if m:
        ckpt_dir = os.path.join(CHECKPOINT_DIR, spec)
        adapter = os.path.join(ckpt_dir, "adapter_model")
        if os.path.isdir(adapter):
            return adapter
        raise FileNotFoundError(f"Checkpoint {spec} not found. Available: {available}")

    # Try as full path
    if os.path.isdir(spec):
        adapter = os.path.join(spec, "adapter_model")
        if os.path.isdir(adapter):
            return adapter
        raise FileNotFoundError(f"No adapter_model in {spec}")

    raise FileNotFoundError(
        f"Invalid checkpoint spec: {spec}. "
        f"Use step number, 'latest', 'base', or full path. Available: {available}"
    )

def initialize(checkpoint: str = "latest"):
    """Load models and restore state. Called once at startup."""
    global _model, _tokenizer

    print(f"[{_now()}] Initializing Memory-R1 Inference Demo...")
    print(f"[{_now()}] Data dir: {DATA_DIR}")

    # 1. Load embedding model (CPU)
    print(f"[{_now()}] Loading embedding model (CPU)...")
    emb = get_embedding_model()
    print(f"[{_now()}]   Embedding model ready.")

    # 2. Load main model (GPU, 4-bit LoRA)
    print(f"[{_now()}] Loading Qwen3.5-0.8B (4-bit LoRA)...")
    _model, _tokenizer = load_model_and_tokenizer()

    # 3. Load specified checkpoint (or base model)
    adapter_path = resolve_checkpoint_path(checkpoint)
    if adapter_path:
        _model.load_adapter(adapter_path, adapter_name="default")
        print(f"[{_now()}]   Loaded checkpoint: {os.path.dirname(adapter_path)}")
    else:
        print(f"[{_now()}]   Using base model (no checkpoint loaded)")

    # 4. Set inference mode
    set_inference_mode(_model)
    print(f"[{_now()}]   Model ready (inference mode).")

    # 5. Load or create memory bank
    _load_memory()

    # 6. Load conversation history
    _load_dialogs()

    print(f"[{_now()}] Initialization complete. Starting server...")
    print(f"[{_now()}]   Memory entries: {len(_memory_store)}")
    print(f"[{_now()}]   Conversation turns: {len(_conversation_history) // 2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-R1 Inference Demo")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default="latest",
        help="Checkpoint to use: step number (128), 'latest', 'base' (no LoRA), "
             "'step_0128', or full path to checkpoint dir. Default: latest.",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available checkpoints and exit.",
    )
    args = parser.parse_args()

    if args.list:
        available = _list_checkpoints()
        if available:
            print("Available checkpoints:")
            for c in available:
                print(f"  step_{int(c):04d}  (step {c})")
        else:
            print("No checkpoints found.")
        sys.exit(0)

    initialize(checkpoint=args.checkpoint)
    app.run(host="0.0.0.0", port=5000, debug=False)
