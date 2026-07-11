"""
Reward computation: r1 (semantic consistency) + r2 (format correctness).
"""
import json
import re
from typing import Dict, List, Tuple, Set


# Valid memory operation events
VALID_EVENTS: Set[str] = {"ADD", "UPDATE", "DELETE", "NONE"}

# Required JSON keys (exactly these, no more, no less)
REQUIRED_KEYS: Set[str] = {"id", "text", "event"}


def extract_json_from_text(text: str) -> str:
    """
    Extract the first valid JSON object from model-generated text.
    Handles Qwen3 thinking mode: strips <think>...</think> blocks first.
    """
    text = text.strip()

    # Strip Qwen3 thinking block: <think>...</think>
    # The model may output reasoning in <think> tags, then the actual JSON
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):].strip()

    # Try to find JSON between { and }
    start = text.find("{")
    if start == -1:
        return ""

    # Track brace depth to find matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return ""


def check_format(action_json_str: str, existing_memory_ids: Set[str]) -> Tuple[bool, float]:
    """
    Check the format correctness of a Memory Manager output.

    Rules:
    1. Must be parseable JSON with exactly 'id', 'text', 'event' fields
    2. All values must be strings (not lists, dicts, etc.)
    3. 'event' must be one of: ADD, UPDATE, DELETE, NONE
    4. The operation must be executable:
       - ADD: requires a valid id (any non-empty string, not duplicate)
       - UPDATE: id must exist in current memory
       - DELETE: id must exist in current memory
       - NONE: id must exist in current memory

    Returns:
        (is_correct, r2_score) where r2 is 1.0 or 0.0
    """
    # Step 1: Extract JSON from text
    json_str = extract_json_from_text(action_json_str)
    if not json_str:
        return False, 0.0

    # Step 2: Parse JSON
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return False, 0.0

    if not isinstance(obj, dict):
        return False, 0.0

    # Step 3: Check exactly the required keys
    if set(obj.keys()) != REQUIRED_KEYS:
        return False, 0.0

    # Step 4: All values must be strings
    for key in REQUIRED_KEYS:
        if not isinstance(obj[key], str):
            return False, 0.0

    # Step 5: event must be valid
    event = obj["event"].strip().upper()
    if event not in VALID_EVENTS:
        return False, 0.0

    # Step 6: Check executability
    mem_id = obj["id"].strip()

    if event == "ADD":
        # For ADD: id should not already exist, text should not be empty
        if not mem_id or not obj["text"].strip():
            return False, 0.0
        # Warn but don't fail if ID already exists (model might reuse IDs)
        # Actually, ADD with existing ID can cause issues, so fail it
        if mem_id in existing_memory_ids:
            return False, 0.0

    elif event in ("UPDATE", "DELETE", "NONE"):
        # For UPDATE/DELETE/NONE: id must exist and text must not be empty
        if not mem_id:
            return False, 0.0
        if mem_id not in existing_memory_ids:
            return False, 0.0
        if event in ("UPDATE",) and not obj["text"].strip():
            return False, 0.0

    return True, 1.0


def compute_step_reward(
    action_output: str,
    existing_memory_ids: Set[str],
) -> float:
    """
    Compute r2 for a single step (memory operation output).

    Args:
        action_output: Raw text output from Memory Manager model
        existing_memory_ids: Set of current memory IDs

    Returns:
        r2 = 1.0 if format is correct, 0.0 otherwise
    """
    _, r2 = check_format(action_output, existing_memory_ids)
    return r2


def compute_trajectory_rewards(
    step_r2_list: List[float],
    r1: float,
) -> List[float]:
    """
    Compute total reward for each step in a trajectory.

    r_{t,i} = r1_i + r2_{t,i}

    Args:
        step_r2_list: List of r2 scores for each step
        r1: Semantic consistency rate (shared across all steps)

    Returns:
        List of total rewards per step: [r1 + r2_0, r1 + r2_1, ...]
    """
    return [r1 + r2 for r2 in step_r2_list]


def apply_memory_operation(
    memory_store,
    action_json_str: str,
) -> bool:
    """
    Apply a memory operation from model output to a MemoryStore.

    Args:
        memory_store: MemoryStore instance
        action_json_str: Raw JSON output from model

    Returns:
        True if operation was applied successfully, False otherwise.
    """
    json_str = extract_json_from_text(action_json_str)
    if not json_str:
        return False

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return False

    event = obj.get("event", "").strip().upper()
    mem_id = obj.get("id", "").strip()
    text = obj.get("text", "").strip()

    if event == "ADD":
        if not mem_id or not text:
            return False
        if memory_store.has_id(mem_id):
            return False
        memory_store.add(text, mem_id=mem_id)
        return True

    elif event == "UPDATE":
        if not memory_store.has_id(mem_id):
            return False
        if not text:
            return False
        memory_store.update(mem_id, text)
        return True

    elif event == "DELETE":
        if not memory_store.has_id(mem_id):
            return False
        memory_store.delete(mem_id)
        return True

    elif event == "NONE":
        if not memory_store.has_id(mem_id):
            return False
        # NONE: do nothing
        return True

    return False
