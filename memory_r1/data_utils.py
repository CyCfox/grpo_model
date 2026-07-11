"""
Data loading and preprocessing for locomo10.json.
Parses conversations, sessions, dialogs, and maps QAs to sessions.
"""
import json
import re
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .config import (
    LOCOMO_DATA_PATH, TRAIN_CONV_INDICES, VAL_CONV_INDICES,
    TEST_CONV_INDICES, QA_PER_SESSION_MAX
)


def load_locomo_data(path: str = LOCOMO_DATA_PATH) -> List[Dict]:
    """Load the full locomo10.json dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    return data


def parse_session_key(session_key: str) -> int:
    """Extract session number from key like 'session_3' -> 3."""
    m = re.search(r"session_(\d+)", session_key)
    return int(m.group(1)) if m else 0


def parse_dia_id_session(dia_id: str) -> int:
    """Extract session number from dia_id like 'D3:5' -> 3."""
    m = re.match(r"D(\d+):", dia_id)
    return int(m.group(1)) if m else 0


def get_inner_conversation(conv_item: Dict) -> Dict:
    """
    Extract the inner conversation dict from a locomo data item.
    The outer item has keys: qa, conversation, event_summary, observation, session_summary, sample_id
    The inner 'conversation' dict has: speaker_a, speaker_b, session_1, session_1_date_time, ...
    """
    return conv_item.get("conversation", conv_item)


def get_conversation_sessions(conv_item: Dict) -> List[Tuple[int, str, List[Dict]]]:
    """
    Parse a conversation item and return sorted sessions.
    Handles both outer (full item) and inner (just 'conversation') dicts.
    Returns: list of (session_num, session_key, dialog_list)
    """
    # If this is the outer dict, extract the inner conversation
    inner = conv_item.get("conversation", conv_item)

    sessions = []
    for key in inner:
        # Match only session_N (digits only, no suffix like _summary)
        m = re.match(r"^session_(\d+)$", key)
        if m:
            session_num = int(m.group(1))
            sessions.append((session_num, key, inner[key]))
    sessions.sort(key=lambda x: x[0])
    return sessions


def get_session_datetime(conv_item: Dict, session_num: int) -> str:
    """Get the date_time string for a given session number."""
    inner = conv_item.get("conversation", conv_item)
    dt_key = f"session_{session_num}_date_time"
    return inner.get(dt_key, "")


def build_session_qa_map(conversation: Dict) -> Dict[int, List[Dict]]:
    """
    Map QAs to sessions based on evidence dia_ids.
    A QA belongs to the MAX session found in its evidence.
    Returns: {session_num: [qa_items]}
    """
    qa_list = conversation.get("qa", [])
    session_qas = defaultdict(list)

    for qa in qa_list:
        evidence = qa.get("evidence", [])
        if not evidence:
            continue
        # Find the max session ID referenced in evidence
        max_session = 0
        for ev in evidence:
            s = parse_dia_id_session(ev)
            if s > max_session:
                max_session = s
        if max_session > 0:
            session_qas[max_session].append(qa)

    return dict(session_qas)


def get_available_qas(
    session_qa_map: Dict[int, List[Dict]],
    current_session: int,
    max_count: int = QA_PER_SESSION_MAX,
    used_qa_indices: Optional[set] = None
) -> Tuple[List[Dict], set]:
    """
    Get QAs available at or before current_session.
    Returns up to max_count QAs that haven't been used yet.

    Returns: (qa_list, updated_used_indices)
    """
    if used_qa_indices is None:
        used_qa_indices = set()

    # Collect all available QAs from sessions <= current_session
    available = []
    for s in sorted(session_qa_map.keys()):
        if s <= current_session:
            for idx, qa in enumerate(session_qa_map[s]):
                global_idx = f"s{s}_{idx}"  # unique index
                if global_idx not in used_qa_indices:
                    available.append((global_idx, qa))

    # Take up to max_count, prioritizing later sessions first
    available.sort(key=lambda x: int(x[0].split("_")[0][1:]), reverse=True)
    selected = available[:max_count]

    for global_idx, _ in selected:
        used_qa_indices.add(global_idx)

    return [qa for _, qa in selected], used_qa_indices


def split_conversations(data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split locomo data into train/val/test sets."""
    train = [data[i] for i in TRAIN_CONV_INDICES if i < len(data)]
    val = [data[i] for i in VAL_CONV_INDICES if i < len(data)]
    test = [data[i] for i in TEST_CONV_INDICES if i < len(data)]
    return train, val, test


def get_speakers(conversation: Dict) -> Tuple[str, str]:
    """Get the two speaker names from a conversation."""
    return conversation.get("speaker_a", ""), conversation.get("speaker_b", "")


def format_dialog_context(dialog: Dict, date_time: str) -> str:
    """Format a single dialog turn with context."""
    speaker = dialog.get("speaker", "Unknown")
    text = dialog.get("text", "")
    dia_id = dialog.get("dia_id", "")
    return f"[{date_time}] [{dia_id}] {speaker}: {text}"


def count_total_sessions(conversations: List[Dict]) -> int:
    """Count total sessions across conversations."""
    total = 0
    for conv in conversations:
        sessions = get_conversation_sessions(conv)
        total += len(sessions)
    return total
