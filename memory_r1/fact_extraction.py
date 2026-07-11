"""
Fact extraction using DeepSeek API.
Extracts key facts from dialog turns with caching for resumption.
"""
import json
import os
import time
from typing import Optional, Dict
from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEEPSEEK_MAX_RETRIES, DEEPSEEK_TIMEOUT, EXTRACTED_FACTS_DIR
)
from .prompts import FACT_EXTRACTION_SYSTEM, FACT_EXTRACTION_USER

# Initialize DeepSeek client
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=DEEPSEEK_TIMEOUT,
        )
    return _client


def _get_cache_path(conv_idx: int, session_num: int) -> str:
    """Get the cache file path for a conversation's session facts."""
    return os.path.join(
        EXTRACTED_FACTS_DIR,
        f"conv_{conv_idx}_session_{session_num}.json"
    )


def load_cached_facts(conv_idx: int, session_num: int) -> Optional[list]:
    """Load cached facts for a session. Returns None if not cached."""
    cache_path = _get_cache_path(conv_idx, session_num)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_facts(conv_idx: int, session_num: int, facts: list):
    """Save extracted facts for a session to disk."""
    os.makedirs(EXTRACTED_FACTS_DIR, exist_ok=True)
    cache_path = _get_cache_path(conv_idx, session_num)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)


def extract_fact(
    speaker: str,
    text: str,
    date_time: str,
    conv_idx: int = 0,
) -> str:
    """
    Extract a single key fact from a dialog turn using DeepSeek API.

    Args:
        speaker: The speaker's name
        text: The dialog text
        date_time: Session date/time string
        conv_idx: Conversation index (for caching context, not used for API)

    Returns:
        Extracted fact string, or "NONE" if no substantive information.
    """
    client = get_client()

    user_prompt = FACT_EXTRACTION_USER.format(
        date_time=date_time,
        speaker=speaker,
        text=text,
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
            fact = response.choices[0].message.content.strip()
            if fact:
                return fact
            return "NONE"

        except Exception as e:
            if attempt < DEEPSEEK_MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Fact extraction failed after {DEEPSEEK_MAX_RETRIES} attempts: {e}"
                )

    return "NONE"


def extract_session_facts(
    conv_idx: int,
    session_num: int,
    dialogs: list,
    date_time: str,
    force_refresh: bool = False,
) -> list:
    """
    Extract facts for all dialog turns in a session.
    Uses disk caching to avoid redundant API calls on resume.

    Args:
        conv_idx: Conversation index
        session_num: Session number
        dialogs: List of dialog dicts with 'speaker', 'text', 'dia_id'
        date_time: Session date/time
        force_refresh: If True, ignore cache and re-extract

    Returns:
        List of fact dicts: [{dia_id, speaker, text, fact, date_time}, ...]
    """
    # Try loading from cache
    if not force_refresh:
        cached = load_cached_facts(conv_idx, session_num)
        if cached is not None:
            return cached

    facts = []
    for dialog in dialogs:
        speaker = dialog.get("speaker", "Unknown")
        text = dialog.get("text", "")
        dia_id = dialog.get("dia_id", "")

        fact_text = extract_fact(
            speaker=speaker,
            text=text,
            date_time=date_time,
            conv_idx=conv_idx,
        )

        facts.append({
            "dia_id": dia_id,
            "speaker": speaker,
            "text": text,
            "fact": fact_text,
            "date_time": date_time,
        })

        # Small delay to avoid rate limiting
        time.sleep(0.05)

    # Save to cache
    save_facts(conv_idx, session_num, facts)
    return facts
