"""
Answer generation and semantic consistency judgment using DeepSeek API.
"""
import time
from typing import List, Dict, Tuple
from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEEPSEEK_MAX_RETRIES, DEEPSEEK_TIMEOUT
)
from .prompts import (
    ANSWER_GENERATION_SYSTEM, ANSWER_GENERATION_USER,
    SEMANTIC_JUDGE_SYSTEM, SEMANTIC_JUDGE_USER,
)

_client: OpenAI = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=DEEPSEEK_TIMEOUT,
        )
    return _client


def _format_memories(memories: List[Dict[str, str]]) -> str:
    """Format a list of memory dicts into a readable string."""
    if not memories:
        return "(No relevant memories found)"

    lines = []
    for mem in memories:
        lines.append(f"- [{mem.get('id', '?')}] {mem.get('text', '')}")
    return "\n".join(lines)


def generate_answer(
    question: str,
    memories: List[Dict[str, str]],
) -> str:
    """
    Generate an answer to a question based on retrieved memories.

    Args:
        question: The question to answer
        memories: List of memory dicts from retrieval

    Returns:
        Generated answer string (5-6 words or less).
    """
    client = get_client()
    memories_text = _format_memories(memories)

    user_prompt = ANSWER_GENERATION_USER.format(
        memories_text=memories_text,
        question=question,
    )

    for attempt in range(DEEPSEEK_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": ANSWER_GENERATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            answer = response.choices[0].message.content.strip()
            return answer or "Insufficient information"

        except Exception as e:
            if attempt < DEEPSEEK_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"Answer generation failed after {DEEPSEEK_MAX_RETRIES} attempts: {e}"
                )

    return "Insufficient information"


def judge_semantic_consistency(
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> bool:
    """
    Judge whether the generated answer is semantically consistent with the gold answer.

    Args:
        question: The original question
        gold_answer: The ground truth answer
        generated_answer: The model-generated answer (via DeepSeek)

    Returns:
        True if semantically consistent, False otherwise.
    """
    client = get_client()

    user_prompt = SEMANTIC_JUDGE_USER.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    for attempt in range(DEEPSEEK_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SEMANTIC_JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            result = response.choices[0].message.content.strip().upper()
            return result.startswith("YES")

        except Exception as e:
            if attempt < DEEPSEEK_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"Semantic judgment failed after {DEEPSEEK_MAX_RETRIES} attempts: {e}"
                )

    return False


def evaluate_trajectory_answers(
    qa_list: List[Dict],
    memory_store,
    top_k: int = 10,
) -> Tuple[List[Dict], float]:
    """
    For a trajectory, generate answers for all QAs and compute r1.
    For each QA, retrieves top-K memories using the question as query.

    Args:
        qa_list: List of QA dicts with 'question' and 'answer' keys
        memory_store: MemoryStore instance for retrieval
        top_k: Number of memories to retrieve per QA

    Returns:
        (detailed_results, r1_score)
        detailed_results: [{question, gold_answer, generated_answer, is_consistent}, ...]
        r1_score: semantic_consistent_count / total_count
    """
    results = []
    consistent_count = 0

    for qa in qa_list:
        question = qa.get("question", "")
        gold_answer = qa.get("answer", "")

        if not question or not gold_answer:
            continue

        # Retrieve top-K memories relevant to this question
        retrieved = memory_store.search(question, top_k=top_k)

        # Generate answer using retrieved memories
        generated = generate_answer(question, retrieved)

        # Judge semantic consistency
        is_consistent = judge_semantic_consistency(question, gold_answer, generated)

        if is_consistent:
            consistent_count += 1

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated,
            "is_consistent": is_consistent,
        })

    total = len(results)
    r1 = consistent_count / total if total > 0 else 0.0

    return results, r1
