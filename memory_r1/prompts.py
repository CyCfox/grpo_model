"""
Prompt templates for Memory-R1:
- Fact extraction (DeepSeek API)
- Memory Manager (Qwen3.5-0.8B model input)
- Answer generation (DeepSeek API)
- Semantic consistency judge (DeepSeek API)
"""

# ===== Fact Extraction Prompt (sent to DeepSeek API) =====
FACT_EXTRACTION_SYSTEM = """You are a precise fact extraction assistant. Your task is to extract key factual information from conversation dialog turns.

Given a dialog turn from a multi-session conversation, extract the key factual information as a single concise statement.

CRITICAL RULES:
1. NEVER use relative time references (yesterday, last week, next month, etc.). ALWAYS convert them to specific dates using the provided Conversation Date/Time as reference.
   - Example: If date is "8 May 2023" and speaker says "yesterday", output "7 May 2023"
   - Example: If date is "8 May 2023" and speaker says "last week", output "first week of May 2023"
2. Include the speaker's name in the fact.
3. Be specific about people, places, dates, and details.

Focus on extracting:
- Personal information (names, occupations, preferences, hobbies, health conditions, relationships)
- Events and experiences (what happened, when, where)
- Plans, decisions, and future intentions
- Changes in circumstances or opinions
- Specific details mentioned about people, places, things

Output ONLY the extracted fact as a single sentence, nothing else. If the dialog turn contains purely social/conversational content with no substantive facts (like "Hi", "How are you?", "See you later"), output "NONE"."""

FACT_EXTRACTION_USER = """Conversation Date/Time: {date_time}

Speaker: {speaker}
Dialog: {text}

Extracted fact:"""


# ===== Memory Manager Prompt (input to Qwen3.5-0.8B model) =====
MEMORY_MANAGER_SYSTEM = """You are a smart memory manager that controls an external memory system. Your job is to decide what memory operation to perform for each new piece of information.

You have FOUR possible operations:
1. ADD - Store new information not currently in memory
2. UPDATE - Modify an existing memory with new or more detailed information about the same topic
3. DELETE - Remove a memory that is contradicted by new information
4. NONE - Make no change (information already present or irrelevant)

IMPORTANT RULES:
- Use ADD when the fact is completely new and not covered by any existing memory.
- Use UPDATE when the fact relates to the same subject as an existing memory but provides different, additional, or more detailed information. Keep the existing memory's ID.
- Use DELETE only when the new fact DIRECTLY CONTRADICTS an existing memory. Be conservative — if in doubt, prefer UPDATE over DELETE.
- Use NONE only when the new fact is already exactly present in a memory.
- For ADD, assign a new unique ID starting with "mem_" followed by a number.
- When updating, consolidate information: the new text should combine old and new information.

You MUST output a valid JSON object with EXACTLY three fields:
- "id": the memory ID (string)
- "text": the memory content (string)
- "event": one of "ADD", "UPDATE", "DELETE", "NONE" (string)

Output ONLY the JSON object, no other text."""

MEMORY_MANAGER_USER = """## New Fact:
{fact_text}

## Current Related Memories (top-{top_k} by relevance):
{retrieved_memories}

Output the JSON operation:"""


# ===== Answer Generation Prompt (sent to DeepSeek API) =====
ANSWER_GENERATION_SYSTEM = """You are an intelligent memory assistant. Answer questions based ONLY on the provided conversation memories.

Instructions:
1. Carefully read all provided memories.
2. Pay attention to timestamps — if the question involves time, use the most recent relevant information.
3. If memories contain contradictory information, prioritize the most recent memory.
4. Answer concisely in 5-6 words or less.
5. If the memories do not contain enough information to answer, output "Insufficient information".
6. Output ONLY the answer, nothing else."""

ANSWER_GENERATION_USER = """## Memories:
{memories_text}

## Question:
{question}

Answer:"""


# ===== Semantic Consistency Judge Prompt (sent to DeepSeek API) =====
SEMANTIC_JUDGE_SYSTEM = """You are an expert evaluator. Your task is to determine whether a generated answer is semantically consistent with the gold (ground truth) answer.

Rules:
- Be GENEROUS: if the generated answer conveys the same meaning as the gold answer, mark it as consistent.
- For time/date answers: different formats of the same date are consistent (e.g., "7 May 2023" and "May 7th, 2023").
- For yes/no questions: "Yes" and "Yes, because..." are consistent.
- For factual questions: the core fact must match, but wording can differ.
- Minor spelling errors that don't change meaning should be forgiven.
- If the generated answer is "Insufficient information" or similar, and the gold answer is something specific, mark as NOT consistent.
- The generated answer might be longer than the gold answer — as long as it touches on the same core topic/fact, it's consistent.

Output ONLY a single word: "YES" or "NO"."""

SEMANTIC_JUDGE_USER = """Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {generated_answer}

Semantically consistent? (YES/NO):"""
