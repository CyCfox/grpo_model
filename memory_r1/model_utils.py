"""
Model loading and management using Unsloth + LoRA 4-bit.
Handles Qwen3.5-0.8B with FastLanguageModel.
"""
# CRITICAL: Unsloth must be imported before transformers, peft, torch (for patching)
import unsloth  # noqa: F401

import os
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .config import (
    MODEL_PATH, LOAD_IN_4BIT, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, MAX_SEQ_LENGTH, DEVICE
)


def load_model_and_tokenizer() -> Tuple[FastLanguageModel, any]:
    """
    Load the Qwen3.5-0.8B model with 4-bit quantization and LoRA adapters.

    Qwen3.5 uses Qwen3VLProcessor which wraps a text tokenizer and image processor.
    We extract the inner text tokenizer for text-only operations.
    Also patches for Qwen3 thinking mode.

    Returns:
        (model, tokenizer) where tokenizer is the text tokenizer (not the VL processor)
    """
    model, processor = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,           # auto-detect
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Extract the inner text tokenizer from VL processor
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def set_inference_mode(model: FastLanguageModel):
    """Switch model to fast inference mode (no gradients)."""
    FastLanguageModel.for_inference(model)


def set_training_mode(model: FastLanguageModel):
    """Switch model to training mode (gradients enabled)."""
    FastLanguageModel.for_training(model)
    model.train()


def save_checkpoint(
    model: FastLanguageModel,
    tokenizer,
    optimizer: Optional[torch.optim.Optimizer],
    global_session_count: int,
    checkpoint_dir: str,
    extra_info: Optional[dict] = None,
    memory_store=None,
):
    """
    Save a training checkpoint: LoRA adapter + optimizer state + metadata + memory bank.

    Args:
        model: The FastLanguageModel with LoRA
        tokenizer: The tokenizer
        optimizer: Current optimizer (for resuming)
        global_session_count: Global session counter for naming
        checkpoint_dir: Base checkpoint directory
        extra_info: Optional dict to save alongside checkpoint
        memory_store: Optional MemoryStore to serialize alongside checkpoint
    """
    import json
    step_dir = os.path.join(checkpoint_dir, f"step_{global_session_count:04d}")
    adapter_dir = os.path.join(step_dir, "adapter_model")
    os.makedirs(adapter_dir, exist_ok=True)

    # Save LoRA adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save optimizer state
    if optimizer is not None:
        opt_path = os.path.join(step_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), opt_path)

    # Save metadata
    if extra_info:
        meta_path = os.path.join(step_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(extra_info, f, ensure_ascii=False, indent=2)

    # Save memory bank
    if memory_store is not None:
        mem_dir = os.path.join(step_dir, "memory_bank")
        memory_store.save(mem_dir)

    # Save resume state (conv_idx, session_num for precise resume)
    resume_info = {
        "global_session": global_session_count,
        "conv_idx": extra_info.get("conv_idx") if extra_info else None,
        "session_num": extra_info.get("session_num") if extra_info else None,
    }
    resume_path = os.path.join(step_dir, "resume_state.json")
    with open(resume_path, "w", encoding="utf-8") as f:
        json.dump(resume_info, f, ensure_ascii=False, indent=2)

    # Update latest pointer
    latest_path = os.path.join(checkpoint_dir, "latest_step.txt")
    with open(latest_path, "w") as f:
        f.write(str(global_session_count))

    return step_dir


def load_checkpoint(
    model: FastLanguageModel,
    checkpoint_path: str,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load LoRA adapter weights (and optionally optimizer state) from a checkpoint.

    Args:
        model: The FastLanguageModel
        checkpoint_path: Path to step_NNNN directory
        load_optimizer: If True, load optimizer state from optimizer.pt
        optimizer: Optimizer instance to load state into (required if load_optimizer=True)

    Returns:
        dict with keys: metadata, resume_state, memory_bank_path
    """
    import json
    adapter_path = os.path.join(checkpoint_path, "adapter_model")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Checkpoint adapter not found at {adapter_path}")

    # Load adapter weights
    model.load_adapter(adapter_path, adapter_name="default")

    # Load optimizer state
    if load_optimizer and optimizer is not None:
        opt_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location="cpu")
            optimizer.load_state_dict(opt_state)

    # Load metadata
    meta_path = os.path.join(checkpoint_path, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Load resume state
    resume_path = os.path.join(checkpoint_path, "resume_state.json")
    resume_state = {}
    if os.path.exists(resume_path):
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_state = json.load(f)

    # Check for memory bank
    mem_path = os.path.join(checkpoint_path, "memory_bank")
    memory_bank_path = mem_path if os.path.isdir(mem_path) else None

    return {
        "metadata": metadata,
        "resume_state": resume_state,
        "memory_bank_path": memory_bank_path,
    }


def generate_action(
    model: FastLanguageModel,
    tokenizer,
    prompt_text: str,
    temperature: float = 1.0,
    max_new_tokens: int = 256,
) -> Tuple[str, List[int], List[float]]:
    """
    Generate a memory operation action from the model.

    Args:
        model: The FastLanguageModel (in inference mode)
        tokenizer: The tokenizer
        prompt_text: Full prompt text
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate

    Returns:
        (generated_text, token_ids, log_probs_per_token)
    """
    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LENGTH - max_new_tokens)
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Extract generated token IDs (exclude input)
    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute old log probabilities from scores
    old_log_probs = []
    if hasattr(outputs, "scores") and outputs.scores:
        for i, score in enumerate(outputs.scores):
            # score shape: (1, vocab_size)
            log_prob = F.log_softmax(score, dim=-1)
            token_id = generated_ids[i].item()
            old_log_probs.append(log_prob[0, token_id].item())

    return generated_text, generated_ids.tolist(), old_log_probs


def compute_action_log_probs(
    model: FastLanguageModel,
    tokenizer,
    prompt_text: str,
    action_token_ids: List[int],
) -> List[float]:
    """
    Compute log probabilities of action tokens given the prompt.
    Used during GRPO update to get new policy log probs.

    Args:
        model: The model (in training mode)
        tokenizer: The tokenizer
        prompt_text: The prompt text (without action)
        action_token_ids: Token IDs of the generated action

    Returns:
        List of log probabilities, one per action token (Python floats, detached)
    """
    # Tokenize prompt
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                              max_length=MAX_SEQ_LENGTH - len(action_token_ids))
    prompt_ids = prompt_inputs.input_ids[0].tolist()

    # Combine: [prompt_tokens | action_tokens]
    full_ids = prompt_ids + list(action_token_ids)
    full_ids_tensor = torch.tensor([full_ids], device=DEVICE)

    # Forward pass
    outputs = model(input_ids=full_ids_tensor)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Get log probs for action tokens only
    action_start = len(prompt_ids) - 1
    action_end = action_start + len(action_token_ids)

    action_logits = logits[0, action_start:action_end, :]  # (action_len, vocab_size)
    log_probs = F.log_softmax(action_logits, dim=-1)

    action_log_probs = []
    action_tensor = torch.tensor(action_token_ids, device=DEVICE)
    for i, token_id in enumerate(action_tensor):
        lp = log_probs[i, token_id].item()
        action_log_probs.append(lp)

    return action_log_probs


def compute_step_loss(
    model: FastLanguageModel,
    tokenizer,
    prompt_text: str,
    action_token_ids: List[int],
    old_log_probs: List[float],
    advantage: float,
) -> torch.Tensor:
    """
    Compute the token-level GRPO clipped loss for one (prompt, action) pair.
    This stays connected to the computation graph for backprop.

    Args:
        model: The model (in training mode, gradients tracked)
        tokenizer: The tokenizer
        prompt_text: The prompt text
        action_token_ids: Token IDs of the action
        old_log_probs: Log probs under old policy (detached floats)
        advantage: Group-normalized advantage for this step

    Returns:
        Loss tensor (scalar) with grad_fn connected to model parameters.
    """
    if not action_token_ids or not old_log_probs:
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)

    # Tokenize prompt
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                              max_length=MAX_SEQ_LENGTH - len(action_token_ids))
    prompt_ids = prompt_inputs.input_ids[0].tolist()

    # Combine
    full_ids = prompt_ids + list(action_token_ids)
    full_ids_tensor = torch.tensor([full_ids], device=DEVICE)

    # Forward pass (gradients tracked)
    outputs = model(input_ids=full_ids_tensor)
    logits = outputs.logits

    # Action logits positions
    action_start = len(prompt_ids) - 1
    action_end = action_start + len(action_token_ids)
    action_logits = logits[0, action_start:action_end, :]
    log_probs = F.log_softmax(action_logits, dim=-1)

    # Compute per-token clipped loss
    action_tensor = torch.tensor(action_token_ids, device=DEVICE)
    min_len = min(len(action_token_ids), len(old_log_probs))
    advantage_tensor = torch.tensor(advantage, device=DEVICE)
    step_loss = torch.tensor(0.0, device=DEVICE)

    for j in range(min_len):
        new_lp = log_probs[j, action_tensor[j]]  # stays in graph
        old_lp = old_log_probs[j]  # Python float
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        token_loss = -torch.min(ratio * advantage_tensor, clipped * advantage_tensor)
        step_loss = step_loss + token_loss

    return step_loss
