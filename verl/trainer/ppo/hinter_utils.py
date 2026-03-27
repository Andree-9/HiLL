# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility helpers for the hinter in HiLL co-training.

This module provides helper functions for:
- Formatting prompts for the hint generator
- Computing hint reward (variance / non-degenerate / transfer-weighted)
- Selecting the best hint among M candidates
- Answer-leakage detection (string-matching gate)
- Combining original questions with generated hints
"""

import math
import re
import torch
import numpy as np
from typing import Any, Optional
from verl import DataProto


# Penalty reward for failed gates (answer leakage, overlong, format error, etc.)
# Set to -0.2 to be clearly worse than any natural negative variance reward
GATE_FAILURE_PENALTY = -0.2

def extract_predicted_answer_from_response(
    response_text: str,
) -> str:
    """
    Extract the predicted answer from a model's response text.

    Looks for the content inside the last \\boxed{} in the response.

    Args:
        response_text: Decoded response text from the model

    Returns:
        Predicted answer string (content inside \\boxed{}), or empty string if not found
    """
    # Find all \boxed{...} patterns, handle nested braces
    # Use a stack-based approach for nested braces
    matches = list(re.finditer(r'\\boxed\s*\{', response_text))

    if not matches:
        return ""

    # Get the last match (models typically put final answer at the end)
    last_match = matches[-1]
    start_pos = last_match.end()  # Position right after the opening {

    # Find matching closing brace, handling nesting
    brace_count = 1
    pos = start_pos
    while pos < len(response_text) and brace_count > 0:
        if response_text[pos] == '{':
            brace_count += 1
        elif response_text[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count != 0:
        # Unbalanced braces, return everything from start to end
        return response_text[start_pos:].strip()

    # Extract content between braces (pos is one past the closing brace)
    predicted_answer = response_text[start_pos:pos-1]

    return predicted_answer


def check_answer_leaking_string_match(
    generated_text: str,
    gold_answer: str,
) -> bool:
    """
    Simple string-matching gate to detect if a generated hint leaks the gold answer.

    For short answers (<=2 chars), checks specific patterns to avoid false positives.
    For longer answers, checks direct substring containment.

    Args:
        generated_text: The hint text from the hinter
        gold_answer: The gold final answer string

    Returns:
        True if answer leaking is detected (gate FAILED), False if clean
    """
    if not gold_answer or not gold_answer.strip():
        return False  # Can't check without a valid answer

    gold_clean = gold_answer.strip()
    text_clean = generated_text.strip()

    # Always check for explicit boxed answer
    if f"\\boxed{{{gold_clean}}}" in text_clean:
        return True

    # For very short answers (1-2 chars), only flag on specific revealing patterns
    # to avoid false positives (e.g., gold="2" matching "Consider 2 variables")
    if len(gold_clean) <= 2:
        leak_patterns = [
            f"= {gold_clean}",
            f"equals {gold_clean}",
            f"is {gold_clean}",
            f"answer is {gold_clean}",
            f"answer: {gold_clean}",
            f"result is {gold_clean}",
            f"value is {gold_clean}",
            f"equal to {gold_clean}",
        ]
        # Case-insensitive check for short answers
        text_lower = text_clean.lower()
        return any(p.lower() in text_lower for p in leak_patterns)

    # For longer answers, direct substring match (case-sensitive for math expressions)
    if gold_clean in text_clean:
        return True

    # Also try case-insensitive match for word-based answers
    if gold_clean.lower() in text_clean.lower():
        return True

    return False


HINT_GENERATOR_TEMPLATE = """You are a Pedagogical Hint Generator for a Mathematical Reasoner.
TASK: GENERATE A MINIMAL HINT

OBJECTIVE
The Reasoner failed the Original Question. Your goal is to generate a minimal but useful hint that helps the Reasoner find the correct logical path.
CRITICAL: Provide a strategic nudge, not a solution. The Reasoner must still perform the deduction themselves.

INPUTS
1) Original Question:
{original_question}

2) Reasoner's Failed Attempt:
{rollout_responses}

3) Ground Truth Solution (Hidden from Reasoner; For Your Reference Only):
{ground_truth_solution}

GUIDELINES
1. Analyze the failure to identify the missing insight or misconception.
2. Provide a conceptual pointer (e.g., a relevant theorem, a structural property, an alternative representation, or an intermediate goal).
3. Do not reveal the final answer or any specific numerical values, formulas with substituted numbers, or computed intermediate steps.

OUTPUT FORMAT (Follow strictly)
<analysis>
[Briefly identify the logic breakdown and the necessary insight.]
</analysis>

<hint>
[Your concise hint — 1 to 3 sentences maximum.]
</hint>
"""

def format_prompt_for_hinter(
    original_question: str,
    ground_truth_solution: str,
    ground_truth_answer: str,
    rollout_responses: Optional[list[str]] = None,
) -> str:
    """Format the prompt for the hinter model.

    Args:
        original_question: The original question text
        ground_truth_solution: The ground truth solution (hidden from reasoner)
        ground_truth_answer: The final answer that must be preserved
        rollout_responses: List with 1 example rollout response from the reasoner

    Returns:
        Formatted prompt string for the hinter
    """
    template = HINT_GENERATOR_TEMPLATE

    # Format single rollout response
    if rollout_responses and len(rollout_responses) > 0:
        formatted_responses = rollout_responses[0]
    else:
        formatted_responses = "No response provided."

    # Fill template
    return template.format(
        original_question=original_question,
        rollout_responses=formatted_responses,
        ground_truth_solution=ground_truth_solution,
        ground_truth_answer=ground_truth_answer
    )


# HINT_APPEND_PREFIX = "Consider the following when solving the problem:\n"


def _append_hint_to_original_question(
    original_question: str,
    hint_text: str,
) -> str:
    original_question = original_question.strip()
    hint_text = hint_text.strip()
    question_with_hint = (
        f"{original_question.strip()}\n\n"
        f"{hint_text.strip()}"
    )
    return question_with_hint

def compute_hint_reward(
    seq_rewards: torch.Tensor,
    positive_threshold: float,
    parsing_succeeded: bool = True,
    semantic_passed: bool = True,
    reward_type: str = "variance",
    avg_hint_reliance: Optional[float] = None,
    transfer_temperature: float = 0.3,
) -> tuple[float, bool, float, Optional[float], float]:
    """
    Compute reward for a single hint.

    Reward types:
    - "variance": R = p̂(1-p̂)
    - "non_deg": R = 1 - p̂ⁿ - (1-p̂)ⁿ
      where n = number of rollouts. This is the probability that n i.i.d.
      Bernoulli(p̂) trials are neither all-correct nor all-incorrect.
      Zero-var (p̂=0 or 1) still gets 0, gates still penalized.
    - "transfer_weighted_variance": R = p̂(1-p̂) · exp(min(Δ_hr, 0) / T)
      where Δ_hr is avg_hint_reliance and T is transfer_temperature.
    - "transfer_weighted_non_deg": R = (1 - p̂ⁿ - (1-p̂)ⁿ) · exp(min(Δ_hr, 0) / T)
      where n = number of rollouts. The term (1 - p̂ⁿ - (1-p̂)ⁿ) is the probability
      that n i.i.d. Bernoulli(p̂) trials are neither all-correct nor all-incorrect.
      Zero-var (p̂=0 or 1) still gets 0, gates still penalized.

    Args:
        seq_rewards: Sequence rewards for N rollouts, shape [N]
        positive_threshold: Threshold for correct answer (reward > threshold)
        parsing_succeeded: Whether parsing succeeded
        semantic_passed: Whether semantic gate passed
        reward_type: "variance", "non_deg", "transfer_weighted_variance", or "transfer_weighted_non_deg"
        avg_hint_reliance: Average hint reliance across correct rollouts
            (only used when reward_type="transfer_weighted_variance" or "transfer_weighted_non_deg")
        transfer_temperature: Temperature T for transfer weight

    Returns:
        Tuple of (reward, is_zero_variance, balance_term, avg_hint_reliance_used, prob_term)
        avg_hint_reliance_used is the Δ_hr value used (None if not applicable)
        prob_term is the base probability term (variance or non-degenerate) before
        transfer weighting, used to compare against the original prompt's prob_term
    """
    # Gate failures get penalty
    if not parsing_succeeded or not semantic_passed:
        return GATE_FAILURE_PENALTY, True, 0.0, None, 0.0

    # Compute pass rate
    correct_mask = seq_rewards > positive_threshold
    p_hat = correct_mask.float().mean().item()
    balance_term = p_hat * (1.0 - p_hat)

    # Check for zero-variance
    if p_hat == 0.0 or p_hat == 1.0:
        return 0.0, True, balance_term, None, 0.0

    if reward_type == "non_deg":
        n = len(seq_rewards)
        non_deg = 1.0 - p_hat**n - (1.0 - p_hat)**n
        reward = non_deg
        prob_term = non_deg
    elif reward_type == "transfer_weighted_variance":
        prob_term = balance_term
        if avg_hint_reliance is not None:
            clamped_delta = min(avg_hint_reliance, 0.0)
            w_transfer = math.exp(clamped_delta / transfer_temperature)
            reward = balance_term * w_transfer
        else:
            reward = balance_term
    elif reward_type == "transfer_weighted_non_deg":
        n = len(seq_rewards)
        non_deg = 1.0 - p_hat**n - (1.0 - p_hat)**n
        prob_term = non_deg
        if avg_hint_reliance is not None:
            clamped_delta = min(avg_hint_reliance, 0.0)
            w_transfer = math.exp(clamped_delta / transfer_temperature)
            reward = non_deg * w_transfer
        else:
            reward = non_deg
    else:
        reward = balance_term
        prob_term = balance_term

    return reward, False, balance_term, avg_hint_reliance, prob_term


def select_best_hint(
    hinted_prompts_batches: list[DataProto],
    num_repeat: int,
    hard_prompt_indices: list[int] = None,
    original_prob_terms: Optional[dict[int, float]] = None,
    success_map: dict[tuple[int, int], bool] = None,
    timeout_map: dict[tuple[int, int], bool] = None,
    semantic_passed_map: dict[tuple[int, int], bool] = None,
    positive_threshold: float = 0.5,
    reward_type: str = "variance",
    hint_reliance_map: Optional[dict[tuple[int, int], torch.Tensor]] = None,
    transfer_temperature: float = 0.3,
) -> tuple[dict[int, dict], dict[int, list[float]], dict[int, list[Optional[float]]]]:
    """
    Select the best hinted prompt based on reward.

    Reward types:
    - "variance": R = p̂(1-p̂)
    - "non_deg": R = 1 - p̂ⁿ - (1-p̂)ⁿ
    - "transfer_weighted_variance": R = p̂(1-p̂) · exp(min(Δ_hr, 0) / T)
    - "transfer_weighted_non_deg": R = (1 - p̂ⁿ - (1-p̂)ⁿ) · exp(min(Δ_hr, 0) / T)

    Args:
        hinted_prompts_batches: List of M DataProtos with re-rollout results
        num_repeat: Number of rollouts per prompt (N)
        hard_prompt_indices: Original indices of prompts that were hinted
        original_prob_terms: Dict group_base_idx -> original prompt prob_term.
            If provided, hints whose prob_term does not improve over the
            original get hinter reward clamped to 0 (gate penalties remain).
        success_map: Dict (hint_id, prompt_idx) -> bool for parsing success
        timeout_map: Dict (hint_id, prompt_idx) -> bool for timeout penalty
        semantic_passed_map: Dict (hint_id, prompt_idx) -> bool for semantic gate
        positive_threshold: Threshold for correct answer (reward > threshold)
        reward_type: "variance", "non_deg", "transfer_weighted_variance", or "transfer_weighted_non_deg"
        hint_reliance_map: Dict (hint_id, prompt_idx) -> per-rollout hint reliance
            tensor [num_repeat]. Required for "transfer_weighted_variance" and "transfer_weighted_non_deg".
        transfer_temperature: Temperature T for transfer weight

    Returns:
        Tuple of (best_hints, all_hint_rewards, all_hint_reliance) where:
        - best_hints: Dict mapping group_base_idx -> {
            "data": DataProto with best hint's N rollouts,
            "reward": reward achieved,
            "hint_id": which hint was selected,
            "is_zero_variance": whether this hint has zero variance,
            "avg_hint_reliance": float or None (set for non-zero-var hints
              when reward_type="transfer_weighted_*"),
            "prob_term": base probability term (variance or non-deg) before
              transfer weighting, used for replacement eligibility check
          }
        - all_hint_rewards: Dict mapping group_base_idx -> list of M rewards
        - all_hint_reliance: Dict mapping group_base_idx ->
            list of M avg_hint_reliance values (None for zero-var / gate-failed)
    """
    num_hints = len(hinted_prompts_batches)
    if num_hints == 0:
        return {}, {}, {}

    batch_size_per_hint = len(hinted_prompts_batches[0])
    num_hard_prompts = batch_size_per_hint // num_repeat

    best_hints = {}
    all_hint_rewards = {}
    all_hint_reliance = {}

    for prompt_idx in range(num_hard_prompts):
        best_reward = -float('inf')
        best_batch_data = None
        best_hint_id = -1
        best_is_zero_variance = True
        best_avg_hint_reliance = None
        best_prob_term = 0.0

        # Compute group_base for storing results
        if hard_prompt_indices is not None:
            original_idx = hard_prompt_indices[prompt_idx]
            group_base = (original_idx // num_repeat) * num_repeat
        else:
            group_base = prompt_idx * num_repeat

        # Collect rewards for all hints
        hint_rewards = []
        hint_reliance_values = []

        for hint_id, hinted_batch in enumerate(hinted_prompts_batches):
            # Get the N rollouts for this prompt in this hint
            start_idx = prompt_idx * num_repeat
            end_idx = start_idx + num_repeat

            # Extract group data
            group_data = hinted_batch[start_idx:end_idx]

            # Check gates
            parsing_succeeded = success_map.get((hint_id, prompt_idx), False) if success_map else True
            has_timeout_penalty = timeout_map.get((hint_id, prompt_idx), False) if timeout_map else False
            semantic_passed = semantic_passed_map.get((hint_id, prompt_idx), True) if semantic_passed_map else True

            # Get sequence rewards
            token_level_rewards = group_data.batch["token_level_rewards"]
            response_mask = group_data.batch["response_mask"]
            seq_rewards = (token_level_rewards * response_mask).sum(dim=-1)

            # Compute avg hint reliance from CORRECT rollouts only.
            # This is a transferability proxy: how dependent successful
            # rollouts are on seeing the hint.
            avg_hint_reliance = None
            if (hint_reliance_map is not None
                    and (hint_id, prompt_idx) in hint_reliance_map):
                per_rollout_hint_reliance = hint_reliance_map[(hint_id, prompt_idx)]
                correct_mask = seq_rewards > positive_threshold
                if correct_mask.any():
                    if correct_mask.device != per_rollout_hint_reliance.device:
                        correct_mask = correct_mask.to(per_rollout_hint_reliance.device)
                    avg_hint_reliance = per_rollout_hint_reliance[correct_mask].mean().item()

            # Compute reward
            reward, is_zero_variance, balance_term, hint_reliance_used, prob_term = compute_hint_reward(
                seq_rewards=seq_rewards,
                positive_threshold=positive_threshold,
                parsing_succeeded=parsing_succeeded and not has_timeout_penalty,
                semantic_passed=semantic_passed,
                reward_type=reward_type,
                avg_hint_reliance=avg_hint_reliance,
                transfer_temperature=transfer_temperature,
            )

            # For hinter training, if the hint's probability term does not
            # improve over the original prompt's term, clamp positive reward to 0.
            # Keep negative gate penalties unchanged.
            if original_prob_terms is not None:
                original_prob = original_prob_terms.get(group_base, 0.0)
                if prob_term <= original_prob and reward > 0.0:
                    reward = 0.0

            hint_rewards.append(reward)
            hint_reliance_values.append(hint_reliance_used)

            # Select if this has higher reward
            if reward > best_reward:
                best_reward = reward
                best_batch_data = group_data
                best_hint_id = hint_id
                best_is_zero_variance = is_zero_variance
                best_avg_hint_reliance = hint_reliance_used
                best_prob_term = prob_term

        all_hint_rewards[group_base] = hint_rewards
        all_hint_reliance[group_base] = hint_reliance_values

        # Store best hint (even if zero-variance, for consistency)
        # The caller will compare against original and filter if needed
        if best_batch_data is not None:
            best_hints[group_base] = {
                "data": best_batch_data,
                "reward": best_reward,
                "hint_id": best_hint_id,
                "is_zero_variance": best_is_zero_variance,
                "avg_hint_reliance": best_avg_hint_reliance,
                "prob_term": best_prob_term,
            }

    return best_hints, all_hint_rewards, all_hint_reliance


def build_batch_with_best_hints(
    batch_with_rollouts: DataProto,
    best_hints: dict[int, dict],
    num_repeat: int,
) -> tuple[DataProto, dict[str, float]]:
    """
    Build the final batch by replacing hard prompts with their best hint.

    The caller pre-filters best_hints so that only hints whose
    prob_term exceeds the original prompt's prob_term are included.

    Strategy (simple replacement, no gap-filling):
    1. Start with the original batch
    2. For each prompt in best_hints, replace the original N rollouts
       with the best hint's N rollouts
    3. If a prompt is not in best_hints, keep original rollouts unchanged
    4. Batch size is always the same as the original

    Args:
        batch_with_rollouts: Original batch with all prompts and rollouts
        best_hints: Dict mapping group_base -> best hint info
            (already filtered to reward > ORIGINAL_REWARD by the caller)
        num_repeat: Number of rollouts per prompt (N)

    Returns:
        Tuple of (final_batch, statistics_dict)
    """
    num_original_prompts = len(batch_with_rollouts) // num_repeat

    # Build the batch: for each prompt, either use best hint or original
    batch_parts = []
    num_replaced = 0

    for prompt_idx in range(num_original_prompts):
        group_base = prompt_idx * num_repeat

        if group_base in best_hints:
            # Use the best hint's rollouts
            batch_parts.append(best_hints[group_base]["data"])
            num_replaced += 1
        else:
            # Keep original rollouts
            prompt_rollouts = batch_with_rollouts[group_base:group_base + num_repeat]
            batch_parts.append(prompt_rollouts)

    for part in batch_parts:
        if part.meta_info is not None:
            part.meta_info.pop("timing", None)
    final_batch = DataProto.concat(batch_parts)

    stats = {
        "num_original_prompts": num_original_prompts,
        "num_replaced": num_replaced,
        "num_kept_original": num_original_prompts - num_replaced,
        "final_batch_prompts": len(final_batch) // num_repeat,
    }

    return final_batch, stats


def filter_zero_advantage_groups(
    batch: DataProto,
    num_repeat: int,
    epsilon: float = 1e-6,
) -> tuple[DataProto, dict[str, float]]:
    """
    Remove prompt groups where all rollouts have zero advantages.

    All-correct and unreplaced all-incorrect groups produce zero advantages
    under GRPO, contributing no gradient but diluting the loss denominator
    and wasting forward/backward compute.

    Args:
        batch: DataProto with computed advantages and response_mask
        num_repeat: Number of rollouts per prompt (N)
        epsilon: Threshold below which summed |advantage| is treated as zero

    Returns:
        Tuple of (filtered_batch, filter_stats).
        If ALL groups are zero-advantage, returns the original batch unchanged
        to avoid an empty-batch crash.
    """
    num_prompts = len(batch) // num_repeat
    keep_indices = []
    num_filtered = 0

    for prompt_idx in range(num_prompts):
        start = prompt_idx * num_repeat
        end = start + num_repeat

        group_adv = batch.batch["advantages"][start:end]
        group_mask = batch.batch["response_mask"][start:end]
        abs_adv_sum = (group_adv.abs() * group_mask).sum().item()

        if abs_adv_sum > epsilon:
            keep_indices.extend(range(start, end))
        else:
            num_filtered += 1

    stats = {
        "zero_adv/num_prompts_original": num_prompts,
        "zero_adv/num_filtered": num_filtered,
        "zero_adv/num_prompts_actual": num_prompts - num_filtered,
        "zero_adv/filter_ratio": num_filtered / num_prompts if num_prompts > 0 else 0.0,
    }

    if not keep_indices:
        stats["zero_adv/all_filtered_fallback"] = 1.0
        return batch, stats

    filtered_batch = batch[keep_indices]
    return filtered_batch, stats


def check_hinted_prompt_length(
    original_question: str,
    tokenizer: Any,
    config: Any,
    hint_text: str = None,
) -> int:
    """
    Check the token length of a hinted prompt WITHOUT raising errors.

    Pre-filters hinted prompts that would exceed max_prompt_length,
    matching the dataset loading behavior (filter_overlong_prompts=True).

    Args:
        original_question: The original question text
        tokenizer: HuggingFace tokenizer
        config: Data config with max_prompt_length setting
        hint_text: The generated minimal hint text

    Returns:
        Token length of the formatted prompt
    """
    if hint_text is None:
        raise ValueError("Must provide hint_text")

    user_content = _append_hint_to_original_question(
        original_question=original_question,
        hint_text=hint_text,
    )

    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        **apply_chat_template_kwargs
    )

    return len(token_ids)


def format_hinted_prompt_with_chat_template(
    original_question: str,
    tokenizer: Any,
    config: Any,
    hint_text: str = None,
) -> dict[str, torch.Tensor]:
    """
    Format a hinted prompt with the same chat template as original prompts.

    User message = original question + appended hint.

    Args:
        original_question: The original question text
        tokenizer: HuggingFace tokenizer
        config: Data config with max_prompt_length, truncation settings
        hint_text: The generated minimal hint text

    Returns:
        Dictionary with tokenized inputs (input_ids, attention_mask, position_ids)
    """
    if hint_text is None:
        raise ValueError("Must provide hint_text")

    user_content = _append_hint_to_original_question(
        original_question=original_question,
        hint_text=hint_text,
    )

    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

    # Text-only case
    raw_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_chat_template_kwargs
    )
    model_inputs = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask")

    # Post-process data (same as in RLHFDataset)
    import verl.utils.torch_functional as verl_F

    max_prompt_length = config.get("max_prompt_length", 1024)
    truncation = "middle"

    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=truncation,
    )

    # Compute position IDs
    from verl.utils.model import compute_position_id_with_mask
    position_ids = compute_position_id_with_mask(attention_mask)

    return {
        "input_ids": input_ids[0],  # Remove batch dimension
        "attention_mask": attention_mask[0],
        "position_ids": position_ids[0],
    }


def batch_format_hinted_prompts(
    original_questions: list[str],
    tokenizer: Any,
    config: Any,
    hint_texts: list[Optional[str]],
    original_batch: Optional[DataProto] = None,
    original_indices: Optional[list[int]] = None,
) -> DataProto:
    """
    Format a batch of hinted prompts with proper chat templates.

    Args:
        original_questions: List of original question texts
        tokenizer: HuggingFace tokenizer
        config: Data config
        hint_texts: Per-entry generated hint text
        original_batch: Optional original batch to copy metadata from
        original_indices: Optional list of original indices for metadata copying

    Returns:
        DataProto with properly formatted and tokenized hinted prompts
    """
    formatted_prompts = []
    for i, orig_q in enumerate(original_questions):
        formatted = format_hinted_prompt_with_chat_template(
            original_question=orig_q,
            tokenizer=tokenizer,
            config=config,
            hint_text=hint_texts[i] if hint_texts else None,
        )
        formatted_prompts.append(formatted)

    # Stack into batch tensors
    input_ids = torch.stack([p["input_ids"] for p in formatted_prompts], dim=0)
    attention_mask = torch.stack([p["attention_mask"] for p in formatted_prompts], dim=0)
    position_ids = torch.stack([p["position_ids"] for p in formatted_prompts], dim=0)

    # Create DataProto
    batch_proto = DataProto.from_single_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })

    # Copy non_tensor_batch metadata from original batch
    batch_proto.non_tensor_batch = {}

    if original_batch is not None and original_indices is not None:
        indices_array = np.array(original_indices)

        # Copy all fields from original batch (question, ground_truth_solution, etc.)
        for key in original_batch.non_tensor_batch.keys():
            original_array = original_batch.non_tensor_batch[key]
            batch_proto.non_tensor_batch[key] = original_array[indices_array]

    return batch_proto
