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
HiLL Trainer: Reasoner-First Co-Training with Hint Generation.

This trainer implements the reasoner-first workflow where:
1. Reasoner rollouts on original prompts first
2. Identify hard prompts (all-incorrect, or low-correct when configured)
3. Hint generator produces M candidate hints for each hard prompt
4. Reasoner re-rollouts on hinted prompts (original question + hint)
5. Select best hint whose prob_term exceeds the original's
6. Update reasoner on mixed batch (original + best-hinted)
7. Update hint generator using all hints with variance-based rewards
"""

import os
import re
import json
import random
import uuid
import torch
import numpy as np
from typing import Optional
from datetime import datetime

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, compute_response_mask, compute_prompt_statistics
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.hinter_utils import (
    check_hinted_prompt_length,
    batch_format_hinted_prompts,
    select_best_hint,
    build_batch_with_best_hints,
    filter_zero_advantage_groups,
    format_prompt_for_hinter,
    # String-matching answer leakage gate
    check_answer_leaking_string_match,
)
from verl.utils.debug import marked_timer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


class HiLLRayPPOTrainer(RayPPOTrainer):
    """
    HiLL Trainer with Reasoner-First Hint-Generation Workflow.

    Workflow:
    1. Reasoner rollouts N times on original P prompts
    2. Identify hard prompts (all-incorrect)
    3. Hint generator produces M candidate hints for each hard prompt
    4. Reasoner re-rollouts N times on each hinted prompt (question + hint)
    5. Select best hint (by prob_term improvement) to replace original
    6. Update reasoner on mixed batch (original + best-hinted)
    7. Update hint generator using all hints with variance-based rewards
    """
    
    # Threshold for classifying correct/incorrect (reward > threshold = correct)
    POSITIVE_THRESHOLD = 0.5

    def __init__(self, *args, hinter_tokenizer=None, **kwargs):
        """Initialize HiLL trainer."""
        super().__init__(*args, **kwargs)

        # Number of hints per prompt
        self.num_hints = self.config.hinter.get("num_hints", 4)
        rollout_hints = self.config.hinter.rollout.get("n", self.num_hints)
        if rollout_hints != self.num_hints:
            raise ValueError(
                "hinter.num_hints must match hinter.rollout.n "
                f"(got {self.num_hints} vs {rollout_hints})"
            )

        # Minimum number of hard prompts to trigger hint generation
        self.min_hard_prompts = self.config.hinter.get("min_zero_var_prompts", 1)

        # Inspection logging frequency
        self.inspection_log_freq = self.config.hinter.get("inspection_log_freq", 50)

        # Hinter reward type: "variance", "non_deg", "transfer_weighted_variance", or "transfer_weighted_non_deg"
        self.hinter_reward_type = self.config.hinter.get("hinter_reward_type", "variance")
        assert self.hinter_reward_type in ("variance", "non_deg", "transfer_weighted_variance", "transfer_weighted_non_deg"), (
            f"Unknown hinter_reward_type: {self.hinter_reward_type}"
        )
        # Temperature for transfer weight: w = exp(min(Δ_dd, 0) / T)
        self.transfer_temperature = self.config.hinter.get("transfer_temperature", 0.3)

        # Store tokenizers
        self.hinter_tokenizer = hinter_tokenizer

        if self.hinter_tokenizer is None:
            raise ValueError(
                "hinter_tokenizer must be provided"
            )

        # Worker groups (initialized in init_workers)
        self.hinter_wg = None
    
    def init_workers(self):
        """Initialize workers including hinter."""
        # Create resource pools first
        self.resource_pool_manager.create_resource_pool()
        
        # Initialize resource_pool_to_cls mapping
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        # Check if hinter is colocated with actor_rollout
        hinter_colocated = False
        hinter_cls = None
        hinter_resource_pool = None
        
        if Role.Hinter in self.role_worker_mapping:
            hinter_pool = self.resource_pool_manager.get_resource_pool(Role.Hinter)
            actor_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            hinter_colocated = (hinter_pool == actor_pool)
            hinter_resource_pool = hinter_pool
            
            hinter_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Hinter],
                config=self.config.hinter,
                role="hinter",
            )
            
            if not hinter_colocated:
                self.resource_pool_to_cls[hinter_pool]["hinter"] = hinter_cls
                print(f"[HiLL] Hinter will use SEPARATE resource pool")
            else:
                print(f"[HiLL] Hinter will be COLOCATED in separate process")
        
        # Create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # Create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # Create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # Create reward model if needed
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # Create worker groups
        from omegaconf import OmegaConf
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
        wg_kwargs["device_name"] = self.device_name

        from verl.single_controller.ray.base import create_colocated_worker_cls
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # Create hinter as separate worker group when colocated
        if hinter_colocated and hinter_cls is not None:
            hinter_class_dict = {"hinter": hinter_cls}
            worker_dict_cls = create_colocated_worker_cls(class_dict=hinter_class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=hinter_resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=hinter_class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        
        # Initialize hinter
        if Role.Hinter in self.role_worker_mapping:
            self.hinter_wg = all_wg.get("hinter")
            if self.hinter_wg:
                self.hinter_wg.init_model()
                print(f"[HiLL] Initialized hint generator")
                print(f"  - Number of hints per prompt: {self.num_hints}")

        # Create async rollout manager if needed
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )
    
    def _prepare_hinter_batch(
        self,
        batch_with_rollouts: DataProto,
        hard_prompt_indices: list[int],
        num_repeat: int,
    ) -> tuple[DataProto, list[str], list[str], list[str]]:
        """
        Prepare hint-generator input batch for hard prompts.
        
        Returns:
            Tuple of (hinter_batch, gold_answers, original_questions, ground_truth_solutions)
        """
        hinter_inputs = []
        gold_answers = []
        original_questions = []
        ground_truth_solutions = []
        
        # Format hinter prompts for each hard prompt
        for prompt_idx in hard_prompt_indices:
            # Direct field access from sage_data
            question = str(batch_with_rollouts.non_tensor_batch["problem"][prompt_idx])
            ground_truth = str(batch_with_rollouts.non_tensor_batch["solution"][prompt_idx])
            gold_answer = str(batch_with_rollouts.non_tensor_batch["answer"][prompt_idx])
            
            # Extract 1 random rollout response as example for the hinter
            group_base = (prompt_idx // num_repeat) * num_repeat
            selected_idx = random.randint(group_base, group_base + num_repeat - 1)
            
            resp_tokens = batch_with_rollouts.batch["responses"][selected_idx]
            rollout_response = self.tokenizer.decode(resp_tokens, skip_special_tokens=True)
            rollout_responses = [rollout_response]
            
            original_questions.append(question)
            gold_answers.append(gold_answer)
            ground_truth_solutions.append(ground_truth)
            
            hinter_input = format_prompt_for_hinter(
                original_question=question,
                ground_truth_solution=ground_truth,
                ground_truth_answer=gold_answer,
                rollout_responses=rollout_responses,
            )
            hinter_inputs.append(hinter_input)
        
        # Apply chat template and tokenize
        hinter_prompts_formatted = []
        for hinter_input in hinter_inputs:
            messages = [{"role": "user", "content": hinter_input}]
            # Qwen3 tokenizers support enable_thinking; older tokenizers do not.
            try:
                formatted_prompt = self.hinter_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                formatted_prompt = self.hinter_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            hinter_prompts_formatted.append(formatted_prompt)
        
        tokenized_prompts = []
        for hinter_prompt in hinter_prompts_formatted:
            model_inputs = self.hinter_tokenizer(
                hinter_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            
            input_ids_single, attention_mask_single = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=self.config.hinter.rollout.prompt_length,
                pad_token_id=self.hinter_tokenizer.pad_token_id,
                left_pad=True,
                truncation="middle",
            )
            
            position_ids_single = compute_position_id_with_mask(attention_mask_single)
            
            tokenized_prompts.append({
                "input_ids": input_ids_single[0],
                "attention_mask": attention_mask_single[0],
                "position_ids": position_ids_single[0],
            })
        
        input_ids = torch.stack([p["input_ids"] for p in tokenized_prompts], dim=0)
        attention_mask = torch.stack([p["attention_mask"] for p in tokenized_prompts], dim=0)
        position_ids = torch.stack([p["position_ids"] for p in tokenized_prompts], dim=0)
        
        hinter_batch = DataProto.from_single_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })
        
        hinter_batch.non_tensor_batch = {
            "gold_answers": np.array(gold_answers, dtype=object),
            "original_questions": np.array(original_questions, dtype=object),
            "ground_truth_solutions": np.array(ground_truth_solutions, dtype=object),
            "original_prompt_indices": np.array(hard_prompt_indices, dtype=np.int64),
        }
        
        return hinter_batch, gold_answers, original_questions, ground_truth_solutions
    
    
    def _training_step_reasoner_first(
        self,
        batch: DataProto,
        timing_raw: dict,
        metrics: dict,
    ) -> tuple[DataProto, Optional[dict]]:
        """
        Execute one training step with reasoner-first workflow.
        
        Workflow:
        1. Rollout on original prompts
        2. Compute rewards and identify hard prompts
        3. Hint generator produces M candidate hints for hard prompts
        4. Reasoner re-rollouts on hinted prompts
        5. Select best hint to replace hard prompts
        6. Return mixed batch for reasoner training
        """
        num_repeat = self.config.actor_rollout_ref.rollout.n
        num_prompts = len(batch) // num_repeat if hasattr(batch, 'batch') else len(batch)
        hinter_update_state = None
        
        # ===== STEP 1: Initial rollout on original prompts =====
        with marked_timer("gen_original", timing_raw, color="red"):
            gen_batch = self._get_gen_batch(batch)
            gen_batch = gen_batch.repeat(repeat_times=num_repeat, interleave=True)
            
            if not self.async_rollout_mode:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            else:
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
            
            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
            # Avoid concat conflicts when
            # combining batches from different generation calls.
            gen_batch_output.meta_info.pop("timing", None)
        
        # Merge prompts with responses
        batch_with_rollouts = batch.repeat(repeat_times=num_repeat, interleave=True)
        batch_with_rollouts = batch_with_rollouts.union(gen_batch_output)
        
        if "response_mask" not in batch_with_rollouts.batch:
            batch_with_rollouts.batch["response_mask"] = compute_response_mask(batch_with_rollouts)
        
        # ===== STEP 2: Compute rewards =====
        with marked_timer("reward_original", timing_raw, color="yellow"):
            if self.use_rm and "rm_scores" not in batch_with_rollouts.batch:
                reward_tensor = self.rm_wg.compute_rm_score(batch_with_rollouts)
                batch_with_rollouts = batch_with_rollouts.union(reward_tensor)
            
            if self.reward_fn is not None:
                reward_tensor, reward_extra_infos = compute_reward(batch_with_rollouts, self.reward_fn)
                batch_with_rollouts.batch["token_level_scores"] = reward_tensor
                batch_with_rollouts.batch["token_level_rewards"] = reward_tensor
                
                if reward_extra_infos:
                    batch_with_rollouts.non_tensor_batch.update(
                        {k: np.array(v) for k, v in reward_extra_infos.items()}
                    )
        
        # Compute advantages for original batch
        with marked_timer("adv_original", timing_raw, color="brown"):
            batch_with_rollouts = compute_advantage(
                batch_with_rollouts,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=num_repeat,
                config=self.config.algorithm,
            )
        
        # ===== STEP 3: Identify hard prompts for hint generation =====
        with marked_timer("identify_zero_variance", timing_raw, color="cyan"):
            prompt_stats_original, zero_variance_uids, _, all_incorrect_uids = compute_prompt_statistics(
                data=batch_with_rollouts,
                num_repeat=num_repeat,
                positive_threshold=self.POSITIVE_THRESHOLD,
            )
            
            # Log original stats (before evolution)
            metrics["zero_adv/prompts_zero_advantage_ratio_original"] = prompt_stats_original["zero_adv/prompts_zero_advantage_ratio"]
            metrics["zero_adv/prompts_all_correct_ratio_original"] = prompt_stats_original["zero_adv/prompts_all_correct_ratio"]
            metrics["zero_adv/prompts_all_incorrect_ratio_original"] = prompt_stats_original["zero_adv/prompts_all_incorrect_ratio"]
            
            # Convert UIDs to indices
            uid_to_first_idx = {}
            for i, uid in enumerate(batch_with_rollouts.non_tensor_batch["uid"]):
                if uid not in uid_to_first_idx:
                    uid_to_first_idx[uid] = i
            
            # Target all-incorrect prompts for hint generation
            target_uids = list(all_incorrect_uids)
            target_type_map = {uid: "all_incorrect" for uid in all_incorrect_uids}

            hard_prompt_indices = [uid_to_first_idx[uid] for uid in target_uids if uid in uid_to_first_idx]
            num_hard_prompts = len(hard_prompt_indices)
            
            # Build target_types list parallel to hard_prompt_indices
            target_types = [target_type_map[uid] for uid in target_uids if uid in uid_to_first_idx]
            
            # All-incorrect prompts have prob_term=0.
            original_prob_terms = {
                (prompt_idx // num_repeat) * num_repeat: 0.0
                for prompt_idx in hard_prompt_indices
            }

            print(f"[HiLL] Step {self.global_steps}: Found {num_hard_prompts} all-incorrect hard prompts "
                  f"(total zero-var: {len(zero_variance_uids)})")

        # ===== STEP 4-8: Generate hints and re-rollout if we have hard prompts =====
        num_hints_this_step = self.num_hints
        
        if num_hard_prompts >= self.min_hard_prompts:
            print(f"[HiLL] Generating {num_hints_this_step} hints for {num_hard_prompts} hard prompts...")
            
            # Prepare hinter input
            with marked_timer("prepare_hinter_input", timing_raw, color="magenta"):
                hinter_batch, gold_answers, original_questions, ground_truth_solutions = self._prepare_hinter_batch(
                    batch_with_rollouts,
                    hard_prompt_indices,
                    num_repeat,
                )
                
                hinter_world_size = self.hinter_wg.world_size
                hinter_batch_padded, hinter_pad_size = pad_dataproto_to_divisor(
                    hinter_batch, hinter_world_size
                )
            
            # Generate M hints for each hard prompt
            with marked_timer("gen_all_hints", timing_raw, color="purple"):
                hinter_batch_repeated = hinter_batch_padded.repeat(
                    repeat_times=num_hints_this_step,
                    interleave=True,
                )
                
                hinter_gen_batch = hinter_batch_repeated.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=[],
                )
                
                all_hints_output = self.hinter_wg.generate_sequences(hinter_gen_batch)
                timing_raw.update(all_hints_output.meta_info.get("timing", {}))
                # Avoid meta_info merge conflicts across generate/compute calls.
                all_hints_output.meta_info.pop("timing", None)
                hinter_batch_repeated = hinter_batch_repeated.union(all_hints_output)
                
                # Extract hints per hint_id
                num_padded_prompts = len(hinter_batch_padded)
                all_hints_full_unpadded = []
                for hint_id in range(num_hints_this_step):
                    indices = [i * num_hints_this_step + hint_id for i in range(num_padded_prompts)]
                    hint_full = hinter_batch_repeated[indices]
                    hint_full_unpadded = unpad_dataproto(hint_full, hinter_pad_size)
                    all_hints_full_unpadded.append(hint_full_unpadded)
            
            # Extract, validate, and combine hints (parsing + gates)
            with marked_timer("extract_hints", timing_raw, color="cyan"):
                max_prompt_length = self.config.data.get("max_prompt_length", 1024)
                
                all_hint_texts = []
                all_success_flags = []
                all_hinter_raw_outputs = []
                
                num_parsing_errors = 0
                num_overlong = 0
                num_answer_leaking = 0
                
                hint_token_lengths = []
                
                for hint_id, hint_batch_iter in enumerate(all_hints_full_unpadded):
                    hint_texts = []
                    success_flags = []
                    hinter_raw_outputs = []
                    
                    for prompt_idx in range(len(hint_batch_iter)):
                        resp_tokens = hint_batch_iter.batch["responses"][prompt_idx]
                        full_response = self.hinter_tokenizer.decode(resp_tokens, skip_special_tokens=True)
                        hinter_raw_outputs.append(full_response)
                        
                        original_question = original_questions[prompt_idx]
                        
                        # ===== HINT parsing =====
                        match = re.search(
                            r'<hint>\s*(.*?)\s*</hint>',
                            full_response,
                            re.IGNORECASE | re.DOTALL,
                        )
                        
                        if match:
                            hint_text = match.group(1).strip()
                            
                            if not hint_text:
                                hint_texts.append(None)
                                success_flags.append(False)
                                num_parsing_errors += 1
                            elif check_answer_leaking_string_match(hint_text, gold_answers[prompt_idx]):
                                hint_texts.append(hint_text)
                                success_flags.append(False)
                                num_answer_leaking += 1
                            else:
                                prompt_length = check_hinted_prompt_length(
                                    original_question=original_question,
                                    tokenizer=self.tokenizer,
                                    config=self.config.data,
                                    hint_text=hint_text,
                                )
                                if prompt_length > max_prompt_length:
                                    hint_texts.append(hint_text)
                                    success_flags.append(False)
                                    num_overlong += 1
                                else:
                                    hint_tokens = self.tokenizer.encode(hint_text, add_special_tokens=False)
                                    hint_token_lengths.append(len(hint_tokens))
                                    hint_texts.append(hint_text)
                                    success_flags.append(True)
                        else:
                            hint_texts.append(None)
                            success_flags.append(False)
                            num_parsing_errors += 1
                    
                    all_hint_texts.append(hint_texts)
                    all_success_flags.append(success_flags)
                    all_hinter_raw_outputs.append(hinter_raw_outputs)
                
                # No separate semantic_passed_map needed — answer leaking is
                # folded into success_flags above, so pass empty dict to
                # select_best_hint (defaults to True for all).
                semantic_passed_map = {}
                
                # Log gate statistics
                metrics["hinter/num_parsing_errors"] = num_parsing_errors
                metrics["hinter/num_overlong"] = num_overlong
                metrics["hinter/num_answer_leaking"] = num_answer_leaking
                
                total_attempts = num_hard_prompts * num_hints_this_step
                metrics["hinter/success_ratio"] = sum(
                    sum(flags) for flags in all_success_flags
                ) / total_attempts if total_attempts > 0 else 0.0
                
                # Log hint token length statistics
                if hint_token_lengths:
                    hint_min = min(hint_token_lengths)
                    hint_avg = np.mean(hint_token_lengths)
                    hint_max = max(hint_token_lengths)
                    metrics["hinter/hint_tokens_min"] = hint_min
                    metrics["hinter/hint_tokens_avg"] = hint_avg
                    metrics["hinter/hint_tokens_max"] = hint_max
                
                print(f"[HiLL] Parsing: {sum(sum(f) for f in all_success_flags)} passed, "
                      f"{num_parsing_errors} parse errors, {num_overlong} overlong, "
                      f"{num_answer_leaking} answer leaking")
            
            # Format hinted prompts and re-rollout
            with marked_timer("format_hinted_prompts", timing_raw, color="blue"):
                success_map = {}
                success_idx_to_hint_prompt = []
                
                all_hinted_prompt_batches = []
                for hint_id, (hint_texts_batch, parsing_flags) in enumerate(
                    zip(all_hint_texts, all_success_flags)
                ):
                    successful_orig_questions = []
                    successful_hint_texts = []
                    successful_original_indices = []
                    for prompt_idx, success in enumerate(parsing_flags):
                        success_map[(hint_id, prompt_idx)] = success
                        if success:
                            successful_orig_questions.append(original_questions[prompt_idx])
                            successful_hint_texts.append(hint_texts_batch[prompt_idx])
                            success_idx_to_hint_prompt.append((hint_id, prompt_idx))
                            original_idx = hard_prompt_indices[prompt_idx]
                            successful_original_indices.append(original_idx)
                    
                    if successful_orig_questions:
                        hinted_prompt_batch = batch_format_hinted_prompts(
                            original_questions=successful_orig_questions,
                            tokenizer=self.tokenizer,
                            config=self.config.data,
                            hint_texts=successful_hint_texts,
                            original_batch=batch_with_rollouts,
                            original_indices=successful_original_indices,
                        )
                        all_hinted_prompt_batches.append(hinted_prompt_batch)
                
                if all_hinted_prompt_batches:
                    combined_hinted_prompts = DataProto.concat(all_hinted_prompt_batches)
                    combined_hinted_prompts = combined_hinted_prompts.repeat(
                        repeat_times=num_repeat,
                        interleave=True
                    )
                else:
                    combined_hinted_prompts = None
            
            # Re-rollout on hinted prompts
            with marked_timer("rerollout_hints", timing_raw, color="orange"):
                if combined_hinted_prompts is not None:
                    combined_hinted_prompts_padded, hinted_pad_size = pad_dataproto_to_divisor(
                        combined_hinted_prompts, 8
                    )
                    
                    combined_hinted_gen_batch = self._get_gen_batch(combined_hinted_prompts_padded)
                    
                    if not self.async_rollout_mode:
                        all_hinted_rollout_output_padded = self.actor_rollout_wg.generate_sequences(
                            combined_hinted_gen_batch
                        )
                    else:
                        all_hinted_rollout_output_padded = self.async_rollout_manager.generate_sequences(
                            combined_hinted_gen_batch
                        )
                    # This timing is call-specific and causes DataProto.concat
                    # meta_info conflicts when mixed with fallback rollouts.
                    all_hinted_rollout_output_padded.meta_info.pop("timing", None)
                    
                    all_hinted_full_batch_padded = combined_hinted_prompts_padded.union(all_hinted_rollout_output_padded)
                    all_hinted_full_batch = unpad_dataproto(all_hinted_full_batch_padded, hinted_pad_size)
                    
                    if "response_mask" not in all_hinted_full_batch.batch:
                        all_hinted_full_batch.batch["response_mask"] = compute_response_mask(all_hinted_full_batch)
                    
                    if self.reward_fn is not None:
                        reward_tensor, _ = compute_reward(all_hinted_full_batch, self.reward_fn)
                        all_hinted_full_batch.batch["token_level_scores"] = reward_tensor
                        all_hinted_full_batch.batch["token_level_rewards"] = reward_tensor
                    
                    all_hinted_full_batch = compute_advantage(
                        all_hinted_full_batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=num_repeat,
                        config=self.config.algorithm,
                    )
                else:
                    all_hinted_full_batch = None
            
            # Reconstruct hint batches for reward computation
            with marked_timer("split_hints", timing_raw, color="cyan"):
                hint_prompt_to_success_idx = {}
                for flat_idx, (hint_id, prompt_idx) in enumerate(success_idx_to_hint_prompt):
                    hint_prompt_to_success_idx[(hint_id, prompt_idx)] = flat_idx
                
                hinted_prompts_batches = []
                
                for hint_id in range(num_hints_this_step):
                    hint_batch_parts = []
                    
                    for prompt_idx in range(len(hinter_batch)):
                        if success_map.get((hint_id, prompt_idx), False) and all_hinted_full_batch is not None:
                            flat_success_idx = hint_prompt_to_success_idx[(hint_id, prompt_idx)]
                            start_idx = flat_success_idx * num_repeat
                            end_idx = start_idx + num_repeat
                            prompt_rollouts = all_hinted_full_batch[start_idx:end_idx]
                            hint_batch_parts.append(prompt_rollouts)
                        else:
                            original_idx = hard_prompt_indices[prompt_idx]
                            group_base = (original_idx // num_repeat) * num_repeat
                            fallback_rollouts = batch_with_rollouts[group_base:group_base + num_repeat]
                            hint_batch_parts.append(fallback_rollouts)
                    
                    if hint_batch_parts:
                        # Defensive cleanup: DataProto.concat enforces
                        # equal meta_info values across parts.
                        for part in hint_batch_parts:
                            if part.meta_info is not None:
                                part.meta_info.pop("timing", None)
                        hint_batch = DataProto.concat(hint_batch_parts)
                        hinted_prompts_batches.append(hint_batch)
            
            # ===== COMPUTE HINT RELIANCE FOR ALL SUCCESSFUL HINTS =====
            # Hint reliance = mean_logprob(response | Q) - mean_logprob(response | Q+hint)
            # Computed for ALL successful hints via two compute_log_prob forward passes.
            # Used functionally by transfer_weighted_* reward types; logged for all modes.
            hint_reliance_map = None  # (hint_id, prompt_idx) -> per-rollout hint reliance [N]
            with marked_timer("compute_all_hint_reliance", timing_raw, color="blue"):
                if all_hinted_full_batch is not None and success_idx_to_hint_prompt:
                    qh_parts = []
                    q_parts = []
                    
                    for flat_idx, (hint_id, prompt_idx) in enumerate(success_idx_to_hint_prompt):
                        original_idx = hard_prompt_indices[prompt_idx]
                        group_base = (original_idx // num_repeat) * num_repeat
                        original_prompts_slice = batch_with_rollouts[group_base:group_base + num_repeat]
                        
                        start_idx = flat_idx * num_repeat
                        end_idx = start_idx + num_repeat
                        hint_rollouts = all_hinted_full_batch[start_idx:end_idx]
                        
                        qh_parts.append(hint_rollouts)
                        
                        # Splice Q prompt + Q+hint response
                        resp_len = hint_rollouts.batch["responses"].size(-1)
                        orig_resp_len = original_prompts_slice.batch["responses"].size(-1)
                        assert resp_len == orig_resp_len, (
                            f"Response length mismatch: hinted {resp_len} vs original {orig_resp_len}. "
                            "input_ids splice assumes identical [prompt|response] layout."
                        )
                        
                        new_input_ids = original_prompts_slice.batch["input_ids"].clone()
                        new_input_ids[:, -resp_len:] = hint_rollouts.batch["responses"]
                        new_attention_mask = original_prompts_slice.batch["attention_mask"].clone()
                        new_attention_mask[:, -resp_len:] = hint_rollouts.batch["attention_mask"][:, -resp_len:]
                        new_position_ids = compute_position_id_with_mask(new_attention_mask)
                        
                        q_batch_entry = DataProto.from_single_dict({
                            "input_ids": new_input_ids,
                            "attention_mask": new_attention_mask,
                            "position_ids": new_position_ids,
                            "responses": hint_rollouts.batch["responses"].clone(),
                            "response_mask": hint_rollouts.batch["response_mask"].clone(),
                        })
                        q_parts.append(q_batch_entry)
                    
                    if qh_parts:
                        all_qh = DataProto.concat(qh_parts)
                        all_q = DataProto.concat(q_parts)
                        
                        all_qh_padded, qh_pad = pad_dataproto_to_divisor(all_qh, 8)
                        all_q_padded, q_pad = pad_dataproto_to_divisor(all_q, 8)
                        
                        qh_log_prob_output = self.actor_rollout_wg.compute_log_prob(all_qh_padded)
                        q_log_prob_output = self.actor_rollout_wg.compute_log_prob(all_q_padded)
                        
                        qh_log_probs = unpad_dataproto(qh_log_prob_output, qh_pad).batch["old_log_probs"]
                        q_log_probs = unpad_dataproto(q_log_prob_output, q_pad).batch["old_log_probs"]
                        resp_masks = all_qh.batch["response_mask"].float()
                        
                        token_counts = resp_masks.sum(dim=-1).clamp(min=1.0)
                        qh_mean_logprob = (qh_log_probs * resp_masks).sum(dim=-1) / token_counts
                        q_mean_logprob = (q_log_probs * resp_masks).sum(dim=-1) / token_counts
                        all_hint_reliance = q_mean_logprob - qh_mean_logprob
                        
                        # Build map: (hint_id, prompt_idx) -> per-rollout hint reliance [N]
                        hint_reliance_map = {}
                        for flat_idx, (hint_id, prompt_idx) in enumerate(success_idx_to_hint_prompt):
                            s = flat_idx * num_repeat
                            e = s + num_repeat
                            hint_reliance_map[(hint_id, prompt_idx)] = all_hint_reliance[s:e]
            
            # Select best hint
            with marked_timer("select_best_hint", timing_raw, color="green"):
                if all_hinted_full_batch is not None and hinted_prompts_batches:
                    best_hints, all_hint_rewards, all_hint_reliance = select_best_hint(
                        hinted_prompts_batches=hinted_prompts_batches,
                        num_repeat=num_repeat,
                        hard_prompt_indices=hard_prompt_indices,
                        original_prob_terms=original_prob_terms,
                        success_map=success_map,
                        timeout_map={},
                        semantic_passed_map=semantic_passed_map,
                        positive_threshold=self.POSITIVE_THRESHOLD,
                        reward_type=self.hinter_reward_type,
                        hint_reliance_map=hint_reliance_map,
                        transfer_temperature=self.transfer_temperature,
                    )
                    
                    # Filter: hint's prob_term must exceed the original
                    # prompt's prob_term.  For all-incorrect prompts the
                    # original is 0 so this reduces to prob_term > 0.
                    filtered_best_hints = {}
                    for group_base, hint_info in best_hints.items():
                        orig_prob = original_prob_terms.get(group_base, 0.0)
                        if hint_info["prob_term"] > orig_prob:
                            filtered_best_hints[group_base] = hint_info
                    
                    best_hints = filtered_best_hints
                    
                    if best_hints:
                        all_rewards = [r["reward"] for r in best_hints.values()]
                        metrics["hinter/avg_variance_reward"] = np.mean(all_rewards)
                        metrics["hinter/max_variance_reward"] = np.max(all_rewards)
                        metrics["hinter/num_successful_replacements"] = len(best_hints)
                else:
                    best_hints = {}
                    all_hint_rewards = {}
                    all_hint_reliance = {}
                
                hard_prompt_group_indices = []
                for group_base in best_hints.keys():
                    hard_prompt_group_indices.extend(range(group_base, group_base + num_repeat))
            
            # ===== LOG HINT RELIANCE =====
            # Log hint reliance only for the best hints used for replacement.
            with marked_timer("log_hint_reliance", timing_raw, color="blue"):
                hint_reliance_values = []
                
                if best_hints:
                    for hint_info in best_hints.values():
                        hint_reliance = hint_info.get("avg_hint_reliance")
                        if hint_reliance is not None:
                            hint_reliance_values.append(hint_reliance)
                
                if hint_reliance_values:
                    metrics["hinter/hint_reliance_min"] = min(hint_reliance_values)
                    metrics["hinter/hint_reliance_avg"] = np.mean(hint_reliance_values)
                    metrics["hinter/hint_reliance_max"] = max(hint_reliance_values)
            
            # Save hinter inspection logs to local (after rewards are computed)
            if (self.global_steps - 1) % self.inspection_log_freq == 0:
                log_dir = os.path.join(
                    self.config.trainer.default_local_dir, 
                    "hinter_logs"
                )
                os.makedirs(log_dir, exist_ok=True)
                
                log_file = os.path.join(log_dir, f"step_{self.global_steps}.jsonl")
                with open(log_file, "w") as f:
                    for hint_id in range(num_hints_this_step):
                        for prompt_idx in range(len(hinter_batch)):
                            original_idx = hard_prompt_indices[prompt_idx]
                            group_base = (original_idx // num_repeat) * num_repeat
                            hint_rewards_list = all_hint_rewards.get(group_base, [])
                            variance_reward = hint_rewards_list[hint_id] if hint_id < len(hint_rewards_list) else None
                            
                            # Extract re-rollout responses and rewards
                            rerollouts = []
                            rerollout_pass_rate = None
                            
                            if success_map.get((hint_id, prompt_idx), False) and hinted_prompts_batches:
                                hint_batch_iter = hinted_prompts_batches[hint_id]
                                start_idx = prompt_idx * num_repeat
                                end_idx = start_idx + num_repeat
                                
                                for local_id, rollout_idx in enumerate(range(start_idx, end_idx)):
                                    if rollout_idx < len(hint_batch_iter):
                                        resp_tokens = hint_batch_iter.batch["responses"][rollout_idx]
                                        resp_mask = hint_batch_iter.batch["response_mask"][rollout_idx]
                                        valid_tokens = resp_tokens[resp_mask.bool()]
                                        resp_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                                        
                                        token_rewards = hint_batch_iter.batch["token_level_rewards"][rollout_idx]
                                        seq_reward = (token_rewards * resp_mask).sum().item()
                                        
                                        rerollouts.append({
                                            "rollout_id": local_id,
                                            "content": resp_text,
                                            "reward": seq_reward,
                                        })
                                
                                if rerollouts:
                                    num_correct = sum(1 for r in rerollouts if r["reward"] > self.POSITIVE_THRESHOLD)
                                    rerollout_pass_rate = num_correct / len(rerollouts)
                            
                            log_entry = {
                                "step": self.global_steps,
                                "timestamp": datetime.now().isoformat(),
                                "hint_id": hint_id,
                                "prompt_idx": prompt_idx,
                                "original_question": original_questions[prompt_idx],
                                "gold_answer": gold_answers[prompt_idx],
                                "target_type": target_types[prompt_idx],
                                "hint_text": all_hint_texts[hint_id][prompt_idx],
                                "success": all_success_flags[hint_id][prompt_idx],
                                "raw_output": all_hinter_raw_outputs[hint_id][prompt_idx],
                                "variance_reward": variance_reward,
                                "rerollouts": rerollouts,
                                "rerollout_pass_rate": rerollout_pass_rate,
                            }
                            f.write(json.dumps(log_entry) + "\n")
                print(f"[HiLL] Saved hinter inspection log to {log_file}")
            
            # ===== Build final batch: replace improved prompts, keep originals otherwise =====
            with marked_timer("build_final_batch", timing_raw, color="blue"):
                final_batch, replacement_stats = build_batch_with_best_hints(
                    batch_with_rollouts=batch_with_rollouts,
                    best_hints=best_hints,
                    num_repeat=num_repeat,
                )
                
                print(f"[HiLL] Replaced {replacement_stats['num_replaced']}/{replacement_stats['num_original_prompts']} "
                      f"prompts with best hints (kept {replacement_stats['num_kept_original']} original)")
                
                metrics["hinter/num_replaced"] = replacement_stats["num_replaced"]
                metrics["hinter/num_kept_original"] = replacement_stats["num_kept_original"]
                metrics["hinter/final_batch_prompts"] = replacement_stats["final_batch_prompts"]
            
            # Recompute advantages
            with marked_timer("recompute_advantages", timing_raw, color="green"):
                final_batch = compute_advantage(
                    final_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=num_repeat,
                    config=self.config.algorithm,
                )
            
            # Prepare hinter training state separately so reasoner update_actor
            # receives only lean training tensors/metadata.
            with marked_timer("prepare_hinter_update_state", timing_raw, color="magenta"):
                hinter_update_state = {
                    "hinter_full_batches": all_hints_full_unpadded,
                    "all_hint_rewards": all_hint_rewards,
                    "all_hint_reliance": all_hint_reliance,
                    "hard_prompt_indices": hard_prompt_indices,
                }
            
            # Compute hinted statistics
            with marked_timer("compute_hinted_stats", timing_raw, color="green"):
                prompt_stats_hinted, _, _, _ = compute_prompt_statistics(
                    data=final_batch,
                    num_repeat=num_repeat,
                    positive_threshold=self.POSITIVE_THRESHOLD,
                )
                
                metrics["zero_adv/prompts_zero_advantage_ratio"] = prompt_stats_hinted["zero_adv/prompts_zero_advantage_ratio"]
                metrics["zero_adv/prompts_all_correct_ratio"] = prompt_stats_hinted["zero_adv/prompts_all_correct_ratio"]
                metrics["zero_adv/prompts_all_incorrect_ratio"] = prompt_stats_hinted["zero_adv/prompts_all_incorrect_ratio"]
        else:
            print(f"[HiLL] Only {num_hard_prompts} hard prompts (min: {self.min_hard_prompts}), skipping hint generation")
            final_batch = batch_with_rollouts
            
            metrics["zero_adv/prompts_zero_advantage_ratio"] = prompt_stats_original["zero_adv/prompts_zero_advantage_ratio"]
            metrics["zero_adv/prompts_all_correct_ratio"] = prompt_stats_original["zero_adv/prompts_all_correct_ratio"]
            metrics["zero_adv/prompts_all_incorrect_ratio"] = prompt_stats_original["zero_adv/prompts_all_incorrect_ratio"]
            
        # Filter zero-advantage prompt groups before reasoner update.
        # These groups contribute no policy gradient under GRPO.
        # with marked_timer("filter_zero_advantage_groups", timing_raw, color="cyan"):
        #     final_batch, filter_stats = filter_zero_advantage_groups(
        #         batch=final_batch,
        #         num_repeat=num_repeat,
        #     )
        #     metrics.update(filter_stats)

        return final_batch, hinter_update_state
    
    def fit(self):
        """Main training loop with reasoner-first workflow."""
        if not hasattr(self, "actor_rollout_wg") or self.actor_rollout_wg is None:
            self.init_workers()

        print("[HiLL] Starting training with reasoner-first workflow")
        
        # Load checkpoint
        self.global_steps = self._load_checkpoint()
        
        # Setup logger
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # Training loop
        from tqdm import tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training",
        )
        
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch))], dtype=object
                )
                
                batch.meta_info = batch.meta_info or {}
                batch.meta_info["global_steps"] = self.global_steps
                
                is_last_step = self.global_steps >= self.total_training_steps
                
                # ===== Main training step =====
                with marked_timer("step", timing_raw):
                    final_batch, hinter_update_state = self._training_step_reasoner_first(
                        batch=batch,
                        timing_raw=timing_raw,
                        metrics=metrics,
                    )
                
                # ===== Compute log probs for training =====
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(final_batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = final_batch.batch["response_mask"]
                    
                    from verl.trainer.ppo.core_algos import agg_loss
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=loss_agg_mode,
                    )
                    metrics["actor/entropy"] = entropy_agg.detach().item()
                    old_log_prob.batch.pop("entropys")
                    final_batch = final_batch.union(old_log_prob)
                
                # Reference log probs if needed
                if self.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(final_batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(final_batch)
                        final_batch = final_batch.union(ref_log_prob)
                
                final_batch.meta_info["global_token_num"] = torch.sum(
                    final_batch.batch["attention_mask"], dim=-1
                ).tolist()
                
                # Critic values if needed
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(final_batch)
                        final_batch = final_batch.union(values)
                    
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(final_batch)
                    
                    from verl.utils.metric import reduce_metrics
                    critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_metrics)
                
                # ===== Update reasoner =====
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw, color="red"):
                        final_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(final_batch)
                    
                    from verl.utils.metric import reduce_metrics
                    actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_metrics)
                
                # ===== Update hinter =====
                if hinter_update_state is not None:
                    with marked_timer("update_hinter", timing_raw, color="purple"):
                        hinter_full_batches = hinter_update_state["hinter_full_batches"]
                        all_hint_rewards = hinter_update_state.get("all_hint_rewards", {})
                        hard_prompt_indices = hinter_update_state.get("hard_prompt_indices", [])
                        
                        if hinter_full_batches and len(hard_prompt_indices) > 0:
                            hinter_training_samples = []
                            hinter_uids = []
                            hinter_hint_rewards = []
                            
                            num_repeat = self.config.actor_rollout_ref.rollout.n
                            
                            for i, prompt_idx in enumerate(hard_prompt_indices):
                                group_base = (prompt_idx // num_repeat) * num_repeat
                                uid = f"hinter_step{self.global_steps}_prompt{i}"
                                
                                hint_rewards = all_hint_rewards.get(group_base, [0.0] * self.num_hints)
                                
                                for hint_id in range(self.num_hints):
                                    if hint_id < len(hinter_full_batches) and i < len(hinter_full_batches[hint_id]):
                                        hint_sample = hinter_full_batches[hint_id][i:i+1]
                                        hinter_training_samples.append(hint_sample)
                                        hinter_uids.append(uid)
                                        
                                        hint_reward = hint_rewards[hint_id] if hint_id < len(hint_rewards) else 0.0
                                        hinter_hint_rewards.append(hint_reward)
                            
                            if hinter_training_samples:
                                for sample in hinter_training_samples:
                                    if sample.meta_info is not None:
                                        sample.meta_info.pop("timing", None)
                                hinter_batch_to_train = DataProto.concat(hinter_training_samples)
                                hinter_batch_to_train.non_tensor_batch["uid"] = np.array(hinter_uids, dtype=object)
                                
                                if "response_mask" not in hinter_batch_to_train.batch:
                                    hinter_batch_to_train.batch["response_mask"] = compute_response_mask(hinter_batch_to_train)
                                
                                response_length = hinter_batch_to_train.batch["responses"].shape[1]
                                response_mask = hinter_batch_to_train.batch["response_mask"]
                                token_level_rewards = torch.zeros(len(hinter_hint_rewards), response_length)
                                
                                for i, reward in enumerate(hinter_hint_rewards):
                                    valid_positions = response_mask[i].nonzero(as_tuple=True)[0]
                                    if len(valid_positions) > 0:
                                        last_pos = valid_positions[-1].item()
                                        token_level_rewards[i, last_pos] = reward
                                
                                hinter_batch_to_train.batch["token_level_rewards"] = token_level_rewards
                                hinter_batch_to_train.batch["token_level_scores"] = token_level_rewards.clone()
                                
                                hinter_batch_padded, h_pad = pad_dataproto_to_divisor(
                                    hinter_batch_to_train, self.hinter_wg.world_size
                                )
                                
                                old_log_prob_h = self.hinter_wg.compute_log_prob(hinter_batch_padded)
                                
                                entropys_h = old_log_prob_h.batch["entropys"]
                                response_masks_h = hinter_batch_padded.batch["response_mask"]
                                from verl.trainer.ppo.core_algos import agg_loss
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg_h = agg_loss(
                                    loss_mat=entropys_h,
                                    loss_mask=response_masks_h,
                                    loss_agg_mode=loss_agg_mode,
                                )
                                metrics["hinter_training/entropy"] = entropy_agg_h.detach().item()
                                old_log_prob_h.batch.pop("entropys")
                                hinter_batch_padded = hinter_batch_padded.union(old_log_prob_h)
                                
                                hinter_batch_padded = compute_advantage(
                                    hinter_batch_padded,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.num_hints,
                                    config=self.config.algorithm,
                                )
                                
                                hinter_batch_padded.meta_info = hinter_batch_padded.meta_info or {}
                                hinter_batch_padded.meta_info["global_token_num"] = torch.sum(
                                    hinter_batch_padded.batch["attention_mask"], dim=-1
                                ).tolist()
                                hinter_batch_padded.meta_info["multi_turn"] = False
                                
                                hinter_output = self.hinter_wg.update_actor(hinter_batch_padded)
                                
                                from verl.utils.metric import reduce_metrics
                                hinter_metrics = reduce_metrics(hinter_output.meta_info["metrics"])
                                hinter_metrics = {f"hinter_training/{k}": v for k, v in hinter_metrics.items()}
                                metrics.update(hinter_metrics)
                                
                                metrics["hinter_training/avg_hint_reward"] = np.mean(hinter_hint_rewards)
                                metrics["hinter_training/num_training_samples"] = len(hinter_hint_rewards)
                                
                                # Log hint_reliance for hinter training
                                all_hint_reliance = hinter_update_state.get("all_hint_reliance", {})
                                if all_hint_reliance:
                                    hint_reliance_train = []
                                    for pi, prompt_idx in enumerate(hard_prompt_indices):
                                        group_base = (prompt_idx // num_repeat) * num_repeat
                                        hint_reliance_list = all_hint_reliance.get(group_base, [])
                                        for hint_reliance_val in hint_reliance_list:
                                            if hint_reliance_val is not None:
                                                hint_reliance_train.append(hint_reliance_val)
                                    if hint_reliance_train:
                                        metrics["hinter_training/hint_reliance_avg"] = np.mean(hint_reliance_train)
                                
                                # Reuse compute_data_metrics for hinter batch
                                from verl.trainer.ppo.metric_utils import compute_data_metrics
                                hinter_data_metrics = compute_data_metrics(
                                    batch=hinter_batch_padded, 
                                    use_critic=False,
                                )
                                hinter_data_metrics = {
                                    f"hinter_training/{k}": v 
                                    for k, v in hinter_data_metrics.items()
                                }
                                metrics.update(hinter_data_metrics)
                
                # ===== Validation =====
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                
                # ===== Checkpointing =====
                from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
                esi_close = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                
                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                
                # ===== Logging =====
                from verl.trainer.ppo.metric_utils import (
                    compute_data_metrics,
                    compute_timing_metrics,
                    compute_throughout_metrics,
                )
                
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=final_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=final_batch, timing_raw=timing_raw))
                
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=final_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                logger.log(data=metrics, step=self.global_steps)
                
                progress_bar.update(1)
                self.global_steps += 1
                
                if is_last_step:
                    from pprint import pprint
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
            
            if self.global_steps >= self.total_training_steps:
                break
        
        progress_bar.close()
        print("[HiLL] Training completed")
