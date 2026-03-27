#!/bin/bash

set -xeuo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export WORKING_DIR="${PWD}"

# Reasoner LLM
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
model_name="Qwen2.5-7B-Instruct"

# Hinter LLM
hinter_model_path=Qwen/Qwen3-4B-Instruct-2507

# Wandb setting
project_name="HiLL"
experiment_name="HiLL-${model_name}"

# Output
ckpts_dir="./checkpoints/${project_name}/${experiment_name}"
mkdir -p "${ckpts_dir}/logs"

# Training setting
num_gpus=8

# Batch size calculation for HiLL:
train_prompt_bsz=128  # Full batch size
train_prompt_mini_bsz=64  # For PPO mini-batch updates

# Algorithm setting
algorithm=grpo
n=8  # Number of rollouts per prompt for reasoning LLM
kl_coef=0.0
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Hint Generator Settings
num_hints=4  # M: Number of candidate hints per hard prompt
min_zero_var_prompts=1  # Minimum hard prompts to trigger hint generation (1 = always run unless none)
hinter_reward_type="transfer_weighted_non_deg"  # Reward type: "variance", "non_deg", "transfer_weighted_variance", or "transfer_weighted_non_deg"
transfer_temperature=0.3  # Temperature T for transfer weight: w = exp(min(Δ_c, 0) / T)
hinter_ppo_epochs=1   # PPO epochs per hinter update
hinter_prompt_length=10240  # Max prompt length for hinter
hinter_response_length=1024  # Max response length for hinter

# Training data
train_path="./data/train/train.parquet"
test_path="./data/validation/test.parquet"
train_files="['$train_path']"
test_files="['$test_path']"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${algorithm} \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=2048 \
    data.max_response_length=10240 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path=${model_name_or_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((2048 + 8192)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${num_gpus} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.default_local_dir=${ckpts_dir} \
    trainer.test_freq=-1 \
    trainer.total_training_steps=500 \
    trainer.total_epochs=1000 \
    +use_hinter=True \
    hinter.model.path=${hinter_model_path} \
    hinter.model.enable_gradient_checkpointing=True \
    hinter.model.use_remove_padding=True \
    hinter.actor.ppo_mini_batch_size=64 \
    hinter.actor.ppo_micro_batch_size_per_gpu=4 \
    hinter.actor.ppo_epochs=${hinter_ppo_epochs} \
    hinter.actor.use_kl_loss=False \
    hinter.actor.kl_loss_coef=0.0 \
    hinter.actor.kl_loss_type=low_var_kl \
    hinter.actor.optim.lr=1e-6 \
    hinter.actor.clip_ratio_low=${clip_ratio_low} \
    hinter.actor.clip_ratio_high=${clip_ratio_high} \
    hinter.actor.fsdp_config.param_offload=False \
    hinter.actor.fsdp_config.optimizer_offload=False \
    hinter.rollout.n=${num_hints} \
    hinter.rollout.prompt_length=${hinter_prompt_length} \
    hinter.rollout.response_length=${hinter_response_length} \
    hinter.rollout.gpu_memory_utilization=0.85 \
    hinter.rollout.log_prob_micro_batch_size_per_gpu=64 \
    hinter.rollout.max_num_batched_tokens=$((hinter_prompt_length + hinter_response_length)) \
    hinter.ref.log_prob_micro_batch_size_per_gpu=64 \
    hinter.num_hints=${num_hints} \
    hinter.min_zero_var_prompts=${min_zero_var_prompts} \
    +hinter.hinter_reward_type=${hinter_reward_type} \
    +hinter.transfer_temperature=${transfer_temperature} 2>&1 | tee ${ckpts_dir}/logs/log

