# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
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
Worker for hint generation.

This worker manages a separate LLM that generates pedagogical hints
for a reasoning LLM, helping to improve training efficiency by targeting
challenging prompts.
"""

import torch
from typing import Optional

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class HinterWorker(ActorRolloutRefWorker):
    """
    Worker for hint generation that inherits from ActorRolloutRefWorker.

    This worker specializes in generating hints based on:
    - Original prompt
    - Reasoning LLM's rollout attempts
    - Ground truth answer

    The goal is to generate hinted prompts that elicit better responses
    from the reasoning LLM.
    """

    def __init__(self, config, role="hinter", **kwargs):
        """
        Initialize hinter worker.

        Args:
            config: Configuration for the hinter model
            role: Role name for this worker (used for identification)
            **kwargs: Additional arguments passed to parent class
        """
        # Parent class (ActorRolloutRefWorker) only accepts specific roles.
        # We always use "actor_rollout" mode to ensure proper weight syncing.
        #
        # Key behavior:
        # - "actor_rollout" mode: Enables both actor (training) and rollout (inference)
        # - Weight syncing: Happens automatically in generate_sequences() when _is_actor=True
        # - Training: Only occurs when update_actor() is explicitly called

        # Always use actor_rollout mode for proper weight initialization
        parent_role = "actor_rollout"

        super().__init__(config, role=parent_role, **kwargs)

        # Store training flag
        train_hinter = config.get("train_hinter", True)

        # Store metadata
        self.hinter_role = role
        self.train_hinter = train_hinter
