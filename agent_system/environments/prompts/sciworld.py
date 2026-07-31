# Copyright 2026 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

# --------------------- ScienceWorld --------------------- #
SCIWORLD_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in ScienceWorld, a text-based science simulation environment.
Your task is to: {task_description}
Your current observation is: {current_observation}
Admissible action templates (OBJ stands for an object name):
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which action best advances the science task. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should output exactly one concrete action for the current step (with OBJ slots filled by real object names) within <action> </action> tags.
"""

SCIWORLD_TEMPLATE = """
You are an expert autonomous agent operating in ScienceWorld, a text-based science simulation environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Admissible action templates (OBJ stands for an object name):
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which action best advances the science task. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should output exactly one concrete action for the current step (with OBJ slots filled by real object names) within <action> </action> tags.
"""

# PS-GRPO (预测充分性) 变体: <think> 与 <action> 之间要求受限预测 <predict> 块。
# Φ_sciworld 任务无关、规则可验 (verifiable_features.create_sciworld_feature_extractor);
# 字段与 ALFWorld v0.2 schema 同名 → 解析/验证/gold 三路复用。
_SCIWORLD_PREDICT_INSTRUCTION = """After your reasoning, predict the outcome of your action, enclosed within <predict> </predict> tags in exactly this format:
<predict>next_location: [room name if your action moves you to another room, otherwise 'none']; objects_visible: [yes/no - will the next observation list at least one object?]; visible_objects: [comma-separated object names you expect to see, or 'none']</predict>
Finally, you should output exactly one concrete action for the current step (with OBJ slots filled by real object names) within <action> </action> tags."""

SCIWORLD_TEMPLATE_NO_HIS_PS = SCIWORLD_TEMPLATE_NO_HIS.replace(
    "Once you've finished your reasoning, you should output exactly one concrete action for the current step (with OBJ slots filled by real object names) within <action> </action> tags.",
    _SCIWORLD_PREDICT_INSTRUCTION,
)

SCIWORLD_TEMPLATE_PS = SCIWORLD_TEMPLATE.replace(
    "Once you've finished your reasoning, you should output exactly one concrete action for the current step (with OBJ slots filled by real object names) within <action> </action> tags.",
    _SCIWORLD_PREDICT_INSTRUCTION,
)
