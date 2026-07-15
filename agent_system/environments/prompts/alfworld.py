# Copyright 2025 Nanyang Technological University (NTU), Singapore
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

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# PS-GRPO (预测充分性) 变体: 在 <think> 与 <action> 之间要求受限预测 <predict> 块。
# 预测由 verifiable_features.parse_predict_block 规则解析、由环境的下一步观测裁决，
# 不经过 LLM judge。见 docs/ps_grpo_integration_design.md §2A。
_ALFWORLD_PREDICT_INSTRUCTION = """After your reasoning, predict the outcome of your action, enclosed within <predict> </predict> tags in exactly this format:
<predict>next_location: [the location you will be at after this action, e.g. 'cabinet 2', or 'none' if unchanged]; target_visible: [yes/no - will any object mentioned in your task be visible in the next observation?]; task_done: [yes/no - will the task be completed after this action?]</predict>
Finally, you should choose an admissible action for current step and present it within <action> </action> tags."""

ALFWORLD_TEMPLATE_NO_HIS_PS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
""" + _ALFWORLD_PREDICT_INSTRUCTION + "\n"

ALFWORLD_TEMPLATE_PS = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
""" + _ALFWORLD_PREDICT_INSTRUCTION + "\n"

# v0.2 schema 协议变体 (feature_protocol=schema): 预测问题全部任务无关
# (docs/ps_grpo_integration_design.md §7.2)。objects_visible 取代任务语义的
# target_visible; visible_objects 为开放集探针 (F1 记日志,S4b 不计奖励)。
_ALFWORLD_PREDICT_INSTRUCTION_SCHEMA = (
    "After your reasoning, predict the outcome of your action, enclosed within <predict> </predict> tags in exactly this format:\n"
    "<predict>next_location: [the location you will be at after this action, e.g. 'cabinet 2', or 'none' if unchanged]; "
    "objects_visible: [yes/no - will the next observation list at least one object (\"you see ...\")?]; "
    "visible_objects: [comma-separated objects you expect to see next, or 'none']; "
    "task_done: [yes/no - will the task be completed after this action?]</predict>\n"
    "Finally, you should choose an admissible action for current step and present it within <action> </action> tags."
)

ALFWORLD_TEMPLATE_NO_HIS_PS_SCHEMA = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
""" + _ALFWORLD_PREDICT_INSTRUCTION_SCHEMA + "\n"

ALFWORLD_TEMPLATE_PS_SCHEMA = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
""" + _ALFWORLD_PREDICT_INSTRUCTION_SCHEMA + "\n"