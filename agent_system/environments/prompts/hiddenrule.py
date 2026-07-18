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

# --------------------- HiddenRule-Gym --------------------- #
# 措辞与 ALFWorld 模板保持同风格,便于跨环境比较 agent 行为。

HIDDENRULE_TEMPLATE_NO_HIS = """
You are an expert agent operating in the HiddenRule Environment: several rooms contain levers, dials, buttons and notes. A hidden mechanism controls the vault. Your task is to: discover the hidden mechanism and open the vault.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

HIDDENRULE_TEMPLATE = """
You are an expert agent operating in the HiddenRule Environment: several rooms contain levers, dials, buttons and notes. A hidden mechanism controls the vault. Your task is to: discover the hidden mechanism and open the vault.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# PS-GRPO 变体 (schema 协议): <think> 与 <action> 之间要求受限预测 <predict> 块。
# 字段与 ALFWorld schema 版同名 (next_location/objects_visible/visible_objects/task_done),
# 整条 PS 管线跨环境复用。
_HIDDENRULE_PREDICT_INSTRUCTION = (
    "After your reasoning, predict the outcome of your action, enclosed within <predict> </predict> tags in exactly this format:\n"
    "<predict>next_location: [the room you will be in after this action, e.g. 'room 3', or 'none' if unchanged]; "
    "objects_visible: [yes/no - will the next observation list any devices, notes or items?]; "
    "visible_objects: [comma-separated device/note/item names you expect to see next, or 'none']; "
    "task_done: [yes/no - will the vault be open after this action?]</predict>\n"
    "Finally, you should choose an admissible action for current step and present it within <action> </action> tags."
)

HIDDENRULE_TEMPLATE_NO_HIS_PS = """
You are an expert agent operating in the HiddenRule Environment: several rooms contain levers, dials, buttons and notes. A hidden mechanism controls the vault. Your task is to: discover the hidden mechanism and open the vault.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
""" + _HIDDENRULE_PREDICT_INSTRUCTION + "\n"

HIDDENRULE_TEMPLATE_PS = """
You are an expert agent operating in the HiddenRule Environment: several rooms contain levers, dials, buttons and notes. A hidden mechanism controls the vault. Your task is to: discover the hidden mechanism and open the vault.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
""" + _HIDDENRULE_PREDICT_INSTRUCTION + "\n"

# C-sweep 变体 (主图 1): predict 块加入 device_states 字段 —— 设备读数是覆盖度
# 阶梯的主要字段维度,没有它 Φ 池只有 room/objects_visible,C 档位拉不开。
# 门控: env.hiddenrule.prediction.predict_device_states=True。
_HIDDENRULE_PREDICT_INSTRUCTION_DEV = _HIDDENRULE_PREDICT_INSTRUCTION.replace(
    "task_done:",
    "device_states: [comma-separated device readings you expect in the next observation, "
    "e.g. 'lever_a=up, dial_b=2', or 'none']; "
    "task_done:",
)

HIDDENRULE_TEMPLATE_NO_HIS_PS_DEV = HIDDENRULE_TEMPLATE_NO_HIS_PS.replace(
    _HIDDENRULE_PREDICT_INSTRUCTION, _HIDDENRULE_PREDICT_INSTRUCTION_DEV)
HIDDENRULE_TEMPLATE_PS_DEV = HIDDENRULE_TEMPLATE_PS.replace(
    _HIDDENRULE_PREDICT_INSTRUCTION, _HIDDENRULE_PREDICT_INSTRUCTION_DEV)
