# ALFWorld
ALFWORLD_INIT_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].
Your response should be a valid json file in the following format: \n{{\n\"thoughts\": \"first describe what do you see in the image using the text description, then carefully think about which action to complete the task.\", \n\"action\": \"an admissible action\"\n}}
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} actions you took and the corresponding environment feedback: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].
Your response should be a valid json file in the following format: \n{{\n\"thoughts\": \"first describe what do you see in the image using the text description, then carefully think about which action to complete the task.\", \n\"action\": \"an admissible action\"\n}}
"""

# Sokoban
SOKOBAN_INIT_TEMPLATE = """
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Your Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls (`#`).
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Current Step
Your current observation is:
{current_observation}
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about the current situation before deciding on a final action. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose the final action and present it within <action> </action> tags.
"""

SOKOBAN_TEMPLATE = """
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Your Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls (`#`).
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Current Step
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is:
{current_observation}
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about the current situation before deciding on a final action. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose the final action and present it within <action> </action> tags.
"""


# Sokoban Visual

SOKOBAN_VISUAL_INIT_TEMPLATE = """
You are an expert agent operating in the Sokoban environment. Your goal is to push all the boxes onto the target spots. Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls.
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Visual Elements in the Image:
Character: A small, green alien-like figure with two antennae and black eyes. It represents you.
Box: A yellow crate marked with an orange "X" across its front. It is the box you need to push.
Target: A black tile outlined in red, with a small red diamond shape in the center. It marks the destination where a box should be pushed.

# Current Step
Your current observation is shown in the image: <image>
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about the current situation before deciding on a final action. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose the final action and present it within <action> </action> tags.
"""

SOKOBAN_VISUAL_TEMPLATE = """
You are an expert agent operating in the Sokoban environment. Your goal is to push all the boxes onto the target spots. Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls.
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Text Symbols
Walls (`#`), Floor (`_`), Targets (`O`), Boxes (`X`), Player (`P`), Box on Target (`√`), Player on Target (`S`)

# Visual Elements in the Image:
Character: A small, green alien-like figure with two antennae and black eyes. It represents you.
Box: A yellow crate marked with an orange "X" across its front. It is the box you need to push.
Target: A black tile outlined in red, with a small red diamond shape in the center. It marks the destination where a box should be pushed.

# Current Step
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} text observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is shown in the image: <image>
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about the current situation before deciding on a final action. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose the final action and present it within <action> </action> tags.
"""

# Gym Cards
GYM_CARDS_NUMBERLINE_TEMPLATE = """
<image>You are playing a game called number line. You will see a target number and a current number in the image. And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number.

Your response should be a valid json file in the following format: \n{\n\"current number\": \"x\", \n\"target number\": \"x\", \n\"thoughts\": \"{first read out the current and target number, then think carefully about which action to choose}\", \n\"action\": \"-\" or \"+\" \n}
"""

GYM_CARDS_BLACKJACK_TEMPLATE = """
<image>You are an expert blackjack player helping to decide the optimal action based on the current game state displayed in the image. 

From the image, you can see:
- Dealer: two cards that belong to the dealer (one is visible; the other remains unseen).
- Player (you): the cards that belong to you.

Your goal is to make the best possible action to maximize your chances of beating the dealer without exceeding 21.
Your admissible actions are ["hit", "stand"], where "hit" means taking another card and "stand" means keeping your current hand.

Your response should be a valid json file in the following format:
{{
"thoughts": "Analyze the image to identify your cards and the dealer's visible card. Then, reason step-by-step based on standard blackjack strategy and the current game state to decide the best action.",
"action": "an admissible action"
}},
"""

GYM_CARDS_EZPOINTS_TEMPLATE = """
<image>You are an expert card game player helping to build a math formula that evaluates to 12 using onlt **two** playing cards. You can choose characters from ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. Your goal is to extend the current (incomplete) formula, one character at a time, so that when completed, it evaluates to 12. You are shown an image containing two playing cards. Each card represents a number: 1. Number cards (2–10) equal their face value. 2. Face cards ('J', 'Q', and 'K') are all treated as '10'. 

Important Rules: 
1. You can only use the two numbers shown on the cards (no other numbers). 
2. Each number can only be used **once** in the formula. 
3. You must build the formula step by step, by adding ONE character (number or operator) at a time to the end of the current formula. 
4. The appended character MUST ensure that the formula remains both mathematically valid and syntactically correct.

The current formula is {text_formula}. Now it's your turn to add a number or operator to the end of the formula.

Your response MUST be a valid json file in the following format: 
{{
"thoughts": "You should first describle the two numbers shown in the image and the current formula. Then step-by-step thinking which character (number or operator) should be added next based on the current formula and remaining numbers.",
"action": "next character (number or operator) to append"
}}
"""

GYM_CARDS_POINTS24_TEMPLATE = """
<image>You are an expert 24 points card game player. You are observing thee four cards in the image. You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. The number or operator you choose will be appended to the current formula. Note that 'J', 'Q', and 'K' count as '10'. Your goal is to output a formula that evaluates to 24, and each number can only be used once. 

The current formula is {text_formula}. Now it's your turn to add a number or operator to the end of the formula.

Your response should be a valid json file in the following format: 
{{
"cards": [x, y, z, w], 
"current formula": ,
"thoughts": First check whether the current formula equals 24. If the current formula equals 24, output '='. Otherwise consider which number or operator should be appended to the current formula to make it equal 24. 
"action": "number" or "operator" 
}}
"""