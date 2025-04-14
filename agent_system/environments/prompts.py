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
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on targets, you win!

# Rules
1. **You Can Only Push Boxes**: You can't pull them, so plan ahead to avoid getting stuck.
2. **No Moving Through Walls**: You can't walk through or push boxes into walls (`#`).
3. **Avoid Traps**: Don't push boxes into corners or against walls where they can't be moved again.

# Current Step
Your current observation is: [{current_observation}]
Your admissible actions are ["up", "down", "left", "right"].

Now its your turn to make a move (ONE action only for the current step).
You should first reason step-by-step about the current situation before deciding on a final action. This reasoning process MUST be enclosed within <think> </think> tags. 
After reasoning, you should choose the final action that you think is the best. The final action MUST be enclosed within <action> </action> tags.
"""


# Sokoban
SOKOBAN_VISUAL_TEMPLATE = """
You are an expert agent operating in the Sokoban environment. Your goal is to push all the boxes onto the target spots. Once all boxes are on targets, you win!

# Rules
- You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
- You can't walk through or push boxes into walls.
- To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Visual Elements Description:
- Character: A small, green alien-like figure with two antennae and black eyes. It represents you.
- Box: A yellow crate marked with an orange "X" across its front. It is the box you need to push.
- Target: A black tile outlined in red, with a small red diamond shape in the center. It marks the destination where a box should be pushed.

# Current Step
Your current observation is shown in the image.
Your admissible actions are ["up", "down", "left", "right"].

Now its your turn to make a move (ONE action only for the current step).
You should first reason step-by-step about the current situation before deciding on a final action. This reasoning process MUST be enclosed within <think> </think> tags. 
After reasoning, you should choose the final action that you think is the best. The final action MUST be enclosed within <action> </action> tags.
"""

# Gym Cards
GYM_CARDS_NUMBERLINE_TEMPLATE = """
You are playing a game called number line. You will see a target number and a current number in the image. And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number.

Your response should be a valid json file in the following format: \n{\n\"current number\": \"x\", \n\"target number\": \"x\", \n\"thoughts\": \"{first read out the current and target number, then think carefully about which action to choose}\", \n\"action\": \"-\" or \"+\" \n}
"""

GYM_CARDS_BLACKJACK_TEMPLATE = """
You are an expert blackjack player helping to decide the optimal move based on the current game state. As shown in the image, you are given: 1. Player Cards: the cards that belong to you. 2. Dealer Cards: two cards that belong to the dealer (one is visible to you; the other remains unseen). \nGame Rules:\n1. Your goal is to beat the dealer by choosing between the actions: \"hit\" (draw another card) or \"stand\" (keep your current hand).\n2. The player whose cards number sum is closest to 21 (without exceeding) wins.\n3. If the sum of your cards exceeds 21, you bust and immediately lose.\n4. Number cards (2–10) are worth their face value. Face cards (J, Q, K) are each worth 10 points. An Ace (\"A\") counts as 1 or 11 (always choose the value that benefits you the most).\n You should try to increase your chances of winning by choosing \"hit\" as much as possible to get closer to 21. However, if you believe hitting will most likely cause you to bust, then you should choose \"stand\".

Your response should be a valid json file in the following format:\n{\n\"reasoning\": \"Reason step-by-step based on the current game state shown in the image and then determine your next action.\",\n\"action\": \"hit\" or \"stand\"\n}",
"""

GYM_CARDS_EZPOINTS_TEMPLATE = """
You are an expert card game player helping to build a math formula that evaluates to 12 using onlt **two** playing cards. You can choose characters from ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. Your goal is to extend the current (incomplete) formula, one character at a time, so that when completed, it evaluates to 12. You are shown an image containing two playing cards. Each card represents a number: 1. Number cards (2–10) equal their face value. 2. Face cards ('J', 'Q', and 'K') are all treated as '10'. \nImportant Rules: 1. You can only use the two numbers shown on the cards (no other numbers). 2. Each number can only be used **once** in the formula. 3. You must build the formula step by step, by adding ONE character (number or operator) at a time to the end of the current formula. 4. The appended character MUST ensure that the formula remains both mathematically valid and syntactically correct.

The current formula is {text_formula}. Now it's your turn to add a number or operator to the end of the formula.

Your response MUST be a valid json file in the following format: {\n\"reasoning\": \"You should first describle the two numbers shown in the image and the current formula. Then step-by-step thinking which character (number or operator) should be added next based on the current formula and remaining numbers.\", \n\"action\": \"next character (number or operator) to append\"\n}
"""

GYM_CARDS_POINTS24_TEMPLATE = """
You are an expert 24 points card game player. You are observing thee four cards in the image. You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. The number or operator you choose will be appended to the current formula. Note that 'J', 'Q', and 'K' count as '10'. Your goal is to output a formula that evaluates to 24, and each number can only be used once. 

The current formula is {text_formula}. Now it's your turn to add a number or operator to the end of the formula.

Your response should be a valid json file in the following format: \\{\n \"cards\": [x, y, z, w], \n\"current formula\": , \n\"thoughts\": {First check whether the current formula equals 24. If the current formula equals 24, output '='. Otherwise consider which number or operator should be appended to the current formula to make it equal 24.} \n\"action\": \"{number}\" or \"{operator}\" \n \\}
"""