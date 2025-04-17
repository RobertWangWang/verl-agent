# --------------------- ALFWorld --------------------- #
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

# --------------------- Sokoban --------------------- #
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

# --------------------- Gym Cards --------------------- #
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


# --------------------- Appworld --------------------- #
APPWORLD_INIT_TEMPLATE = """
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.


----------------------------------------------
Here is an example:

My name is: {supervisor_first_name} {supervisor_last_name}. My personal email is {supervisor_email} and phone number is {supervisor_phone_number}.

Your task is: What is the password for my Spotify account?

You:
<think>Okay. Lets first find which apps are available to get the password by looking at the app descriptions.</think>
<code>print(apis.api_docs.show_app_descriptions())</code>

Environment:
[
  {{
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  }},
  {{
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  }},
  ...
  {{
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  }},
  {{
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  }},
  ...
]

You:
<think>Looks like the supervisor app could help me with that. Lets see what apis are available under this app.</think>
<code>print(apis.api_docs.show_api_descriptions(app_name='supervisor'))</code>

Environment:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]

You:
<think>I can use `show_account_passwords` to get the passwords. Let me see its detailed specification to understand its arguments and output structure.</think>
<code>print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))</code>

Environment:
{{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {{
    'success': [{{'account_name': 'string', 'password': 'string'}}],
    'failure': {{'message': 'string'}}
  }}
}}

You:
<think>Okay, it requires no arguments. So I can just call it directly.</think>
<code>print(apis.supervisor.show_account_passwords())</code>

Environment:
[
  {{
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  }},
  {{
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  }},
  ...
]


You:
<think>So the Spotify password is an entry in the `passwords` list with the account_name=spotify.</think>
<code>spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)</code>


Environment:
dummy_spotify_pass

You:
<think>When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.</think>
<code>apis.supervisor.complete_task(answer=spotify_password)</code>

Environment:
Marked the active task complete.

----------------------------------------------

**Key instructions and disclaimers**:

1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. Code execution behaves like in a Jupyter Notebook (.ipynb) — it maintains a persistent execution context, so any variables, imports, or functions defined in previous steps will still be available in the current step.
4. You should not generate the entire solution in one go. You should write small chunks of code and only one chunk of code in every step, using the result of each previously executed code block to inform your next move. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current data or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.

Using these APIs, now begin writing code cells step-by-step — just like working in a Jupyter Notebook — to solve the task:

My name is: {supervisor_first_name} {supervisor_last_name}. My personal email is {supervisor_email} and phone number is {supervisor_phone_number}.

Your task is: {task_description}

Now it's your turn to generate code to solve the task.
You should first reason step-by-step about which APIs to call, what arguments to use, and how to build your code block to complete the task. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you present the solution code body within <code> </code> tags.
"""


APPWORLD_TEMPLATE = """
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.


----------------------------------------------
Here is an example:

My name is: {supervisor_first_name} {supervisor_last_name}. My personal email is {supervisor_email} and phone number is {supervisor_phone_number}.

Your task is: What is the password for my Spotify account?

You:
<think>Okay. Lets first find which apps are available to get the password by looking at the app descriptions.</think>
<code>print(apis.api_docs.show_app_descriptions())</code>

Environment:
[
  {{
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  }},
  {{
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  }},
  ...
  {{
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  }},
  {{
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  }},
  ...
]

You:
<think>Looks like the supervisor app could help me with that. Lets see what apis are available under this app.</think>
<code>print(apis.api_docs.show_api_descriptions(app_name='supervisor'))</code>

Environment:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]

You:
<think>I can use `show_account_passwords` to get the passwords. Let me see its detailed specification to understand its arguments and output structure.</think>
<code>print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))</code>

Environment:
{{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {{
    'success': [{{'account_name': 'string', 'password': 'string'}}],
    'failure': {{'message': 'string'}}
  }}
}}

You:
<think>Okay, it requires no arguments. So I can just call it directly.</think>
<code>print(apis.supervisor.show_account_passwords())</code>

Environment:
[
  {{
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  }},
  {{
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  }},
  ...
]


You:
<think>So the Spotify password is an entry in the `passwords` list with the account_name=spotify.</think>
<code>spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)</code>


Environment:
dummy_spotify_pass

You:
<think>When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.</think>
<code>apis.supervisor.complete_task(answer=spotify_password)</code>

Environment:
Marked the active task complete.

----------------------------------------------

**Key instructions and disclaimers**:

1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. Code execution behaves like in a Jupyter Notebook (.ipynb) — it maintains a persistent execution context, so any variables, imports, or functions defined in previous steps will still be available in the current step.
4. You should not generate the entire solution in one go. You should write small chunks of code and only one chunk of code in every step, using the result of each previously executed code block to inform your next move. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current data or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.

Using these APIs, now begin writing code cells step-by-step — just like working in a Jupyter Notebook — to solve the task:

My name is: {supervisor_first_name} {supervisor_last_name}. My personal email is {supervisor_email} and phone number is {supervisor_phone_number}.

Your task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} codes you generated and the corresponding environment feedback: {action_history}

You are now at step {current_step}, and it's your turn to generate code for this step.
You should first reason step-by-step about the last {history_length} histories, and think about which APIs to call, what arguments to use, and how to build your code block to complete the task. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you present the solution code body within <code> </code> tags.
"""