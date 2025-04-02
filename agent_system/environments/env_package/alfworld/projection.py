import torch
import random
from typing import List
import re

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """
    valids = [0] * len(actions)
    for i in range(len(actions)):
        actions[i] = actions[i].lower()
        # TODO: need to figure this out
        if len(actions[i]) == 0:
            print("Action is empty!!!!")
            # randomly choose an action from the action list if illegal
            actions[i] = action_pools[i][random.randint(0, len(action_pools[i])-1)]
        else:
            try:
                index = actions[i].find('"action":')
                # string has the following format '"action": "look"\n'
                if index == -1:
                    # if we cannot find "action":, then we pick the last 30 characters
                    string = actions[i][-30:]
                else:
                    string = actions[i][index:]
                # post processing by removing the first and last part of the string
                for act in action_pools[i]:
                    if act in string:
                        actions[i] = act
                        # if found legal action, set valids to 1
                        valids[i] = 1
                        break

                # If no valid action found, randomly select from pool
                if valids[i] == 0:
                    actions[i] = random.choice(action_pools[i])
            except:
                # randomly choose an action from the action list if illegal
                actions[i] = action_pools[i][random.randint(0, len(action_pools[i])-1)]

    return actions, valids

# def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
#     """
#     An function to process the actions
#     actions: the list of actions to be processeed, it is a list of strings.
#     action_pools: the list of action pools, each pool is a list of strings.
#     """
#     valids = [0] * len(actions)
#     for i in range(len(actions)):
#         actions[i] = actions[i].lower()
#         # TODO: need to figure this out
#         if len(actions[i]) == 0:
#             print("Action is empty!!!!")
#             # randomly choose an action from the action list if illegal
#             actions[i] = random.choice(action_pools[i])
#         else:
#             try:
#                 # Extract action from <action> tags using regex
#                 match = re.search(r'<action>(.*?)</action>', actions[i], re.IGNORECASE)
                
#                 if match:
#                     extracted_action = match.group(1).strip().lower()
#                 else:
#                     extracted_action = actions[i][-30:]

#                 # Validate extracted action against action pool
#                 for valid_action in action_pools[i]:
#                     if valid_action in extracted_action:
#                         actions[i] = valid_action
#                         valids[i] = 1
#                         break
        
#                 # If no valid action found, randomly select from pool
#                 if valids[i] == 0:
#                     actions[i] = random.choice(action_pools[i])
#             except:
#                 # randomly choose an action from the action list if illegal
#                 actions[i] = random.choice(action_pools[i])

#     return actions, valids