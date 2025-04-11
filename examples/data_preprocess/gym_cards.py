# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the Geometry3k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='ezpoints', choices=['blackjack', 'numberLine', 'ezpoints', 'points24'])
    parser.add_argument('--local_dir', default='~/data/gym_cards')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--add_few_shot', default=False, type=bool)

    args = parser.parse_args()
    print(f"processing data for task: {args.task}; shot({args.add_few_shot})")
    args.local_dir = os.path.join(args.local_dir, args.task)
    args.local_dir = os.path.join(args.local_dir, 'few_shot' if args.add_few_shot else 'zero_shot')

    data_source = 'hiyouga/geometry3k'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train'].select(range(128))
    test_dataset = dataset['test'].select(range(256))

    # instruction_following = {
    #     "blackjack": "<image>You are a blackjack player. Given the current state, you can choose between ['stand', 'hit']. Your response should be a valid json file in the following format: \n{\n\"thoughts\": {first describe your total points and the dealer's total points then think about which action to choose}, \n\"action\": \"stand\" or \"hit\" \n}",
    #     "numberLine": "<image>You are playing a game called number line. You will see a target number and a current number in the image. And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. Your response should be a valid json file in the following format: \n{\n\"current number\": \"x\", \n\"target number\": \"x\", \n\"thoughts\": \"{first read out the current and target number, then think carefully about which action to choose}\", \n\"action\": \"-\" or \"+\" \n}",
    #     "points24": "<image>You are an expert 24 points card game player. You are observing thee four cards in the image. You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. The number or operator you choose will be appended to the current formula. Note that 'J', 'Q', and 'K' count as '10'. Your goal is to output a formula that evaluates to 24, and each number can only be used once. Your response should be a valid json file in the following format: \\{\n \"cards\": [x, y, z, w], \n\"current formula\": , \n\"thoughts\": {First check whether the current formula equals 24. If the current formula equals 24, output '='. Otherwise consider which number or operator should be appended to the current formula to make it equal 24.} \n\"action\": \"{number}\" or \"{operator}\" \n \\}",
    #     "ezpoints": "<image>You are an expert card game player. You are observing two cards in the image. You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. The number or operator you choose will be appended to the current formula. Note that 'J', 'Q', and 'K' count as '10'. Your goal is to output a formula that evaluates to 12, and each number can only be used once. Your response should be a valid json file in the following format: \\{\n \"cards\": [x, y], \n\"current formula\": , \n\"thoughts\": {First check whether the current formula 'z' is complete. If the current formula 'z' is complete, output '='. Otherwise consider which number or operator should be appended to the current formula to make it equal 12.} \n\"action\": \"{number}\" or \"{operator}\" \n \\}"
    #     }
    instruction_following = {
        # "blackjack": "<image>You are a blackjack player. Given the current state, you can choose between ['stand', 'hit']. Your response should be a valid json file in the following format: \n{\n\"thoughts\": {first describe your total points and the dealer's total points then think about which action to choose}, \n\"action\": \"stand\" or \"hit\" \n}",
        "numberLine": "<image>You are playing a game called number line. You will see a target number and a current number in the image. And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. Your response should be a valid json file in the following format: \n{\n\"current number\": \"x\", \n\"target number\": \"x\", \n\"thoughts\": \"{first read out the current and target number, then think carefully about which action to choose}\", \n\"action\": \"-\" or \"+\" \n}",
        # "points24": "<image>You are an expert 24 points card game player. You are observing thee four cards in the image. You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. The number or operator you choose will be appended to the current formula. Note that 'J', 'Q', and 'K' count as '10'. Your goal is to output a formula that evaluates to 24, and each number can only be used once. Your response should be a valid json file in the following format: \\{\n \"cards\": [x, y, z, w], \n\"current formula\": , \n\"thoughts\": {First check whether the current formula equals 24. If the current formula equals 24, output '='. Otherwise consider which number or operator should be appended to the current formula to make it equal 24.} \n\"action\": \"{number}\" or \"{operator}\" \n \\}",
        # "ezpoints": "<image>You are an expert card game player. You are given an image containing two playing cards, each representing a number. Your goal is to construct a mathematical formula that evaluates to 12, following strict rules. \nRules & Constraints: \n1. You can only use the numbers shown on the two cards in the image. You cannot use any numbers that are not present in the image. Numbered cards (2-10) represent their face value. Face cards (J, Q, K) count as 10. 2. You can use any mathematical operators from the following set: ['+', '*', '=']. 3. You must construct the formula step by step by appending only one number or operator per turn to the end of the current formula in the image. 4. Once the formula correctly evaluates to 12, you must append '=' to indicate completion. 5. Each number from the cards can be used only once in the formula. \nOutput Format: \nYour output must include four key elements: \n1. \"cards\": The two numbers from the given playing cards in the image. 2. \"current formula\": The mathematical expression currently being constructed in the image. 3. \"reasoning\": You FIRST think step by step about the reasoning process for the next number/operator, e.g., whether the current formula is complete, how to reach 12 given the current formula, and which number or operator to append next. 4. \"action\": The next number or operator to append to the formula. \nYour output must be a valid JSON file in the following structure:\n{\n\"cards\": [x, y],\n\"current formula\": 'z',\n\"reasoning\": \"step-by-step reasoning for the next number or operator\",\n\"action\": \"number\" or \"operator\"\n}"
        "blackjack": "<image>You are an expert blackjack player helping to decide the optimal move based on the current game state. As shown in the image, you are given: 1. Player Cards: the cards that belong to you. 2. Dealer Cards: two cards that belong to the dealer (one is visible to you; the other remains unseen). \nGame Rules:\n1. Your goal is to beat the dealer by choosing between the actions: \"hit\" (draw another card) or \"stand\" (keep your current hand).\n2. The player whose cards number sum is closest to 21 (without exceeding) wins.\n3. If the sum of your cards exceeds 21, you bust and immediately lose.\n4. Number cards (2–10) are worth their face value. Face cards (J, Q, K) are each worth 10 points. An Ace (\"A\") counts as 1 or 11 (always choose the value that benefits you the most).\n You should try to increase your chances of winning by choosing \"hit\" as much as possible to get closer to 21. However, if you believe hitting will most likely cause you to bust, then you should choose \"stand\". \nYour response should be a valid json file in the following format:\n{\n\"reasoning\": \"Reason step-by-step based on the current game state shown in the image and then determine your next action.\",\n\"action\": \"hit\" or \"stand\"\n}",
        "ezpoints": "<image>You are an expert card game player helping to build a math formula that evaluates to 12 using onlt **two** playing cards. You can choose characters from ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. Your goal is to extend the current (incomplete) formula, one character at a time, so that when completed, it evaluates to 12. You are shown an image containing two playing cards. Each card represents a number: 1. Number cards (2–10) equal their face value. 2. Face cards ('J', 'Q', and 'K') are all treated as '10'. \nImportant Rules: 1. You can only use the two numbers shown on the cards (no other numbers). 2. Each number can only be used **once** in the formula. 3. You must build the formula step by step, by adding ONE character (number or operator) at a time to the end of the current formula. 4. The appended character MUST ensure that the formula remains both mathematically valid and syntactically correct. \nYour response MUST be a valid json file in the following format: {\n\"reasoning\": \"You should first describle the two numbers shown in the image and the current formula. Then step-by-step thinking which character (number or operator) should be added next based on the current formula and remaining numbers.\", \n\"action\": \"next character (number or operator) to append\"\n}",
        # "ss": "{\n\"cards\": [x, y],\n\"current formula\": \"what the formula looks like so far\",\n\"reasoning\": \"First, check if the current formula already evaluates to 12. If so, output '='. If not, step-by-step thinking which number or operator should be added next, based on the current (incomplete) formula and unused numbers.\", \n\"action\": \"next number or operator to append, must be a **single** character\"\n}
        }
    few_shot_instruction = {
        "blackjack": "Here are two examples: 1. {\n\"reasoning\": \"My total is 11, which is an excellent position to hit, as drawing a 10-value card would give me 21. The dealer has a 10, a strong card, so I should take a risk to improve my hand.\",\n\"action\": \"hit\"\n}\n2. {\n\"reasoning\": \"My total is 17, which is a strong hand, but not unbeatable. The dealer's visible card is 9, meaning they have a high chance of reaching 19 or higher. However, hitting risks busting since any card 5 or higher will put me over 21. According to basic blackjack strategy, I should stand on 17 to minimize the risk.\",\n\"action\": \"stand\"\n}",
        "ezpoints": "Here are two examples: 1. {\n\"cards\": [4, 3],\n\"current formula\": \"3\",\n\"reasoning\": \"The current formula is '3'. I have already used '3', so only '4' remains. To reach 12, a good option is multiplication: 3*4=12. Therefore, I will add the '*' operator next.\",\n\"action\": \"*\"\n}\n2. {\n\"cards\": [10, 2],\n\"current formula\": \"\",\n\"reasoning\": \"The formula is empty. I need to choose a starting number. Since 10+2=12, addition is a viable path. I will start with '10'.\",\n\"action\": \"10\"\n}\n",
    }

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('problem')
            prompt = instruction_following[args.task]
            if args.add_few_shot:
                few_shot_prompt = few_shot_instruction[args.task]
                prompt = prompt + '\n' + few_shot_prompt
            # answer = example.pop('answer')
            images = example.pop('images')

            data = {
                "data_source": args.task,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "images": images,
                "ability": "game",
                # "reward_model": {
                #     "style": "rule",
                #     "ground_truth": answer
                # },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    # 'answer': answer,
                    # "question": problem,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
