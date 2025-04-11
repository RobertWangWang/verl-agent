from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import ALFWORLD_INIT_TEXT_OBS, ALFWORLD_TEXT_OBS

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    elif isinstance(tensor, (int, float, bool, Tuple, List)):
        tensor = np.array(tensor)
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")
    return tensor


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos

class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, env_name=None):
        """
        Initialize the environment manager.
        
        Parameters:
        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.
        - projection_f: A function that maps text actions to environment actions.
        - env_name (str): The name of the environment.
        """
        self.envs = envs
        self.projection_f = projection_f
        self.env_name = env_name

    def reset(self) -> Dict[str, Any]:
        """
        Reset all environments and return the initial observations.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'raw' (None or Any): Raw observation without any histories or additional info. (for GiGPO only).
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'raw': None}, infos
    
    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
        
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'raw' (None or Any): Raw observation without any histories or additional info. (for GiGPO only).
        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
        
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None, # TODO: Implement this if needed
            'image': next_obs,
            'raw': None # For GiGPO only. raw observation without any histories, hint, etc. Implement this if needed
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        return next_observations, rewards, dones, infos

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check if the total rewards are greater than 0.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        raise NotImplementedError("success_evaluator should be implemented in the subclass.")
    
    def save_image(self, image, step):
        """
        Save an image to a file.
        
        Parameters:
        - image (np.ndarray or torch.Tensor): The image to save.
        - path (str): The path to save the image.
        """
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")
        
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)
        
        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)


# Customizing the your own environment manager: 
class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        super().__init__(envs, projection_f, env_name)
    
    def reset(self) -> Dict[str, Any]:
        obs = self.envs.reset()
        observations = {'text': None, 'image': obs, 'raw': obs.clone()}
        if self.env_name == 'gym_cards/EZPoints-v0' or self.env_name == 'gym_cards/Points24-v0':
            observations['text'] = ["The current formula is empty. Now it's your turn to choose a number or operator as the beginning of the formula."] * len(obs)
        
        infos = [None] * self.envs.num_envs
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        if self.env_name == 'gym_cards/EZPoints-v0' or self.env_name == 'gym_cards/Points24-v0':
            next_observations['text'] = self.build_text_obs(infos)
            
        next_observations['raw'] = next_observations['image'].clone()

        return next_observations, rewards, dones, infos
    
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        episode_rewards = kwargs['episode_rewards']
        success = episode_rewards > 0
        return {'main': success}

    
    def build_text_obs(self, infos: Tuple[Dict]) -> List[str]:
        text_observations = []
        for info in infos:
            text_formula = ''.join(str(element) for element in info['Formula'])
            if text_formula == '' or text_formula == ' ':
                text_observation = "The current formula is empty. Now it's your turn to choose a number or operator as the beginning of the formula."
            else:
                text_observation = f"The current formula is \"{text_formula}\". Now it's your turn to add a number or operator to the end of \"{text_formula}\"."
            text_observations.append(text_observation)

        return text_observations


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
    
    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        if self.buffers is not None:
            del self.buffers
        self.buffers = [[] for _ in range(len(text_obs))]
        self.tasks = []
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'raw': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(actions, text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'raw': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False, history_length: int = 5) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init:
                obs = ALFWORLD_INIT_TEXT_OBS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    feedback = record["text_obs"]
                    action_history += f"[Action {step_number}: '{action}', Feedback {step_number}: '{feedback}']"
                obs = ALFWORLD_TEXT_OBS.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, actions, text_obs):
        for i in range(len(actions)):
            self.buffers[i].append({'action': actions[i], 'text_obs': text_obs[i]})

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['main']) == batch_size
        
        # Convert lists to numpy arrays
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['main'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[task].append(won_value)
                break

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(config.env.env_name, config.env.seed, config.data.train_batch_size, group_n,
                             config.env.gamma, log_dir=None, device='cpu', allow_early_resets=False, num_frame_stack=1)
        _val_envs = build_gymcards_envs(config.env.env_name, config.env.seed + 1000, config.data.val_batch_size, 1,
                            config.env.gamma, log_dir=None, device='cpu', allow_early_resets=False, num_frame_stack=1)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)


if __name__ == "__main__":
    env_name = "gym_cards"
    if env_name == "gym_cards":
        # Test GymCardEnvironmentManager
        env_num = 8
        group_n = 5
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        envs = build_gymcards_envs('gym_cards/EZPoints-v0', 0, env_num, group_n, 0.99, log_dir=None, device='cpu', allow_early_resets=False, num_frame_stack=1)
        projection_f = partial(gym_projection, env_name='gym_cards/EZPoints-v0')
        env_manager = GymCardEnvironmentManager(envs, projection_f, 'gym_cards/EZPoints-v0')
        obs, infos = env_manager.reset()
        for i in range(100):
            random_actions = [str(np.random.randint(0, 10)) for i in range(len(infos))]
            obs, rewards, dones, infos = env_manager.step(random_actions)
            env_manager.save_image(obs['image'], i)
        print("completed")
    elif env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import alfworld_projection
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        import time
        alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        env_num = 8
        group_n = 5
        time1 = time.time()
        envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n)
        # val_envs = build_alfworld_envs(alf_config_path, 1000, 4)
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, 'alfworld/AlfredThorEnv')
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        # val_env_manager = AlfWorldEnvironmentManager(val_envs, alfworld_projection, 'alfworld/AlfredTWEnv')
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset()
            for i in range(20):
                # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
                print("step: ", i)
                random_actions = [np.random.choice(env_manager.envs.get_admissible_commands[i]) for i in range(len(env_manager.envs.get_admissible_commands))]
                # step
                obs, rewards, dones, infos = env_manager.step(random_actions)
                if np.array(dones).any():
                    print("Episode completed")

                for k in range(len(infos)):
                    assert infos[k]['won'] == False
                if obs['image'] is not None:
                    env_manager.save_image(obs['image'], i)
                # print("obs['image'].shape: ", obs['image'].shape)
            time2 = time.time()
            print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
            print("completed")