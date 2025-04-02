import os
import yaml
import torchvision.transforms as T
import gymnasium as gym
from gymnasium import spaces
from alfworld.agents.environment import get_environment
import numpy as np
import torch

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, num_processes):
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        self.multi_modal = env_type == 'AlfredThorEnv'
        env = get_environment(env_type)(config, train_eval='train')
        self.envs = env.init_env(batch_size=num_processes)
        # self.action_space = spaces.Discrete(len(ALF_ACTION_LIST))
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)
        # Add the previous admissible commands for step
        self.prev_admissible_commands = None
        self.envs.seed(seed)

    def step(self, actions):
        ## SZ.3.4: sanity checking legal action as rewards
        # action, legal_action = process_action(self.env, action, self.prev_admissible_commands)
        obs, scores, dones, infos = self.envs.step(actions)
        infos['observation_text'] = obs
        reward = self.compute_reward(infos)
        self.prev_admissible_commands = infos['admissible_commands']
        text_obs = list(obs)
        image_obs = self._get_obs() if self.multi_modal else None
        return text_obs, image_obs, reward, dones, infos

    def reset(
        self,
    ):
        obs, infos = self.envs.reset()
        infos['observation_text'] = obs
        self.prev_admissible_commands = infos['admissible_commands']
        text_obs = list(obs)
        image_obs = self._get_obs() if self.multi_modal else None
        return text_obs, image_obs, infos

    def _get_obs(self):
        image = get_obs_image(self.envs)
        return image
    
    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands
    
    def compute_reward(self, infos):
        # A function to compute the shaped reward for the alfworld environment
        # infos: the info returned by the environment
        ## Tentative rewards: r = success_reward * 10 + goal_conditioned_r - 1*illegal_action
        if self.multi_modal:
            reward = [10*float(infos['won'][i]) + float(infos['goal_condition_success_rate'][i]) for i in range(len(infos['won']))]
        else:
            reward = [10*float(infos['won'][i]) for i in range(len(infos['won']))]
        # reward = [reward]
        return torch.tensor(reward)

def get_encoded_text(observation_text, tokenizer, model):

    encoded_input = tokenizer(observation_text, return_tensors='pt')
    outputs = model(**encoded_input)
    cls_embeddings = outputs.last_hidden_state[:,0,:]

    return cls_embeddings

def get_concat(obs, infos, tokenizer, model, device):
    assert 'observation_text' in infos.keys(), 'observation_text not in infos!'
    obs_text = infos['observation_text']
    obs_text_encode = get_encoded_text(obs_text, tokenizer, model)
    obs_text_encode = obs_text_encode.to(device)
    obs_cat = torch.cat((obs.flatten(start_dim=1), obs_text_encode), dim=1)
    return obs_cat

def get_cards_concat(obs, infos, tokenizer, model, device):
    ## Need to move these codes to a CNN utils or something
    assert 'Formula' in infos[0].keys(), 'Formula not in infos!'
    infos = infos[0]
    formula_list = infos['Formula']
    formula = "".join([str("".join([str(x) for x in formula_list]))])
    obs_text_encode = get_encoded_text(formula, tokenizer, model).to(device)
    obs_cat = torch.cat((obs.flatten(start_dim=1), obs_text_encode), dim=1)
    return obs_cat

def build_alfworld_envs(alf_config_path, seed, num_processes):
    return AlfworldEnvs(alf_config_path, seed, num_processes)