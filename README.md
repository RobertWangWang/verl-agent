<h1 align="center">verl-agent</h1>

`verl-agent` is an extension of [veRL](https://github.com/volcengine/verl), specifically designed for training **large language model (LLM) agents via reinforcement learning (RL)**. `verl-agent` provides a **diverse set of RL algorithms** and a **rich suite of agent environments**, enabling the development of reasoning agents in both visual and text-based tasks.

Unlike prior approaches that concatenate full interaction histories, `verl-agent` processes each step independently and is therefore highly scalable for **very long-horizon, multi-turn RL training** (e.g., tasks in ALFWorld can require up to 50 steps to complete).

# Table of Contents

- [Key Features](#key-features)
- [Results](#results)  
- [Installation](#installation)  
  - [Install veRL](#install-verl)  
  - [Install Supported Environments](#install-supported-environments)  
    - [1. ALFWorld](#1-alfworld)  
    - [2. WebShop](#2-webshop)  
    - [3. Sokoban](#3-sokoban)  
    - [4. Gym Cards](#4-gym-cards)  
    - [5. APPWorld (Experimental)](#5-appworld-experimental)  
- [Run Examples](#run-examples)  
  - [RL Training](#rl-training)  
    - [1. GiGPO](#1-gigpo)  
    - [2. GRPO](#2-grpo)  
    - [3. PPO](#3-ppo)  
    - [4. DAPO](#4-dapo)  
    - [5. GiGPO (dynamic)](#5-gigpo-dynamic)  
  - [Prompt-based Agent with GPT-4o](#prompt-based-agent-with-gpt-4o)  
- [Acknowledgement](#acknowledgement)

# Key Features

- **Multi-Turn Agent-Environment Interaction**

  `verl-agent` supports multi-step interactive loops between agents and environments. Agents perceive environmental feedback after each step, forming the basis for reinforcement learning.

- **Scalable for Very Long-Horizon, Multi-Turn Optimization**

  Prior works like [RAGEN](https://github.com/RAGEN-AI/RAGEN) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1) concatenate the entire history of states and responses. This causes the input/output length to grow rapidly with the number of turns, making them difficult to scale to long-horizon scenarios.
  We implement a step-wise independent interaction paradigm that aligns with standard RL pipelines. Each step is processed individually, without concatenating the entire interaction history into a single input. This makes `verl-agent` highly scalable for long-horizon tasks.
  
- **Parallelized Gym-Style Environments and Group Environments**

  `verl-agent` provides a gym-style interface with support for parallelized environments. This enables high-throughput rollouts, speeding up training. In addtion, `verl-agent` introduces the concept of group environments. All environments within a group share identical initial states during `reset()`. This is especially useful for algorithms like GRPO and DAPO that requires multiple rollouts on the same state. You can configure the number of rollouts per group using the `env.rollou.n` in [ppo_trainer.yaml](/verl/trainer/config/ppo_trainer.yaml) config file.

- **Diverse RL Algorithms**

  `verl-agent` includes implementations of various RL algorithms, such as [GiGPO](https://github.com/langfengQ/verl-agent), [GRPO](https://arxiv.org/abs/2402.03300), [PPO](https://arxiv.org/abs/1707.06347), [DAPO](https://arxiv.org/abs/2503.14476), and their variants with dynamic sampling and clip-higher.


- **Rich Suite of Environments**
  
  `verl-agent` offers a diverse set of interactive environments including embodied AI environments like [ALFWorld](https://github.com/alfworld/alfworld), visual games such as [Sokoban](https://github.com/mpSchrader/gym-sokoban) and [Gym Cards](https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/README.md), and digital interface control tasks like [WebShop](https://github.com/princeton-nlp/WebShop) and [AppWorld](https://github.com/stonybrooknlp/appworld/) (experimental). 

- **Vision-Language Agent Support**

  Beyond text-based agents, `verl-agent` also supports training vision-language agents. This enables multi-modal reasoning in environments where both visual perception and language understanding are required.

# Results
| Algorithm | Task | Model | Success Rate | Training Log | Model Checkpoint [Coming Soon] |
|-|-|-|-|-|-|
| GiGPO | ALFWorld | Qwen2.5-1.5B-Instruct | 86.1% | [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/78zz4sc9) | ![HF](https://img.shields.io/badge/HuggingFace-model-orange?logo=huggingface) |
| GiGPO | WebShop| Qwen2.5-1.5B-Instruct | 67.4% | [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/zfnvpvxe)  | ![HF](https://img.shields.io/badge/HuggingFace-model-orange?logo=huggingface) |
| GiGPO | Sokoban [6x6]| Qwen2.5-VL-3B-Instruct | 81.0% | [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/xm92tyea)  | ![HF](https://img.shields.io/badge/HuggingFace-model-orange?logo=huggingface) |
| GiGPO | NumberLine | Qwen2.5-VL-3B-Instruct | 100.0% | [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/81qzsc3n)| ![HF](https://img.shields.io/badge/HuggingFace-model-orange?logo=huggingface) |
| GiGPO | EZPoints | Qwen2.5-VL-3B-Instruct | 100.0% | [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/k0y51zei)| ![HF](https://img.shields.io/badge/HuggingFace-model-orange?logo=huggingface) |


# Installation
## Install veRL
```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install FlashAttention
pip3 install flash-attn --no-build-isolation

# Install verl-agent
pip3 install -e .

# Install compatible vLLM
pip3 install vllm==0.8.2
```

## Install Supported Environments
<!-- 
Details for installing each environment are provided in the [Environment Setup Guide](agent_system/environments/README.md).

`verl-agent` supports the following environments: **ALFWorld**, **WebShop**, **Gym Cards**, **Sokoban**, and **APPWorld** (experimental). -->

> ⚠️ **Important:** 
To run an agent in any of these environments, you must first install and configure the corresponding environment. We strongly recommend installing ***each environment in its own dedicated conda environment*** to avoid potential package version conflicts.

### 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```

```bash
pip install alfworld
pip install thinc==8.3.4
pip install vllm==0.8.2
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
---

### 2. WebShop
WebShop requires Python 3.9, so begin by creating a new `verl-agent-webshop` environment
```bash
conda create -n verl-agent-webshop python==3.9.18 -y
conda activate verl-agent-webshop
```

Install WebShop
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`.
Or you may need to manually download the files.


Verify that WebShop was installed correctly by running:
```bash
python run_web_agent_text_env.py
```

After WebShop is installed, return to the root directory of the repository and install the verl package in `verl-agent`:
```bash
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
```
The warnings can be safely ignored.

---
### 3. Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```
---
### 4. Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```
---
### 5. APPWorld (Experimental)
Install APPWorld package in `verl-agent` (some warnings may be raised, you can ignore them)
```bash
cd repo_root/
cd ./agent_system/environments/env_package/appworld/appworld
pip install -e .
python -m appworld.cli install
appworld download data

cd repo_root/
appworld download data
```

Refresh dependencies in the `verl-agent` environment:
```bash
cd repo_root/
pip install -e .
pip install vllm==0.8.2
```
You can ignore the warning of incompatiblity for appworld, because we don't run appworld in `verl-agent` environment.

Create a Dedicated Conda Environment `appworld` for the APPWorld Server:
```bash
conda create -n appworld python=3.12 -y
conda activate appworld

cd ./agent_system/environments/env_package/appworld/appworld
pip install -e .
python -m appworld.cli install
```


<!-- > ⚠️ **Important:**  
To run an agent in any of these environments, you must first install and configure the corresponding environment. Please refer to the [Environment Setup Guide](agent_system/environments/README.md) for step-by-step installation instructions. -->

# Run Examples
## RL Training
We provide out-of-the-box scripts in the ["examples/"](examples/) directory for training agents in different environments.

Here are some examples:
### 1. GiGPO
[GiGPO](https://github.com/langfengQ/verl-agent) is our novel algorithm designed to support fine-grained credit assignment in long-horizon LLM agent training. It introduces a two-level grouping mechanism:
- Episode-level groups capture overall task success via total returns (like GRPO).
- Step-level groups gather repeated states across trajectories to compute relative advantages for individual actions.

GiGPO is fully critic-free, maintains the same GPU memory footprint and LLM rollout cost as GRPO, yet achieves significantly better training efficiency and performance.

```bash
bash examples/gigpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_trainer/run_webshop.sh # WebShop
```
```bash
bash examples/gigpo_trainer/run_sokoban_visual.sh # Sokoban
```
### 2. GRPO
[GRPO](https://arxiv.org/abs/2402.03300) is a critic-free algorithm that estimates relative advantages based on a group of full episode trajectories.
```bash
bash examples/grpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/grpo_trainer/run_webshop.sh # WebShop
```
```bash
bash examples/grpo_trainer/run_sokoban_visual.sh # Sokoban
```
### 3. PPO
[PPO](https://arxiv.org/abs/1707.06347) is a classic actor-critic algorithm that updates the policy using a clipped objective to ensure stable learning. It requires a separate value network (critic) to estimate state values.
```bash
bash examples/ppo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/ppo_trainer/run_webshop.sh # WebShop
```
### 4. DAPO
[DAPO](https://arxiv.org/abs/2503.14476) is a critic-free algorithm that enhances GRPO with techniques like dynamic sampling and clip-higher.
```bash
bash examples/dapo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/dapo_trainer/run_webshop.sh # WebShop
```
### 5. GiGPO (dynamic)
GiGPO uses dynamic sampling and clip-higher from DAPO
```bash
bash examples/gigpo_dynamic_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_dynamic_trainer/run_webshop.sh # WebShop
```
```bash
bash examples/gigpo_dynamic_trainer/run_sokoban_visual.sh # Sokoban
```
## Prompt-based Agent with GPT-4o
We also provide a prompt-based GPT-4o agent.
```bash
bash examples/prompt_agent/run_gpt4o_agent.sh # ALFWorld
```

## Acknowledgement

We gratefully acknowledge the contributions of the [veRL](https://github.com/volcengine/verl) team for providing a solid RL infrastructure.

Special thanks to the [RAGEN](https://github.com/RAGEN-AI/RAGEN) project for their codebase, which inspired early design choices during the development of `verl-agent`.

We also thank the developers of [ALFWorld](https://github.com/alfworld/alfworld), [Sokoban](https://github.com/mpSchrader/gym-sokoban), [Gym Cards](https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards), [WebShop](https://github.com/princeton-nlp/WebShop), and [AppWorld](https://github.com/stonybrooknlp/appworld) for providing high-quality interactive environments used in this project.
