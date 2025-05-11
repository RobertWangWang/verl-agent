<h1 style="text-align: center;">verl-agent</h1>

<p align="center"><strong>@ Nanyang Technological University</strong></p>

`verl-agent` is an extension of [veRL](https://github.com/volcengine/verl), specifically designed for training **large language model (LLM) agents** via reinforcement learning. `verl-agent` offers integration between LLM agents and interactive environments, enabling the development of reasoning agents in both visual and language-based tasks.

# Table of Contents

- [Key Features](#key-features)  
- [Installation](#installation)  
  - [1. Install veRL](#1-install-verl)  
  - [2. Install Supported Environments](#2-install-supported-environments)  
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

- **Scalable for Long-Horizon Interaction**

  Prior works like [RAGEN](https://github.com/RAGEN-AI/RAGEN) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1) concatenate all past states and responses, causing the input/output length to grow with each step.
  We implement a step-wise independent interaction paradigm that aligns with standard RL pipelines. Each step is processed individually, without concatenating the entire interaction history into a single input. This makes `verl-agent` highly scalable for long-horizon tasks.
  
- **Parallelized Gym-Style Environments and Group Environments**

  `verl-agent` provides a gym-style interface with support for parallelized environments. This enables high-throughput rollouts, speeding up training. In addtion, `verl-agent` introduces the concept of group environments. All environments within a group share identical initial states during `reset()`. This is especially useful for algorithms like GRPO and DAPO that requires multiple rollouts on the same state. You can configure the number of rollouts per group using the `env.rollou.n` in [ppo_trainer.yaml](/verl/trainer/config/ppo_trainer.yaml) config file.

- **Rich Suite of Environments**
  
  `verl-agent` offers a diverse set of interactive environments including embodied AI environments like [ALFWorld](https://github.com/alfworld/alfworld), visual games such as [Sokoban](https://github.com/mpSchrader/gym-sokoban) and [Gym Cards](https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/README.md), and digital interface control tasks like [WebShop](https://github.com/princeton-nlp/WebShop) and [AppWorld](https://github.com/stonybrooknlp/appworld/) (experimental). 

- **Diverse RL Algorithms**

  `verl-agent` includes implementations of various RL algorithms, such as GiGPO, GRPO, PPO, DAPO, and GiGPO, including variants with dynamic sampling and clip-higher.


- **Vision-Language Agent Support**

  Beyond text-based agents, `verl-agent` also supports training vision-language agents. This enables multi-modal reasoning in environments where both visual perception and language understanding are required.


# Installation
## 1. Install veRL
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

## 2. Install Supported Environments

Details for installing each environment are provided in the [Environment Setup Guide](agent_system/environments/README.md).

`verl-agent` supports the following environments: **ALFWorld**, **WebShop**, **Gym Cards**, **Sokoban**, and **APPWorld** (experimental).

> ⚠️ **Important:**  
To run an agent in any of these environments, you must first install and configure the corresponding environment. Please refer to the [Environment Setup Guide](agent_system/environments/README.md) for step-by-step installation instructions.

# Run Examples
## RL Training
We provide out-of-the-box scripts in the ["examples/"](examples/) directory for training agents in different environments.

Here are some examples:
### 1. GiGPO
GiGPO is our novel algorithm designed to support fine-grained credit assignment in long-horizon LLM agent training. It introduces a two-level grouping mechanism:
- Episode-level groups capture overall task success via total returns (like GRPO).
- Step-level groups use shared anchor states across trajectories to compute relative advantages for individual actions.

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
bash examples/gigpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_trainer/run_webshop.sh # WebShop
```
```bash
bash examples/gigpo_trainer/run_sokoban_visual.sh # Sokoban
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