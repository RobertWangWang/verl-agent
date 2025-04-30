<h1 style="text-align: center;">verl-agent</h1>

`verl-agent` is an extension of [veRL](https://github.com/volcengine/verl), specifically designed for training **large language model (LLM) agents** via reinforcement learning. `verl-agent` offers integration between LLM agents and interactive environments, enabling the development of reasoning agents in both visual and language-based tasks.

## Key Features
- Multi-Turn Agent-Environment Interaction

  `verl-agent` supports multi-step interactive loops between agents and environments. Agents perceive environmental feedback after each step, forming the basis for reinforcement learning.

- Scalable with Long-Horizon Steps

  Prior works like [RAGEN](https://github.com/RAGEN-AI/RAGEN) concatenate all past states and responses, causing the input/output length to grow with each step.
  We implement a step-wise independent interaction paradigm that aligns with standard RL pipelines. Each step is processed individually, without concatenating the entire interaction history into a single input. This makes `verl-agent` highly scalable for long-horizon tasks.
  
- Parallelized Gym-Style Environments and Group Environments

  `verl-agent` provides a gym-style interface with support for parallelized environments. This enables high-throughput rollouts, speeding up training. In addtion, `verl-agent` introduces the concept of group environments. All environments within a group share identical initial states during `reset()`. This is especially useful for algorithms like GRPO and DAPO that requires multiple rollouts on the same state. You can configure the number of rollouts per group using the `env.rollou.n` in [ppo_trainer.yaml](/verl/trainer/config/ppo_trainer.yaml) config file.

- Rich Agent Environment
  
  `verl-agent` offers a diverse set of interactive environments including embodied AI environments like [ALFWorld](https://github.com/alfworld/alfworld), visual games such as [Sokoban](https://github.com/mpSchrader/gym-sokoban) and [Gym Cards](https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/README.md), and digital interface control tasks like [WebShop](https://github.com/princeton-nlp/WebShop) and [AppWorld](https://github.com/stonybrooknlp/appworld/) (experimental). 

- Multi-Modal Agent Support

  Beyond text-based agents, `verl-agent` also supports training vision-language agents. This enables multi-modal reasoning in environments where both visual perception and language understanding are required.


## Installation
### 1. Create and Set Up a Conda Environment
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

### 2. Install Supported Environments

To enable environments such as ALFWorld, Gym Cards, Sokoban, and AppWorld, follow the setup instructions in the [Environment Setup](agent_system/environments/README.md).


## Run Examples
We provide out-of-the-box scripts to train agents in various environments.
### ALFWorld

```bash
bash examples/grpo_trainer/run_alfworld.sh # GRPO
```
```bash
bash examples/dapo_trainer/run_alfworld.sh # DAPO
```
```bash
bash examples/gigpo_trainer/run_alfworld.sh # GiGPO
```
```bash
bash examples/dynamic_gigpo_trainer/run_alfworld.sh # Dynamic GiGPO
```

### Sokoban

```bash
bash examples/grpo_trainer/run_sokoban_visual.sh # GRPO
```
```bash
bash examples/dapo_trainer/run_sokoban_visual.sh # DAPO
```
```bash
bash examples/gigpo_trainer/run_sokoban_visual.sh # GiGPO
```
```bash
bash examples/dynamic_gigpo_trainer/run_sokoban_visual.sh # Dynamic GiGPO
```

### Gym Cards

```bash
bash examples/grpo_trainer/run_ezpoints.sh # GRPO
```
```bash
bash examples/dapo_trainer/run_ezpoints.sh # DAPO
```
```bash
bash examples/gigpo_trainer/run_ezpoints.sh # GiGPO
```
```bash
bash examples/dynamic_gigpo_trainer/run_ezpoints.sh # Dynamic GiGPO
```

### WebShop

```bash
bash examples/grpo_trainer/run_webshop.sh # GRPO
```
```bash
bash examples/dapo_trainer/run_webshop.sh # DAPO
```
```bash
bash examples/gigpo_trainer/run_webshop.sh # GiGPO
```
```bash
bash examples/dynamic_gigpo_trainer/run_webshop.sh # Dynamic GiGPO
```

## Acknowledgement

We gratefully acknowledge the contributions of the [veRL](https://github.com/volcengine/verl) team for providing a solid RL infrastructure.

Special thanks to the [RAGEN](https://github.com/RAGEN-AI/RAGEN) project for their codebase, which inspired early design choices during the development of `verl-agent`.

We also thank the developers of [ALFWorld](https://github.com/alfworld/alfworld), [Sokoban](https://github.com/mpSchrader/gym-sokoban), [Gym Cards](https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards), [WebShop](https://github.com/princeton-nlp/WebShop), and [AppWorld](https://github.com/stonybrooknlp/appworld) for providing high-quality interactive environments used in this project.