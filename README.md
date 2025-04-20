<h1 style="text-align: center;">verl-agent</h1>

`verl-agent` is an extension of [veRL](https://github.com/volcengine/verl), specifically designed for training **large language model (LLM) agents** via reinforcement learning. `verl-agent` offers integration between LLM agents and interactive environments, enabling the development of reasoning agents in both visual and language-based tasks.

## Key Features
- Multi-Turn Agent-Environment Interaction

  `verl-agent` supports multi-step interactive loops between agents and environments. Agents perceive environmental feedback after each step, forming the basis for reinforcement learning.

- Scalable with Long-Horizon Steps

  Prior works like [RAGEN](https://github.com/RAGEN-AI/RAGEN) concatenate all past states and responses, causing the input/output length to grow with each step.
  We implement a step-wise independent interaction paradigm that aligns with standard RL pipelines. Each step is processed individually, without concatenating the entire interaction history into a single input. This makes `verl-agent` highly scalable for long-horizon tasks.
  
- Parallelized Gym-Style Environments

  We provide a gym-style interface with support for parallelized environments. This enables high-throughput rollouts, speeding up training. 

- Group Environments for Consistent Rollouts

  `verl-agent` introduces the concept of group environments. All environments within a group share identical initial states during `reset()`, ensuring consistent evaluation or policy training. This is especially useful for algorithms like GRPO that requires multiple rollouts on the same state. You can configure the number of rollouts per group using the `env.rollou.n` in [ppo_trainer.yaml](/verl/trainer/config/ppo_trainer.yaml) config file.

- Rich Agent Environment Support
  
  `verl-agent` offers a diverse set of interactive environments including embodied AI environments like [ALFWorld](https://github.com/alfworld/alfworld), visual games such as [Sokoban](https://github.com/mpSchrader/gym-sokoban) and [Gym Cards](https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/README.md), and digital interface control tasks like [WebShop](https://github.com/princeton-nlp/WebShop) and [AppWorld](https://github.com/stonybrooknlp/appworld/) (experimental). 

- Multi-Modal Agent Support

  Beyond text-based agents, `verl-agent` also supports training vision-language agents. This enables multi-modal reasoning in environments where both visual perception and language understanding are required.


## Installation
### 1. Create and Set Up a Conda Environment
```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent
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
bash examples/gigpo_trainer/run_alfworld.sh
```
```bash
bash examples/grpo_trainer/run_alfworld.sh
```

### Sokoban

```bash
bash examples/gigpo_trainer/run_sokoban_visual.sh
```

```bash
bash examples/grpo_trainer/run_sokoban_visual.sh
```

### Gym Cards

```bash
bash examples/gigpo_trainer/run_numberline.sh
```

```bash
bash examples/grpo_trainer/run_numberline.sh
```

### WebShop
```bash
bash examples/gigpo_trainer/run_webshop.sh
```

```bash
bash examples/grpo_trainer/run_webshop.sh
```

### Appworld (Experimental)
```bash
bash examples/gigpo_trainer/run_appworld.sh
``` 

```bash
bash examples/grpo_trainer/run_appworld.sh
``` 

## Acknowledgement

We gratefully acknowledge the contributions of the [veRL](https://github.com/volcengine/verl) team for providing a solid RL infrastructure.

Special thanks to the [RAGEN](https://github.com/RAGEN-AI/RAGEN) project for their codebase, which inspired early design choices during the development of `verl-agent`.

