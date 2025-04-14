# Environments

## ALFWorld
Install with pip (python3.10+):
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```

```bash
pip install alfworld[full]
pip install thinc==8.3.4
pip install vllm==0.8.2
```

> **Note:** Without the `full` extra, it will only install the text version of ALFWorld. To enable visual modalities, use `pip install alfworld[vis]`.

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
Play an Embodied-World (THOR) game:
```bash
alfworld-play-thor
```

## Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```

## Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```