# Environments

## Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.4.0
```
<!-- conda create -n verl python==3.12
conda activate verl
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.4.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 uninstall torch
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -->

## ALFWorld
Install with pip (python3.10+):

```bash
pip install alfworld[full]
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