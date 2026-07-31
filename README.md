# The Dark Room in the Reward Channel

Code for the paper **"The Dark Room in the Reward Channel: Dense Prediction Rewards
Collapse GRPO-Trained LLM Agents, and the Channel, Not the Content, Decides What
Works"** ([arXiv:2607.21273](https://arxiv.org/abs/2607.21273)).

> **This repository is a fork.** It builds on
> [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) (GiGPO, NeurIPS
> 2025), which in turn extends [volcengine/verl](https://github.com/volcengine/verl).
> All credit for the agent training framework, the step-independent multi-turn
> rollout, and the GiGPO algorithm goes to the original authors. The upstream README
> is preserved at [`docs/README_upstream_gigpo.md`](docs/README_upstream_gigpo.md).
> This fork adds the experimental apparatus for our study and changes no upstream
> training semantics unless a flag below is switched on.

## What the paper shows

Rewarding an LLM agent for predicting its next observation is a popular recipe for
sparse-reward settings. Under GRPO's std-normalized advantages, it destroys the
policy: every run, at 1.7B/4B/8B, collapses into a "dark room" absorbing state
(prediction accuracy 1.0, success 0). One line of algebra locates the cause: inside
all-fail groups, z-scoring makes the advantage invariant to the shaping coefficient,
so a bounded reward becomes full-scale pressure. Removing only the std normalizer
restores baseline parity. The same signal delivered as an auxiliary teacher-forced
loss helps instead, and a placebo ladder shows the gain is content-free: what works
is the update, not the world-model information. Delivery channel decides; content is
inert.

## What this fork adds

| Component | Where |
|---|---|
| Prediction-sufficiency reward pipeline (`<predict>` block, rule-based verification, potential-difference shaping, trainer-level injection with λ scheduling) | `agent_system/environments/verifiable_features.py`, `agent_system/multi_turn_rollout/rollout_loop.py`, `verl/trainer/ppo/ray_trainer.py` |
| Auxiliary teacher-forced CE channel (gold collection, second update pass, placebo modes: `shuffle` / `random_vocab` / `random_tokens` / `half_gold`, Rademacher `noise_sign` falsifier, interference probe) | `agent_system/multi_turn_rollout/aux_sft.py` |
| Mean-only / decoupled / filtered advantage variants and collapse diagnostics (all-fail-group fraction, advantage fingerprints) | `verl/trainer/ppo/` |
| HiddenRule-Gym: synthetic POMDP with exactly computable feature coverage C = I(Φ;s)/H(s), coverage mask ladder, privileged belief probes with leakage audit | `agent_system/environments/env_package/hiddenrule/` |
| WebShop centralized HTTP environment service (shared SimServer, per-session namespacing) and thin client | `agent_system/environments/env_package/webshop/server.py`, `http_envs.py` |
| ScienceWorld wiring (one-JVM-per-session service, binarized terminal reward, native variation splits) | `agent_system/environments/env_package/sciworld/` |
| Anchor-QA and self-report control arms | `agent_system/environments/env_manager.py`, `agent_system/memory/` |
| Preregistration digests (full SHA256, committed before each run completed) | `prereg_hashes.txt` |
| Unit tests for the above (230+ green) | `tests/` |

## Reproducing the main results

Environment installation (ALFWorld, WebShop, ScienceWorld, etc.) follows the
[upstream README](docs/README_upstream_gigpo.md). Each experiment arm is a single
Hydra command; the recipes below are the core matrix.

```bash
# Baseline (GRPO, std normalization)
bash examples/grpo_trainer/run_alfworld_qwen3_4b_8gpu.sh

# Collapse arm: + prediction reward through the std channel
#   env.alfworld.prediction.enable=True algorithm.pred_reward.enable=True

# Rescue arm: same, with the single-factor switch
#   algorithm.norm_adv_by_std_in_grpo=False

# Auxiliary-loss arm (the gain side)
#   env.alfworld.prediction.enable=True env.alfworld.prediction.collect_gold=True
#   algorithm.aux_sft.enable=True

# Placebo ladder
#   algorithm.aux_sft.placebo_mode=shuffle | random_vocab | random_tokens
#   algorithm.aux_sft.noise_sign=True        # Rademacher falsifier
```

All comparison arms use the aligned 8×4 configuration (32 trajectories/step, 150
steps); see the paper's reproducibility statement and Appendix C for the full
configuration table. Formal endpoint numbers come from 140-game `val_only`
evaluations of the final checkpoints (seen and unseen splits).

## Preregistration

Every substantive arm was preregistered: the full SHA256 digest of the prediction
plaintext was committed to `prereg_hashes.txt` before the corresponding run
completed. The plaintexts ship with the paper's supplementary material, so every
"registered before outcome" claim in the paper is independently checkable against
this repository's git history.

## Citation

If you use the failure analysis, the variance-trajectory criterion, or the
prediction-reward pipeline, please cite the paper:

```bibtex
@article{wang2026darkroom,
  title={The Dark Room in the Reward Channel: Dense Prediction Rewards Collapse
         GRPO-Trained LLM Agents, and the Channel, Not the Content, Decides What Works},
  author={Wang, Yu},
  journal={arXiv preprint arXiv:2607.21273},
  year={2026}
}
```

Please also cite the frameworks this fork builds on:

```bibtex
@article{feng2025group,
  title={Group-in-Group Policy Optimization for LLM Agent Training},
  author={Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal={arXiv preprint arXiv:2505.10978},
  year={2025}
}
```

## License

Apache 2.0, inherited from [verl-agent](https://github.com/langfengQ/verl-agent)
and [verl](https://github.com/volcengine/verl).
