# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`verl-agent` is an extension of [veRL](https://github.com/volcengine/verl) for training LLM agents with multi-turn RL (NeurIPS 2025 GiGPO paper). The upstream veRL framework lives in `verl/`; the agent-specific extension lives in `agent_system/` and `gigpo/`. The key design difference from frameworks like RAGEN/Search-R1 is a **step-independent rollout**: each step's LLM input is built fresh from the current observation plus a configurable memory/history summary, instead of concatenating the full interaction history — this keeps context length near-constant over long horizons (30–50 steps).

## Commands

```bash
# Install (Python 3.12 conda env; environments like WebShop need their own env — see README)
pip install -e .

# Lint (ruff, line-length 300, config in pyproject.toml)
ruff check verl agent_system

# Run a single test
pytest tests/test_verifiable_features.py -x
pytest tests/test_predictive_memory.py::TestName::test_case

# Train (each script prepares data then launches verl.trainer.main_ppo with Hydra overrides)
bash examples/gigpo_trainer/run_alfworld.sh   # also: run_webshop.sh, run_sokoban.sh, run_search.sh …
bash examples/grpo_trainer/run_alfworld.sh    # ppo_trainer/, dapo_trainer/, gspo_trainer/, rloo_trainer/ likewise
```

Training requires the target environment to be installed first (ALFWorld, WebShop, Search retriever server, Sokoban, Gym Cards, AppWorld — installation per environment is in the README; several need dedicated conda environments). Data preparation (`examples/data_preprocess/prepare.py`) only produces placeholder parquet files indicating modality ("text" vs "visual") and dataset size — actual agent inputs come from `env.step()` at rollout time, not from the dataset (Search-R1 is the exception: tasks are passed via `env_kwargs` in the parquet).

## Architecture

The training loop (`verl/trainer/ppo/ray_trainer.py`, entry `verl/trainer/main_ppo.py`) is standard veRL PPO with two agent-specific hooks:

1. **Rollout is replaced** by `agent_system/multi_turn_rollout/rollout_loop.py:TrajectoryCollector.multi_turn_loop()` — it drives batched environments step by step, calling the LLM once per step per env. `vanilla_multi_turn_loop` runs fixed-size batches; `dynamic_multi_turn_loop` supports dynamic sampling (DAPO-style).
2. **Advantage estimation** dispatches on `algorithm.adv_estimator` (`AdvantageEstimator` enum in ray_trainer.py). `gigpo` is implemented in `gigpo/core_gigpo.py`: episode-level groups (GRPO-like, over total return) plus step-level groups (repeated/similar states across trajectories) combined via `algorithm.gigpo.step_advantage_w`.

`agent_system/` layout:

- `environments/env_manager.py` — one `*EnvironmentManager` per environment (ALFWorld, WebShop, Search, Sokoban, GymCards, AppWorld), all subclassing `EnvironmentManagerBase` (`environments/base.py`). Each manager owns a memory instance, builds the per-step text observation (`build_text_obs()`) from prompt templates, and projects LLM text output into env actions (`projection_f` parses `<action>` tags). `make_envs(config)` at the bottom is the registry — new environments are registered here.
- `environments/env_package/<name>/` — gym-style, multi-process (Ray) parallel environment packages. Envs are grouped: all envs in a group share the same initial state on `reset()` (needed by GRPO/GiGPO; group size = `env.rollout.n`).
- `environments/prompts/<name>.py` — per-environment prompt templates.
- `memory/` — pluggable history management (`SimpleMemory`, `SearchMemory` in `memory.py`); consumed by env managers when building observations.
- `reward_manager/episode.py` — episode-level reward manager.

Adding a new environment = env package + prompts file + manager class registered in `make_envs()` (see README FAQ §4; WebShop is the reference implementation).

`recipe/` contains self-contained algorithm variants (HGPO, GraphGPO, DAPO, PRIME, …) with their own trainers/configs/run scripts — entry points like `recipe.hgpo.main_hgpo`, not `verl.trainer.main_ppo`.

## Active research: PS-GRPO (predictive-sufficiency memory rewards)

Governing proposal: `proposal_predictive_belief_memory_RL_v0.2_consensus.md` (**v0.2**, Chinese; supersedes `proposal_predictive_belief_memory_RL.md`). Central claim (H1) is an "adjudicator comparison": memory-reward signals judged by *environment future observations* (this method) vs downstream-task / anchor-QA / self-report / supervised-aux-loss baselines. Design docs: `docs/ps_grpo_integration_design.md` (reward pipeline, stages S0–S4 with acceptance records) and `docs/hiddenrule_gym_design.md` (synthetic env). Every experiment gets a dated record in `research_logs/` (Chinese) — read the latest ones for current status before planning work.

**Implemented and fully wired (S0–S4, unit-tested):**

- Reward pipeline: PS prompt templates ask for a `<predict>` block → `verifiable_features.parse_predict_block` (rule-based, no LLM judge) → verified against the *next* observation inside `AlfWorldEnvironmentManager.step()` (gate: `env.alfworld.prediction.enable`) → per-step `pred_reward`/`pred_accuracy` collected in `rollout_loop.py` → injected at trainer level by `apply_prediction_reward` + `pred_lambda_schedule` in `ray_trainer.py` (gate: `algorithm.pred_reward.enable`, λ anneal constant/linear/cosine). **r_pred must be injected per step-sample**: potential-based shaping telescopes to ≈0 at episode level.
- Feature protocols (`env.alfworld.prediction.feature_protocol`): `schema` (default; v0.2 task-agnostic — `objects_visible` bool, `visible_objects` open-set F1 log-only probe, `receptacle_state`; all tasks share one Φ) vs `task_targets` (v0.1 legacy, kept only to reproduce early pilots). All arms of one comparison must use the same protocol.
- HiddenRule-Gym (`agent_system/environments/env_package/hiddenrule/`): synthetic rooms-and-devices POMDP for the paper's main figures. 4 hidden-rule families (conj/seq/xor/count; train/probe family split), BFS oracle sharing the env's pure `transition()`, exact coverage C = I(Φ;s)/H(s) over non-terminal reachable states + greedy mask ladder (`coverage.py`), text-layer-only noise knobs (p_obs, obs_flip, noisy-TV sensor channels). verl integration (HRG-c) pending.
- Tests: `tests/test_{verifiable_features,predictive_memory,ps_alfworld_env_manager,ps_reward_injection,hiddenrule_core,hiddenrule_coverage}.py`.

**Hard-won gotchas:**

- Qwen3 + `enable_thinking=False` pre-injects an empty `<think>` block into the prompt side, so responses never contain `<think>` tags → set `env.alfworld.require_think_tags=False` for **all** Qwen3 runs, or valid_action_ratio is 0 by construction and every step eats the invalid penalty.
- Run scripts consume `$1` (engine) then pass `$@` to Hydra — always give single-line commands; multi-line paste breakage has silently dropped overrides before.
- **Every experiment needs its own `trainer.default_local_dir`** — verl defaults to `resume_mode=auto`, so a PS arm pointed at the baseline's ckpt dir silently resumes the baseline's weights (bit us on the 4B server 2026-07-18). The 1.7B scripts derive both `experiment_name` and `default_local_dir` from one `EXPERIMENT` env var; copy that pattern.
- **Config alignment across scales**: the 4B/8B servers actually run hand-modified scripts with `train 8 × group 4 = 32 traj/step, val 32` — NOT the repo 4B script's 16×8. All ALFWorld comparison arms (1.7B/4B/8B) must use 8/32/4. Repo `run_alfworld_qwen3_1p7b_2gpu.sh` encodes it; server scripts are hand-edited on the boxes.
- **Queue-script race**: arming a PS auto-queue while the baseline is still in data-prepare (no `main_ppo` yet) makes the wait-for-exit loop pass instantly → PS arm collides with baseline. Fixed in `queue_alfworld_qwen3_1p7b_ps.sh` (waits for main_ppo to appear first); 4B/8B queue scripts still have the race if armed early.
- `pgrep`/`pkill -f main_ppo` self-match: use `mai[n]_ppo`. Over SSH, `pkill -f` also matches the SSH session's own remote command line and kills it.
- Non-interactive SSH to the cloud boxes has no conda env and no HF mirror: prefix `PATH=/root/miniconda3/envs/verl-agent/bin:$PATH`; HF hub retries then falls back to local cache.
- TextWorld env workers leak ~1MB/step/worker RAM (no plateau); the local box has a persistent 256GB NVMe swapfile absorbing the cold pages. At 64 workers (8/32/4 config) the leak is only ~7GB per 150-step run — negligible. Restart-on-checkpoint playbook (backup plan) in `research_logs/2026-07-14_ps_grpo_s3_baseline.md`.
- verl's `perf/max_memory_*_gb` metrics are summed across GPUs, not per-card.
- Val with 32 episodes + sampled decoding (temp 0.4) has ±9% single-point noise: quote late-window means, never the final point. Clean endpoints come from a 140-games `val_only` eval on the saved ckpt.

**Results so far:**
- Qwen2.5-1.5B GRPO ALFWorld baseline (2×5090, 150 steps, 128 traj/step): final val **67.2%** (weakness: look_at_obj_in_light 0%). Pipeline/learning-curve reference only — not comparable to the Qwen3 arms (4× their data budget).
- **HRG pilot final (Qwen3-1.7B, 3 arms, 150 steps each)**: all three arms statistically at the 7.7% random floor. **A** (pure GRPO): gradient starvation (grad_norm 0.05). **B** (PS, generic Φ): pred saturates 0.99 in ~20 steps, F1 probe **collapses 0.51→0.24**. **C** (task_done upweighted): pred saturates identically, but F1 probe **holds ~0.49**. Core claim: reward weights control *which* beliefs are maintained, but reallocation within a non-covering Φ family never converts to success — **coverage C is the decisive variable, not feature weights**. Falsifiable predictions registered for arm D (vault_openable upper-bound Φ) and HRG@4B. See `research_logs/2026-07-17_hrg_pilot_grpo_vs_ps.md`.
- **Qwen3-4B ALFWorld baseline final** (8×RTX Pro 6000, actual config 8/32/4 = 32 traj/step): 150/150 in ~18h, zero crashes; ~315 traj/h (P0 gate passed 2.1×); train success ~24%→~59%; val last-6 mean **≈50%** (peak 65.6% @125). PS arm (`qwen3_4b_ps_grpo`, util 0.85, own ckpt dir) running from scratch. Baseline step-150 ckpt backed up on a third machine (user-managed). See `research_logs/2026-07-17_qwen3_4b_alfworld.md` §5–6.
- **Qwen3-8B ALFWorld baseline** (8×RTX 6000D, same 8/32/4): running healthy ~590s/step (~24h total); PS queue armed with dedicated ckpt dir.
- **Qwen3-1.7B ALFWorld pair** (local 2×5090, same 8/32/4): baseline running + PS auto-queue armed — completes the 1.7B→4B→8B scale ladder for the adjudicator table.
- Entropy across Qwen3 scales: 1.7B 0.19 → 4B 0.141 → **8B ~0.20** — the sharpens-with-scale trend does NOT continue at 8B.

Compute justification memos (8-GPU necessity, measured-data based): `docs/8gpu_compute_justification.md` (CN) / `_en.md` (EN).

Scripts: `run_alfworld_mini.sh` / `run_hiddenrule_mini.sh` (2×5090 smokes), `run_alfworld_full_32gb.sh` (2×5090 full, legacy 128 traj/step), `run_alfworld_qwen3_1p7b_2gpu.sh` + `queue_alfworld_qwen3_1p7b_ps.sh` (2×5090, aligned 8/32/4), `run_alfworld_qwen3_4b_8gpu.sh` + `queue_alfworld_qwen3_4b_ps.sh` (8×96GB server), `run_alfworld_qwen3_8b_8gpu.sh` + `queue_alfworld_qwen3_8b_ps.sh` (8×6000D server; server-side copies are hand-hardened).

## Configuration

Base config is `verl/trainer/config/ppo_trainer.yaml` (Hydra); run scripts override on the command line. Agent-specific keys:

- `env.*` — environment name (`env.env_name=alfworld/AlfredTWEnv`), `max_steps`, `history_length`, `rollout.n` (group size), per-env sub-configs (`env.sokoban.*`, `env.webshop.*`, `env.search.*` incl. retriever URL).
- `algorithm.gigpo.*` — `step_advantage_w`, `mode` (`mean_std_norm`/`mean_norm`), similarity-based step grouping (`enable_similarity`, `similarity_thresh`).
- `actor_rollout_ref.actor.use_invalid_action_penalty` / `invalid_action_penalty_coef` — penalize unparseable actions.
