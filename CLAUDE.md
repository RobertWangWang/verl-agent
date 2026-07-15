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

This fork's ongoing research (proposal: `proposal_predictive_belief_memory_RL.md`, in Chinese) adds a dense, turn-level, self-supervised reward for memory quality: a memory summary is scored by whether it supports predicting *verifiable features* of future observations (rule-checked, no LLM judge), with potential-based shaping `r_pred(t) = Φ(s_t) − γΦ(s_{t−1})` added to the task reward. New code so far:

- `agent_system/environments/verifiable_features.py` — `BaseFeatureExtractor` + ALFWorld extractors (`object_seen`, `location_change`, `action_available`, `task_progress`), combined by `CompositeFeatureExtractor` with per-feature weights; factory `create_alfworld_feature_extractor()`. Each extractor implements `extract()` (parse observation/actions/info into a `VerifiableFeature`) and `verify()` (rule-based predicted-vs-actual check).
- `agent_system/memory/predictive_memory.py` — `PredictiveMemory` (SimpleMemory-style storage plus `predict_future()` / `verify_prediction()` / `compute_prediction_reward()` with potential-based shaping, weighted by `lambda_pred`) and `HybridMemory` (wrapper with an `enable_prediction` toggle). Exported from `agent_system/memory/__init__.py`. The prediction head is currently a placeholder (extracts current features as the "prediction") — a real prediction mechanism is still to be implemented.
- Tests: `tests/test_predictive_memory.py`, `tests/test_verifiable_features.py`.

**Status:** these modules are standalone and unit-tested but **not yet wired into** `env_manager.py`, the rollout loop, or reward computation. Baseline work for the eventual PS-GRPO comparison lives in `docs/grpo_baseline_guide.md` (GRPO ALFWorld baseline guide, Chinese) with matching scripts `examples/grpo_trainer/run_alfworld_full_32gb.sh` (2× RTX 5090 full run) and `examples/grpo_trainer/run_alfworld_mini.sh` (2× RTX 3090 smoke test to verify the rollout→reward→update loop; its comments explain fp32 actor memory math and why `VLLM_ATTENTION_BACKEND=XFORMERS` must not be set on vLLM 0.11).

## Configuration

Base config is `verl/trainer/config/ppo_trainer.yaml` (Hydra); run scripts override on the command line. Agent-specific keys:

- `env.*` — environment name (`env.env_name=alfworld/AlfredTWEnv`), `max_steps`, `history_length`, `rollout.n` (group size), per-env sub-configs (`env.sokoban.*`, `env.webshop.*`, `env.search.*` incl. retriever URL).
- `algorithm.gigpo.*` — `step_advantage_w`, `mode` (`mean_std_norm`/`mean_norm`), similarity-based step grouping (`enable_similarity`, `similarity_thresh`).
- `actor_rollout_ref.actor.use_invalid_action_penalty` / `invalid_action_penalty_coef` — penalize unparseable actions.
