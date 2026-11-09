"""Render the ALFWorld rebuttal matrix; execute one run only with --execute."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Arm:
    model: str = "4b"
    prediction_prompt: bool = True
    prediction_reward: bool = False
    estimator: str = "step"
    norm_std: bool = True
    drop_first: bool = False
    success_only: bool = False
    match_rms: bool = False

    @property
    def steps(self):
        return 150 if self.model == "4b" else 60


ARMS = {
    "A00": Arm(),
    "A01": Arm(prediction_reward=True),
    "A02": Arm(prediction_reward=True, drop_first=True),
    "A03": Arm(norm_std=False),
    "A04": Arm(prediction_reward=True, norm_std=False),
    "A05": Arm(prediction_reward=True, success_only=True),
    "A06": Arm(prediction_reward=True, estimator="gdpo"),
    "A07": Arm(prediction_reward=True, match_rms=True),
    "A08": Arm(prediction_prompt=False),
    "A09": Arm(prediction_prompt=False, norm_std=False),
    "A10": Arm(prediction_reward=True, norm_std=False, drop_first=True),
    "B00": Arm(model="1p7b"),
    "B01": Arm(model="1p7b", prediction_reward=True),
    "B02": Arm(model="1p7b", estimator="trajectory"),
    "B03": Arm(model="1p7b", prediction_reward=True, estimator="trajectory"),
    "B04": Arm(model="1p7b", prediction_reward=True, drop_first=True),
    "B05": Arm(model="1p7b", prediction_reward=True, estimator="trajectory", drop_first=True),
}
CORE = [f"A{i:02d}" for i in range(5)] + [f"B{i:02d}" for i in range(4)]
PACKAGES = {"core": CORE, "expanded": CORE + ["A05", "A06", "A07"], "all": list(ARMS)}


def select_arms(value):
    names = PACKAGES.get(value, value.split(","))
    if not names or len(names) != len(set(names)):
        raise ValueError("arms must be a nonempty list without duplicates")
    missing = set(names) - ARMS.keys()
    if missing:
        raise ValueError(f"unsupported arms: {sorted(missing)}; W/X environments are not implemented by this launcher")
    return names


def parse_seeds(value):
    seeds = [int(v) for v in value.split(",")]
    if not seeds or len(seeds) != len(set(seeds)) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    return seeds


def compose_config(overrides):
    # Hydra composition validates every override without importing main_ppo,
    # starting Ray, loading a model or touching CUDA.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(REPO_ROOT / "verl/trainer/config"), version_base=None):
        return OmegaConf.to_container(compose(config_name="ppo_trainer", overrides=overrides), resolve=True)


def override(key, value):
    if isinstance(value, str):
        # These quotes are Hydra syntax; argv is passed without a shell.
        encoded = json.dumps(value)
    elif isinstance(value, bool):
        encoded = str(value).lower()
    elif value is None:
        encoded = "null"
    else:
        encoded = str(value)
    return f"{key}={encoded}"


def build_run(name, seed, args):
    arm = ARMS[name]
    model = getattr(args, f"model_{arm.model}")
    if model is None:
        raise ValueError(f"{name} requires --model-{arm.model.replace('p', '.')}")
    model = model.expanduser().resolve()
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError(f"local model directory with config.json required: {model}")
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(not g.isdigit() for g in gpu_ids):
        raise ValueError("--gpus must be distinct comma-separated CUDA device indices")
    if len(gpu_ids) not in (1, 2, 4, 8):
        raise ValueError("supported GPU counts are 1, 2, 4, 8 (TP=1)")
    if args.num_cpus < 1 or args.threads < 1 or args.micro_batch < 1 or args.logprob_micro_batch < 1:
        raise ValueError("CPU, thread and microbatch counts must be positive")
    if (256 // len(gpu_ids)) % args.micro_batch:
        raise ValueError("microbatch must divide the per-rank PPO minibatch 256 / GPU count")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("GPU memory utilization must be between zero and one")
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 5:
        raise ValueError("--smoke-steps must be between 1 and 5")
    steps = args.smoke_steps or arm.steps
    run_id = f"{name}_s{seed}"
    if args.smoke_steps is not None:
        run_id += f"_smoke{steps}"
    root = args.output_root.expanduser().resolve() / run_id
    ray_dir = Path("/tmp/verl-rebuttal") / hashlib.sha256(str(root).encode()).hexdigest()[:16]
    eval_seed = (seed + 1000) % 2**32
    values = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.rebuttal.enable": True,
        "algorithm.rebuttal.estimator": arm.estimator,
        "algorithm.rebuttal.epsilon": 1e-6,
        "algorithm.rebuttal.match_advantage_rms": arm.match_rms,
        "algorithm.norm_adv_by_std_in_grpo": arm.norm_std,
        "algorithm.pred_reward.enable": arm.prediction_reward,
        "algorithm.pred_reward.lambda": 0.1,
        "algorithm.pred_reward.success_only": arm.success_only,
        "algorithm.pred_reward.decoupled_advantage": False,
        "algorithm.pred_reward.anneal.style": "constant",
        "algorithm.aux_sft.enable": False,
        "algorithm.filter_all_fail_groups": False,
        "algorithm.filter_groups.enable": False,
        "algorithm.use_kl_in_reward": False,
        "data.train_files": str(root / "data/train.parquet"),
        "data.val_files": str(root / "data/test.parquet"),
        "data.train_batch_size": 8,
        "data.val_batch_size": 32,
        "data.max_prompt_length": 1536,
        "data.max_response_length": 512,
        "data.return_raw_chat": True,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.seed": seed,
        "+data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.model.path": str(model),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 256,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": args.micro_batch,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": 0.01,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.use_invalid_action_penalty": True,
        "actor_rollout_ref.actor.invalid_action_penalty_coef": 0.1,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.seed": seed,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": args.logprob_micro_batch,
        "actor_rollout_ref.rollout.gpu_memory_utilization": args.gpu_memory_utilization,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.4,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": args.logprob_micro_batch,
        "actor_rollout_ref.ref.fsdp_config.param_offload": False,
        "env.env_name": "alfworld/AlfredTWEnv",
        "env.seed": seed,
        "env.eval_seed": eval_seed,
        "env.max_steps": 50,
        "env.rollout.n": 4,
        "env.resources_per_worker.num_cpus": 0.1,
        "env.alfworld.eval_dataset": "eval_in_distribution",
        "env.alfworld.require_think_tags": False,
        "env.alfworld.prediction.enable": arm.prediction_prompt,
        "env.alfworld.prediction.collect_gold": False,
        "env.alfworld.prediction.feature_protocol": "schema",
        "env.alfworld.prediction.reward_mode": "potential",
        "env.alfworld.prediction.drop_first_reward": arm.drop_first,
        "env.alfworld.self_report.enable": False,
        "env.alfworld.anchor_qa.enable": False,
        "trainer.seed": seed,
        "trainer.group_variance_telemetry.enable": True,
        "trainer.group_variance_telemetry.dir": str(root / "telemetry"),
        "trainer.critic_warmup": 0,
        "trainer.project_name": "rebuttal_alfworld",
        "trainer.experiment_name": run_id,
        "trainer.default_local_dir": str(root / "checkpoints"),
        "trainer.validation_data_dir": str(root / "validation"),
        "trainer.resume_mode": "disable",
        "trainer.resume_from_path": None,
        "trainer.n_gpus_per_node": len(gpu_ids),
        "trainer.nnodes": 1,
        "trainer.total_epochs": steps,
        "trainer.total_training_steps": steps,
        "trainer.test_freq": 1 if args.smoke_steps else 5,
        "trainer.save_freq": 1 if args.smoke_steps else (25 if arm.model == "4b" else 20),
        "trainer.val_before_train": True,
        "trainer.max_actor_ckpt_to_keep": 2,
        "ray_init.num_cpus": args.num_cpus,
        "+ray_init.address": "local",
        "+ray_init._temp_dir": str(ray_dir),
        "hydra.run.dir": str(root / "hydra"),
    }
    overrides = [override(k, v) for k, v in values.items()]
    overrides += [
        "trainer.logger=[console]",
        "env.alfworld.prediction.feature_weights={location_change:0.5,objects_visible:0.5,visible_objects:0.0,action_available:0.0,receptacle_state:0.0,task_progress:0.0}",
    ]
    env = {"CUDA_VISIBLE_DEVICES": args.gpus, "PYTHONHASHSEED": str(seed), "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    argv = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    prepare_argv = [sys.executable, "-m", "examples.rebuttal.prepare", "--output-dir", str(root / "data"), "--train-size", "8", "--val-size", "32"]
    return {
        "run_id": run_id,
        "arm": name,
        "arm_definition": asdict(arm),
        "status": "planned",
        "smoke_only": args.smoke_steps is not None,
        "seed": seed,
        "seed_protocol": {"model_init": seed, "data": seed, "env_train": seed, "env_eval": eval_seed, "rollout_train": seed, "rollout_eval": eval_seed, "worker_rng": "(model_init + global_rank) mod 2**32", "rollout_rng": "(rollout_seed + DP_rank) mod 2**32; same stream inside each TP group"},
        "paths": {"root": str(root), "ray_temp": str(ray_dir), "log": str(root / "train.log")},
        "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "environment": env,
        "prepare_argv": prepare_argv,
        "train_argv": argv,
        "train_command": "env " + " ".join(shlex.quote(f"{k}={v}") for k, v in env.items()) + " " + shlex.join(argv),
        "prepare_command": shlex.join(prepare_argv),
        "resolved_config": compose_config(overrides),
    }


def evidence():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    files = {}
    for directory in ("verl", "agent_system", "gigpo", "examples/rebuttal"):
        for path in sorted((REPO_ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}:
                files[str(path.relative_to(REPO_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = REPO_ROOT / "rebuttal最大实验覆盖.md"
    versions = {}
    for package in ("torch", "vllm", "ray", "transformers", "hydra-core", "numpy", "alfworld", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"), "source_sha256": files, "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest() if plan.exists() else None, "python": sys.version, "platform": platform.platform(), "packages": versions}


def execute_run(run, manifest):
    """No shell, no detached jobs, no automatic retries/resume or concurrency."""
    root = Path(run["paths"]["root"])
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("RAY_ADDRESS", None)
    env.update(run["environment"])
    run_manifest = {**manifest, "runs": [run]}
    (root / "manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n")
    status = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    status_path = root / "status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        with Path(run["paths"]["log"]).open("x") as log:
            for argv in (run["prepare_argv"], run["train_argv"]):
                subprocess.run(argv, env=env, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        status["status"] = "completed"
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default="core", help="core (27), expanded (36), all (51), or comma-separated A/B arm IDs; counts at 3 seeds")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--model-4b", dest="model_4b", type=Path)
    p.add_argument("--model-1.7b", dest="model_1p7b", type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--gpus", required=True, help="explicit CUDA indices for EACH rendered run, e.g. 0,1; does not schedule parallel jobs")
    p.add_argument("--num-cpus", default=16, type=int)
    p.add_argument("--threads", default=2, type=int)
    p.add_argument("--micro-batch", default=4, type=int)
    p.add_argument("--logprob-micro-batch", default=8, type=int)
    p.add_argument("--gpu-memory-utilization", default=0.5, type=float)
    p.add_argument("--smoke-steps", type=int, help="1-5 updates with validation/save every update; separate smoke directory, never a formal result")
    p.add_argument("--manifest", type=Path, help="write full JSON manifest to a NEW file; otherwise print JSON to stdout")
    p.add_argument("--execute", action="store_true", help="prepare data and execute exactly one selected arm/seed; default only renders")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        names, seeds = select_arms(args.arms), parse_seeds(args.seeds)
        if args.execute and len(names) * len(seeds) != 1:
            raise ValueError("--execute requires exactly one arm and one seed")
        runs = [build_run(name, seed, args) for name in names for seed in seeds]
    except ValueError as exc:
        p.error(str(exc))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": not args.execute,
        "repository": str(REPO_ROOT),
        "run_count": len(runs),
        "evidence": evidence(),
        "limitations": [
            "H200 memory/throughput and simulator resource use require a real smoke run.",
            "Online val32 is not the formal endpoint eval140.",
            "W/X, fixed-game endpoint evaluation and gradient-vector measurement are not implemented by this launcher.",
            "Seeds do not guarantee bitwise determinism across hardware or library versions; model weight hashes and simulator task IDs must be captured on the training machine.",
        ],
        "runs": runs,
    }
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x") as output:
            output.write(encoded)
        print(f"{'EXECUTE' if args.execute else 'DRY RUN'}: {len(runs)} run(s); manifest: {args.manifest}")
        for run in runs:
            print(run["prepare_command"])
            print(run["train_command"])
    else:
        print(encoded, end="")
    if args.execute:
        execute_run(runs[0], manifest)
    return manifest


if __name__ == "__main__":
    main()
