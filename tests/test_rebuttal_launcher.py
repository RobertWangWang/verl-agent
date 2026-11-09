"""No training, model loading or Ray startup is needed for these checks."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.rebuttal import launch
from examples.rebuttal.prepare import prepare
from verl.utils.reproducibility import derived_seed, isolated_generation_rng, make_device_rng_state, rollout_sampling_kwargs, seed_everything


@pytest.fixture
def argv(tmp_path):
    model = tmp_path / "model with space"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    return ["--model-4b", str(model), "--model-1.7b", str(model), "--output-root", str(tmp_path / "runs"), "--gpus", "0,1"]


def test_full_matrix_has_distinct_paths_and_seeds(argv):
    args = launch.parser().parse_args(argv)
    runs = [launch.build_run(name, seed, args) for name in launch.select_arms("all") for seed in launch.parse_seeds("0,1,2")]
    assert len(runs) == 51
    for path in ("root", "ray_temp", "log"):
        assert len({run["paths"][path] for run in runs}) == 51
    for run in runs:
        c, seed = run["resolved_config"], run["seed"]
        assert c["data"]["train_batch_size"] == 8
        assert c["data"]["val_batch_size"] == 32
        assert c["env"]["rollout"]["n"] == 4
        assert c["trainer"]["total_training_steps"] == (150 if run["arm"].startswith("A") else 60)
        assert c["trainer"]["resume_mode"] == "disable"
        assert c["data"]["seed"] == c["env"]["seed"] == c["trainer"]["seed"] == seed
        assert c["actor_rollout_ref"]["seed"] == c["actor_rollout_ref"]["rollout"]["seed"] == seed
        assert c["actor_rollout_ref"]["eval_seed"] == c["env"]["eval_seed"] == seed + 1000
        assert c["actor_rollout_ref"]["rollout"]["temperature"] == 1.0
        assert c["actor_rollout_ref"]["rollout"]["val_kwargs"]["temperature"] == 0.4
        assert c["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 8
        assert c["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 16
        assert c["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] == 16
        assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.6
        assert c["actor_rollout_ref"]["rollout"]["enforce_eager"] is False
        assert c["actor_rollout_ref"]["rollout"]["free_cache_engine"] is True
        assert c["algorithm"]["rebuttal"]["enable"]
        assert c["env"]["alfworld"]["prediction"]["feature_weights"]["location_change"] == 0.5
        assert c["ray_init"]["address"] == "local"
        assert run["train_argv"][:3] == [launch.sys.executable, "-m", "verl.trainer.main_ppo"]
        assert "model with space" in c["actor_rollout_ref"]["model"]["path"]


def test_packages_and_scientific_arm_switches(argv):
    assert len(launch.select_arms("core")) * 3 == 27
    assert len(launch.select_arms("expanded")) * 3 == 36
    args = launch.parser().parse_args(argv)
    configs = {name: launch.build_run(name, 0, args)["resolved_config"] for name in launch.ARMS}
    for name, config in configs.items():
        arm = launch.ARMS[name]
        assert config["env"]["alfworld"]["prediction"]["enable"] == (name not in {"A08", "A09"})
        assert config["algorithm"]["pred_reward"]["enable"] == arm.prediction_reward
        assert config["algorithm"]["norm_adv_by_std_in_grpo"] == arm.norm_std
    assert configs["A02"]["env"]["alfworld"]["prediction"]["drop_first_reward"]
    assert configs["A10"]["env"]["alfworld"]["prediction"]["drop_first_reward"]
    assert configs["A05"]["algorithm"]["pred_reward"]["success_only"]
    assert configs["A06"]["algorithm"]["rebuttal"]["estimator"] == "gdpo"
    assert configs["A07"]["algorithm"]["rebuttal"]["match_advantage_rms"]
    assert configs["B02"]["algorithm"]["rebuttal"]["estimator"] == "trajectory"
    assert configs["B05"]["algorithm"]["rebuttal"]["estimator"] == "trajectory"
    assert configs["B05"]["env"]["alfworld"]["prediction"]["drop_first_reward"]


def test_dry_run_never_prepares_or_executes(argv, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not execute training")

    monkeypatch.setattr(launch, "execute_run", forbidden)
    monkeypatch.setattr(launch, "evidence", lambda: {"test": True})
    result = launch.main(argv + ["--arms", "A01", "--seeds", "2"])
    assert result["dry_run"]
    assert not (tmp_path / "runs").exists()
    assert json.loads(capsys.readouterr().out)["runs"][0]["seed"] == 2


def test_execute_rejects_multiple_runs_before_any_launch(argv, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must validate before creating runs")

    monkeypatch.setattr(launch, "build_run", forbidden)
    with pytest.raises(SystemExit):
        launch.main(argv + ["--execute"])


@pytest.mark.parametrize("arms", ["W00", "X01", "A00,A00", "A99"])
def test_unsupported_or_duplicate_arms_rejected(arms):
    with pytest.raises(ValueError):
        launch.select_arms(arms)


def test_immutable_per_run_parquet(tmp_path):
    import pyarrow.parquet as pq

    target = tmp_path / "data"
    metadata = prepare(target)
    train = pq.read_table(target / "train.parquet").to_pylist()
    assert len(train) == metadata["train"]["rows"] == 8
    assert pq.read_table(target / "test.parquet").num_rows == 32
    assert train[0]["prompt"] == [{"role": "user", "content": ""}]
    assert {row["extra_info"]["index"] for row in train} == set(range(8))
    with pytest.raises(FileExistsError):
        prepare(target)


def test_execute_orders_prepare_then_train_and_refuses_reuse(argv, monkeypatch):
    run = launch.build_run("A00", 0, launch.parser().parse_args(argv))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    launch.execute_run(run, {"test": True, "run_count": 3})
    assert [call[0] for call in calls] == [run["prepare_argv"], run["train_argv"]]
    assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "RAY_ADDRESS" not in calls[1][1]["env"]
    assert json.loads((launch.Path(run["paths"]["root"]) / "status.json").read_text())["status"] == "completed"
    assert json.loads((launch.Path(run["paths"]["root"]) / "manifest.json").read_text())["run_count"] == 1
    with pytest.raises(FileExistsError):
        launch.execute_run(run, {})


def test_smoke_has_separate_identity_and_save_validation(argv):
    run = launch.build_run("A01", 0, launch.parser().parse_args(argv + ["--smoke-steps", "2"]))
    assert run["smoke_only"]
    assert run["run_id"] == "A01_s0_smoke2"
    assert run["resolved_config"]["trainer"]["total_training_steps"] == 2
    assert run["resolved_config"]["trainer"]["save_freq"] == run["resolved_config"]["trainer"]["test_freq"] == 1


def test_runtime_overrides_are_rendered_without_changing_experiment(argv):
    args = launch.parser().parse_args(argv + [
        "--micro-batch", "4", "--logprob-micro-batch", "8",
        "--gpu-memory-utilization", "0.5", "--enforce-eager",
    ])
    run = launch.build_run("A01", 42, args)
    c = run["resolved_config"]
    assert c["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 4
    assert c["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 8
    assert c["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] == 8
    assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.5
    assert c["actor_rollout_ref"]["rollout"]["enforce_eager"] is True
    assert "actor_rollout_ref.rollout.enforce_eager=true" in run["train_argv"]
    assert c["algorithm"]["pred_reward"]["enable"] is True
    assert c["data"]["train_batch_size"] == 8
    assert c["env"]["rollout"]["n"] == 4
    assert c["trainer"]["seed"] == 42


def test_seed_repeats_all_cpu_sources_and_distinguishes_ranks(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def sample(seed, rank=0):
        seed_everything(seed, rank)
        return random.random(), np.random.random(), torch.rand(1).item()

    assert sample(17, 2) == sample(17, 2)
    assert sample(17, 2) != sample(17, 3)
    seed_everything(10)
    expected = random.getstate()
    assert seed_everything(None) is None
    assert random.getstate() == expected
    assert derived_seed(2**32 - 1, 1) == 0
    with pytest.raises(ValueError):
        derived_seed(-1)


def test_generation_streams_restore_even_on_exception():
    manager = SimpleNamespace(gen_random_states="train")
    validation = ["eval"]
    with pytest.raises(RuntimeError):
        with isolated_generation_rng(manager, validation):
            assert manager.gen_random_states == "eval"
            manager.gen_random_states = "eval-next"
            raise RuntimeError("rollout failed")
    assert manager.gen_random_states == "train"
    assert validation == ["eval-next"]


def test_build_device_state_preserves_active_stream():
    class Device:
        state = 10

        def get_rng_state(self):
            return self.state

        def set_rng_state(self, state):
            self.state = state

        def manual_seed(self, seed):
            self.state = seed

    device = Device()
    assert make_device_rng_state(device, 101) == 101
    assert device.state == 10


@pytest.mark.parametrize("seed", [0, 42, None])
def test_engine_seed_is_not_fixed_request_sampling_seed(seed):
    params = SimpleNamespace(seed=None, temperature=1.0, top_p=1.0, n=1)
    config = {"seed": seed, "temperature": 0.4, "top_p": 0.95, "tensor_model_parallel_size": 1}
    kwargs = rollout_sampling_kwargs(config, params)
    assert kwargs == {"temperature": 0.4, "top_p": 0.95}
    request_params = SimpleNamespace(**{**vars(params), **kwargs})
    assert request_params.seed is None
