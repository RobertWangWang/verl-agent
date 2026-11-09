"""Exercise shell entry points and serial execution without starting training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.rebuttal import launch, run_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    model = tmp_path / "model with spaces"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    output = tmp_path / "runs with spaces"
    manifest = tmp_path / "plan with spaces.json"
    argv = [
        "--model-4b", str(model), "--model-1.7b", str(model),
        "--output-root", str(output), "--gpus", "0,1",
        "--manifest", str(manifest),
    ]
    return output, manifest, argv


def shell(name, argv, cwd):
    return subprocess.run(
        ["bash", str(ROOT / "examples/rebuttal" / name), *argv],
        cwd=cwd, env={**os.environ, "PYTHON_BIN": sys.executable},
        text=True, capture_output=True, timeout=60,
    )


def test_smoke_shell_from_another_directory_defaults_to_two_updates(settings, tmp_path):
    output, manifest, argv = settings
    result = shell("run_smoke.sh", argv, tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(manifest.read_text())
    assert plan["dry_run"]
    assert len(plan["runs"]) == 1
    run = plan["runs"][0]
    assert run["run_id"] == "A01_s0_smoke2"
    config = run["resolved_config"]
    assert config["trainer"]["total_training_steps"] == 2
    assert config["trainer"]["save_freq"] == config["trainer"]["test_freq"] == 1
    assert config["data"]["train_batch_size"] == 8
    assert config["data"]["val_batch_size"] == 32
    assert config["env"]["rollout"]["n"] == 4
    assert config["actor_rollout_ref"]["rollout"]["enforce_eager"] is False
    assert config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 8
    assert not (output / run["run_id"]).exists()


def test_train_shell_keeps_each_arm_seed_distinct_without_execution(settings, tmp_path):
    output, manifest, argv = settings
    result = shell("run_train.sh", argv + ["--arms", "A00,B02", "--seeds", "0,2", "--dry-run", "--enforce-eager"], tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(manifest.read_text())
    assert plan["dry_run"]
    assert [run["run_id"] for run in plan["runs"]] == ["A00_s0", "A00_s2", "B02_s0", "B02_s2"]
    for run in plan["runs"]:
        config = run["resolved_config"]
        assert config["trainer"]["total_training_steps"] == (150 if run["arm"].startswith("A") else 60)
        assert config["trainer"]["resume_mode"] == "disable"
        assert config["actor_rollout_ref"]["rollout"]["enforce_eager"] is True
        assert not (output / run["run_id"]).exists()


def test_all_serial_wrapper_uses_paper_seeds_and_eight_gpu_defaults(settings, tmp_path):
    output, manifest, argv = settings
    argv = argv.copy()
    gpu_arg = argv.index("--gpus")
    del argv[gpu_arg:gpu_arg + 2]
    result = shell("run_all_serial.sh", argv, tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(manifest.read_text())
    assert plan["dry_run"]
    arms = [f"A{i:02d}" for i in range(11)] + [f"B{i:02d}" for i in range(6)]
    assert [run["run_id"] for run in plan["runs"]] == [f"{arm}_s{seed}" for arm in arms for seed in (0, 42, 96)]
    for run in plan["runs"]:
        config = run["resolved_config"]
        assert run["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
        assert config["trainer"]["n_gpus_per_node"] == 8
        assert config["actor_rollout_ref"]["model"]["path"] == str(tmp_path / "model with spaces")
        assert config["actor_rollout_ref"]["rollout"]["enforce_eager"] is False
        assert config["actor_rollout_ref"]["rollout"]["free_cache_engine"] is True
        assert config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.6
        assert config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 8
        assert config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 16
        assert config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] == 16
        assert config["trainer"]["total_training_steps"] == (150 if run["arm"].startswith("A") else 60)
        assert run["paths"]["root"] == str(output / run["run_id"])
        assert not (output / run["run_id"]).exists()


def test_relative_paths_are_relative_to_callers_directory(settings, tmp_path):
    result = shell("run_smoke.sh", [
        "--model-4b", "model with spaces", "--output-root", "relative runs",
        "--gpus", "0,1", "--manifest", "relative plan.json",
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads((tmp_path / "relative plan.json").read_text())
    run = plan["runs"][0]
    assert run["resolved_config"]["actor_rollout_ref"]["model"]["path"] == str(tmp_path / "model with spaces")
    assert run["paths"]["root"] == str(tmp_path / "relative runs" / "A01_s0_smoke2")


def test_execution_stops_after_first_failure(settings, monkeypatch):
    _, manifest, argv = settings
    calls = []

    def execute(run, plan):
        assert manifest.is_file()  # Save the whole reviewed plan before any training starts.
        assert len(plan["runs"]) == 3
        calls.append(run["run_id"])
        if len(calls) == 2:
            raise RuntimeError("simulated training failure")

    monkeypatch.setattr(launch, "execute_run", execute)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        run_matrix.main(["train", *argv, "--arms", "A00,A01,A02", "--seeds", "0", "--execute"])
    assert calls == ["A00_s0", "A01_s0"]


def test_late_run_directory_conflict_prevents_all_execution(settings, monkeypatch):
    output, manifest, argv = settings
    (output / "A01_s0").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(launch, "execute_run", lambda *args: calls.append(args))
    with pytest.raises((SystemExit, ValueError, FileExistsError)):
        run_matrix.main(["train", *argv, "--arms", "A00,A01", "--seeds", "0", "--execute"])
    assert not calls
    assert not (output / "A00_s0").exists()
    assert not manifest.exists()


def test_manifest_cannot_create_a_future_run_directory(settings, monkeypatch):
    output, _, argv = settings
    argv = argv[:-2] + ["--manifest", str(output / "A00_s0" / "plan.json")]
    calls = []
    monkeypatch.setattr(launch, "execute_run", lambda *args: calls.append(args))
    with pytest.raises((SystemExit, ValueError)):
        run_matrix.main(["train", *argv, "--arms", "A00", "--seeds", "0", "--execute"])
    assert not calls
    assert not (output / "A00_s0").exists()


@pytest.mark.parametrize("extra", [["--smoke-steps", "2"], ["--execute", "--dry-run"]])
def test_train_rejects_conflicting_modes_before_execution(settings, monkeypatch, extra):
    _, manifest, argv = settings
    monkeypatch.setattr(launch, "execute_run", lambda *args: pytest.fail("must reject before training"))
    with pytest.raises(SystemExit):
        run_matrix.main(["train", *argv, "--arms", "A00", "--seeds", "0", *extra])
    assert not manifest.exists()


def test_shell_sources_are_in_reproducibility_manifest():
    hashes = launch.evidence()["source_sha256"]
    for name in ("run_smoke.sh", "run_train.sh", "run_all_serial.sh"):
        assert "examples/rebuttal/" + name in hashes
