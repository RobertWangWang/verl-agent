"""Exercise the real DataProto reward -> estimator -> telemetry path on CPU."""

import gzip
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.telemetry.group_variance_logger import dump_group_variance_rows
from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_prediction_reward, compute_advantage
from verl.trainer.ppo.rebuttal_integration import set_optimization_mask, validate_rebuttal_config


def make_batch(tasks, predictions, trajectories, active=None):
    n = len(tasks)
    scores = torch.zeros(n, 3)
    scores[:, -1] = torch.tensor(tasks)
    return DataProto.from_dict(
        tensors={
            "token_level_scores": scores,
            "token_level_rewards": scores.clone(),
            "prompts": torch.zeros(n, 2, dtype=torch.long),
            "responses": torch.ones(n, 3, dtype=torch.long),
            "attention_mask": torch.ones(n, 5, dtype=torch.long),
            "response_mask": torch.ones(n, 3),
        },
        non_tensors={
            "uid": np.array(["g"] * n), "traj_uid": np.array(trajectories),
            "pred_rewards": np.array(predictions, dtype=np.float32),
            "active_masks": np.ones(n, dtype=bool) if active is None else np.array(active, dtype=bool),
        },
    )


def estimate(batch, tasks, *, estimator="step", standard=True, matching=False):
    return compute_advantage(
        batch, "grpo", norm_adv_by_std_in_grpo=standard,
        rebuttal_cfg={"enable": True, "estimator": estimator, "match_advantage_rms": matching},
        rebuttal_task_scores=torch.tensor(tasks),
        rebuttal_prediction_scores=batch.batch["token_level_rewards"].sum(-1) - torch.tensor(tasks),
        rebuttal_prediction_weight=0.1,
    )


def test_success_only_masks_prediction_not_samples_or_penalty():
    batch = make_batch([10., 0., 10.], [1., 1., -.5], ["win", "fail", "win"])
    original = batch.non_tensor_batch["pred_rewards"].copy()
    batch.batch["token_level_scores"][:, -1] -= 20.0  # never infer success from these scores
    before = batch.batch["token_level_scores"].clone()
    batch, _ = apply_prediction_reward(batch, .1, success_only=True, task_scores=torch.tensor([10., 0., 10.]))
    torch.testing.assert_close((batch.batch["token_level_scores"] - before).sum(-1), torch.tensor([.1, 0., -.05]), atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(original, batch.non_tensor_batch["pred_rewards"])
    assert len(batch) == 3


def test_empty_response_and_inactive_rows_never_receive_reward():
    batch = make_batch([0., 0., 0.], [1., 1., 1.], ["a", "b", "c"], [True, True, False])
    batch.batch["attention_mask"][1, 2:] = 0
    batch, _ = apply_prediction_reward(batch, .1)
    torch.testing.assert_close(batch.batch["token_level_scores"].sum(-1), torch.tensor([.1, 0., 0.]))


def test_step_rebuttal_matches_legacy_for_valid_rows():
    a = make_batch([0., 10., 0., 0.], [.8, -.3, .2, 1.], ["a", "b", "c", "d"])
    b = make_batch([0., 10., 0., 0.], [.8, -.3, .2, 1.], ["a", "b", "c", "d"])
    for batch in (a, b):
        apply_prediction_reward(batch, .1)
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"].clone()
    compute_advantage(a, "grpo")
    estimate(b, [0., 10., 0., 0.])
    torch.testing.assert_close(a.batch["advantages"], b.batch["advantages"])


def test_trajectory_counts_task_once_and_telemetry_uses_trajectory_denominator(tmp_path):
    tasks = [10., 10., 0.]
    batch = make_batch(tasks, [1., 0., 1.], ["a", "a", "b"])
    apply_prediction_reward(batch, .1)
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"].clone()
    estimate(batch, tasks, estimator="trajectory")
    diag = batch.meta_info.pop("_rebuttal_diagnostics")
    expected = (torch.tensor(10.1) - 5.1) / (torch.tensor([10.1, .1]).std() + 1e-6)
    torch.testing.assert_close(batch.batch["advantages"][:, 0], torch.stack([expected, expected, -expected]))
    path = dump_group_variance_rows(
        batch, str(tmp_path), 1, torch.tensor(tasks), torch.tensor(tasks), .1, 1e-6,
        {"run_id": "test", "seed": 0, "estimator": "trajectory", "norm_by_std": True}, diagnostics=diag,
    )
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[0]["group_std"] == pytest.approx(torch.tensor([10.1, .1]).std().item())
    assert rows[1]["estimator_trajectory_score"] == pytest.approx(10.1)
    assert rows[1]["x_prenorm"] == pytest.approx(10.0)


def test_inactive_padding_excluded_from_estimator():
    batch = make_batch([0., 0., 999.], [0., 0., 0.], ["a", "b", "padding"], [True, True, False])
    batch.batch["token_level_rewards"][:, -1] = torch.tensor([.1, -.1, 999.])
    estimate(batch, [0., 0., 999.])
    assert torch.count_nonzero(batch.batch["advantages"][2]) == 0
    assert batch.batch["advantages"][0, 0].item() == pytest.approx(2**-.5, abs=1e-5)


def test_missing_success_snapshot_rejected():
    batch = make_batch([0.], [1.], ["a"])
    with pytest.raises(ValueError, match="pure task"):
        apply_prediction_reward(batch, .1, success_only=True)


def test_explicit_padding_mask_preserves_forward_but_excludes_loss_and_telemetry(tmp_path):
    batch = make_batch([0., 0., 0.], [1., -1., 1.], ["a", "b", "a"])
    batch.non_tensor_batch["optimization_padding"] = np.array([False, False, True])
    set_optimization_mask(batch)
    assert batch.batch["attention_mask"][2].all()
    assert not batch.batch["optimization_mask"][2].any()
    apply_prediction_reward(batch, .1)
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"].clone()
    estimate(batch, [0., 0., 0.], estimator="trajectory")
    assert batch.batch["token_level_rewards"][2].sum() == 0
    assert batch.batch["advantages"][2].sum() == 0
    path = dump_group_variance_rows(
        batch, str(tmp_path), 1, torch.zeros(3), torch.zeros(3), .1, 1e-6,
        {"estimator": "trajectory"}, diagnostics=batch.meta_info["_rebuttal_diagnostics"],
    )
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 2
    assert all(row["traj_n_samples"] == 1 for row in rows)


def test_multiturn_uses_intersection_of_loss_and_padding_masks():
    batch = make_batch([0., 0.], [0., 0.], ["a", "b"])
    batch.batch["loss_mask"] = torch.tensor([[1, 1, 0, 1, 0], [1, 1, 1, 1, 1]])
    batch.non_tensor_batch["optimization_padding"] = np.array([False, True])
    set_optimization_mask(batch)
    torch.testing.assert_close(batch.batch["optimization_mask"], torch.tensor([[0., 1., 0.], [0., 0., 0.]]))


@pytest.mark.parametrize("estimator,matching", [("gdpo", False), ("step", True)])
def test_telemetry_matches_actual_final_advantage(tmp_path, estimator, matching):
    batch = make_batch([0., 0., 10., 10.], [1., -.2, .5, -.7], ["a", "a", "b", "b"])
    apply_prediction_reward(batch, .1)
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"].clone()
    tasks = torch.tensor([0., 0., 10., 10.])
    estimate(batch, tasks.tolist(), estimator=estimator, matching=matching)
    path = dump_group_variance_rows(
        batch, str(tmp_path), 1, tasks, tasks, .1, 1e-6,
        {"estimator": estimator}, diagnostics=batch.meta_info["_rebuttal_diagnostics"],
    )
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        expected = (row["estimator_normalization_score"] - row["group_mean"]) / (row["group_std"] + 1e-6)
        expected *= row["estimator_rms_scale"]
        assert row["advantage"] == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize("patch", [
    {"adv_estimator": "gae"}, {"use_kl_in_reward": True}, {"filter_all_fail_groups": True},
    {"pred_reward": {"decoupled_advantage": True}}, {"aux_sft": {"enable": True}},
    {"rebuttal": {"enable": True, "estimator": "trajectory", "match_advantage_rms": True}},
])
def test_incompatible_protocols_rejected(patch):
    cfg = OmegaConf.merge({"algorithm": {"adv_estimator": "grpo", "rebuttal": {"enable": True}}}, {"algorithm": patch})
    with pytest.raises(ValueError):
        validate_rebuttal_config(cfg)
