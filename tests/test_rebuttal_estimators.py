"""Numerical checks for the estimator interventions, without training a model."""

import math

import pytest
import torch

from verl.trainer.ppo.rebuttal import compute_rebuttal_advantages


def estimate(scores, *, task=None, prediction=None, groups=None, trajectories=None, **kwargs):
    scores = torch.as_tensor(scores, dtype=torch.float64)
    count = scores.numel()
    return compute_rebuttal_advantages(
        scores,
        torch.zeros_like(scores) if task is None else torch.as_tensor(task, dtype=scores.dtype),
        scores.clone() if prediction is None else torch.as_tensor(prediction, dtype=scores.dtype),
        ["g"] * count if groups is None else groups,
        list(range(count)) if trajectories is None else trajectories,
        **kwargs,
    )


def test_saturated_boundary_matches_sample_std_and_z_score():
    size, horizon, coefficient = 4, 50, 0.1
    scores = torch.zeros(size * horizon, dtype=torch.float64)
    scores[::horizon] = coefficient
    trajectories = [i for i in range(size) for _ in range(horizon)]
    advantages, diagnostic = estimate(scores, trajectories=trajectories)
    expected_std = coefficient * math.sqrt(size * (horizon - 1) / (horizon * (size * horizon - 1)))
    expected_first = (coefficient - coefficient / horizon) / (expected_std + 1e-6)
    torch.testing.assert_close(diagnostic["normalization_std"], torch.full_like(scores, expected_std))
    assert advantages[0].item() == pytest.approx(expected_first)
    assert expected_std == pytest.approx(0.01403513180)
    assert expected_first == pytest.approx(6.98198, rel=1e-5)
    assert advantages[1] < 0


def test_unequal_length_saturated_trajectories_cancel_before_normalization():
    lengths, coefficient = [1, 2, 5, 8], 0.1
    scores, trajectories = [], []
    for trajectory, length in enumerate(lengths):
        scores.extend([coefficient] + [0.0] * (length - 1))
        trajectories.extend([trajectory] * length)
    step, diagnostic = estimate(scores, trajectories=trajectories)
    count, size = sum(lengths), len(lengths)
    expected = coefficient * math.sqrt(size * (count - size) / (count * (count - 1)))
    assert diagnostic["normalization_std"][0].item() == pytest.approx(expected)
    trajectory, diagnostic = estimate(scores, trajectories=trajectories, estimator="trajectory")
    torch.testing.assert_close(trajectory, torch.zeros_like(trajectory))
    torch.testing.assert_close(diagnostic["trajectory_score"], torch.full_like(trajectory, coefficient))
    assert torch.any(step != 0)


def test_zero_trajectory_sum_can_have_nonzero_step_advantages():
    scores = [0.1, -0.1, 0.2, -0.2]
    trajectories = ["a", "a", "b", "b"]
    step, _ = estimate(scores, trajectories=trajectories)
    trajectory, _ = estimate(scores, trajectories=trajectories, estimator="trajectory")
    assert (step.abs() > 0).all()
    torch.testing.assert_close(trajectory, torch.zeros_like(trajectory))


def test_trajectory_counts_task_once_and_sums_penalties():
    # a: q=10, prediction sum=.1, penalties=.3 -> 9.8.
    # b: q=0, prediction sum=.1, penalties=.4 -> -.3.
    task = [10.0, 10.0, 0.0, 0.0, 0.0]
    prediction = [0.3, -0.2, 0.2, -0.1, 0.0]
    penalty = [0.1, 0.2, 0.1, 0.1, 0.2]
    scores = [q + p - invalid for q, p, invalid in zip(task, prediction, penalty, strict=True)]
    advantages, diagnostic = estimate(scores, task=task, prediction=prediction, trajectories=["a", "a", "b", "b", "b"], estimator="trajectory")
    expected_scores = torch.tensor([9.8, -0.3], dtype=torch.float64)
    z = (expected_scores - expected_scores.mean()) / (expected_scores.std() + 1e-6)
    torch.testing.assert_close(diagnostic["trajectory_score"], torch.tensor([9.8, 9.8, -0.3, -0.3, -0.3], dtype=torch.float64))
    torch.testing.assert_close(advantages, z[torch.tensor([0, 0, 1, 1, 1])])


@pytest.mark.parametrize("estimator", ["step", "trajectory", "gdpo"])
def test_row_order_does_not_change_result(estimator):
    scores = torch.tensor([10.1, 9.8, 0.1, -0.2, 0.4, 0.0], dtype=torch.float64)
    task = torch.tensor([10.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=scores.dtype)
    prediction = torch.tensor([0.1, -0.1, 0.1, 0.0, 0.4, -0.2], dtype=scores.dtype)
    groups, trajectories = ["g", "g", "g", "g", "h", "h"], ["a", "a", "b", "b", "c", "c"]
    base, diagnostic = estimate(scores, task=task, prediction=prediction, groups=groups, trajectories=trajectories, estimator=estimator)
    perm = torch.tensor([4, 2, 0, 5, 3, 1])
    changed, new_diagnostic = estimate(
        scores[perm], task=task[perm], prediction=prediction[perm],
        groups=[groups[i] for i in perm], trajectories=[trajectories[i] for i in perm], estimator=estimator,
    )
    torch.testing.assert_close(changed, base[perm])
    for key in diagnostic:
        torch.testing.assert_close(new_diagnostic[key], diagnostic[key][perm])


def test_step_mean_only_preserves_raw_scale():
    advantages, diagnostic = estimate([1, 2, 10, 14], groups=[0, 0, 1, 1], norm_by_std=False)
    torch.testing.assert_close(advantages, torch.tensor([-0.5, 0.5, -2.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(diagnostic["rms_scale"], torch.ones_like(advantages))


def test_step_singleton_preserves_legacy_convention():
    advantages, diagnostic = estimate([3.0])
    assert advantages.item() == pytest.approx(3 / (1 + 1e-6))
    assert diagnostic["normalization_mean"].item() == 0
    assert diagnostic["normalization_std"].item() == 1
    mean_only, _ = estimate([3.0], norm_by_std=False)
    assert mean_only.item() == 3


@pytest.mark.parametrize("norm_by_std", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_step_matches_existing_estimator_including_singletons(norm_by_std, dtype):
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    scores = torch.tensor([10.2, 9.8, 0.2, 0.2, -0.1], dtype=dtype)
    groups, trajectories = ["a", "a", "b", "b", "c"], list(range(5))
    token_rewards = torch.zeros(5, 3, dtype=dtype)
    token_rewards[:, -1] = scores
    mask = torch.ones_like(token_rewards)
    legacy, _ = compute_grpo_outcome_advantage(token_rewards, mask, groups, trajectories, norm_adv_by_std_in_grpo=norm_by_std)
    current, _ = compute_rebuttal_advantages(
        scores, torch.zeros_like(scores), scores, groups, trajectories, norm_by_std=norm_by_std
    )
    torch.testing.assert_close(current[:, None].expand_as(legacy), legacy)


@pytest.mark.parametrize("estimator", ["trajectory", "gdpo"])
def test_new_estimator_singleton_is_centered(estimator):
    advantages, _ = estimate([3.0], estimator=estimator)
    assert advantages.item() == 0


@pytest.mark.parametrize("estimator", ["step", "trajectory", "gdpo"])
def test_constant_groups_produce_zero(estimator):
    advantages, diagnostic = estimate([0.0, 0.0, 0.0, 0.0], estimator=estimator)
    torch.testing.assert_close(advantages, torch.zeros_like(advantages))
    assert all(torch.isfinite(value).all() for value in diagnostic.values())


def test_gdpo_has_both_channel_and_final_batch_normalization():
    coefficient = 0.2
    task = torch.tensor([10.0, 0.0, 0.0, 10.0], dtype=torch.float64)
    penalties = torch.tensor([0.0, 0.1, 0.2, 0.0], dtype=torch.float64)
    delta = torch.tensor([1.0, -0.2, 0.2, 0.0], dtype=torch.float64)
    scores = task - penalties + coefficient * delta
    task_z = torch.cat([((v := (task - penalties)[start:start + 2]) - v.mean()) / (v.std() + 1e-6) for start in [0, 2]])
    pred_z = torch.cat([((v := delta[start:start + 2]) - v.mean()) / (v.std() + 1e-6) for start in [0, 2]])
    combined = task_z + coefficient * pred_z
    expected = (combined - combined.mean()) / (combined.std() + 1e-6)
    advantages, diagnostic = estimate(scores, task=task, prediction=coefficient * delta, groups=[0, 0, 1, 1], estimator="gdpo", prediction_weight=coefficient)
    torch.testing.assert_close(diagnostic["gdpo_task_advantage"], task_z)
    torch.testing.assert_close(diagnostic["gdpo_prediction_advantage"], pred_z)
    torch.testing.assert_close(diagnostic["gdpo_combined"], combined)
    torch.testing.assert_close(advantages, expected)
    assert not torch.allclose(advantages, combined)


def test_gdpo_zero_prediction_weight_with_no_prediction_component():
    advantages, diagnostic = estimate([1, 3, 0, 0], prediction=[0, 0, 0, 0], estimator="gdpo", prediction_weight=0)
    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(diagnostic["gdpo_prediction_advantage"], torch.zeros_like(advantages))
    with pytest.raises(ValueError, match="zero prediction_weight"):
        estimate([1, 3], estimator="gdpo", prediction_weight=0)


def test_rms_match_uses_one_weighted_batch_factor_without_changing_relative_group_weights():
    scores = [0, 2, 0, 20]
    groups = [0, 0, 1, 1]
    weights = torch.tensor([1, 2, 5, 8], dtype=torch.float64)
    advantages, diagnostic = estimate(scores, groups=groups, match_advantage_rms=True, response_weights=weights)
    mean_only, _ = estimate(scores, groups=groups, norm_by_std=False)
    actual = (weights * advantages.square()).sum() / weights.sum()
    expected = (weights * mean_only.square()).sum() / weights.sum()
    torch.testing.assert_close(actual, expected)
    assert diagnostic["rms_scale"].unique().numel() == 1
    torch.testing.assert_close(advantages, diagnostic["raw_advantages"] * diagnostic["rms_scale"])
    assert not torch.allclose(advantages, mean_only)


def test_rms_zero_group_is_safe():
    advantages, diagnostic = estimate([0, 0], match_advantage_rms=True)
    assert (advantages == 0).all()
    assert (diagnostic["rms_scale"] == 0).all()


def test_rms_is_invariant_to_a_common_scale_of_response_weights():
    scores, groups = [0, 2, 0, 20], [0, 0, 1, 1]
    normal, _ = estimate(scores, groups=groups, match_advantage_rms=True, response_weights=[1, 2, 3, 4])
    huge, _ = estimate(scores, groups=groups, match_advantage_rms=True, response_weights=[1e307, 2e307, 3e307, 4e307])
    torch.testing.assert_close(normal, huge)


def test_fixed_epsilon_restores_coefficient_dependence_below_floor():
    large, _ = estimate([0.1, 0.0])
    small, _ = estimate([0.01, 0.0])
    tiny, _ = estimate([1e-12, 0.0])
    assert large.abs().max().item() == pytest.approx(small.abs().max().item(), rel=2e-4)
    assert tiny.abs().max() < 1e-5


def test_inputs_are_not_mutated_and_outputs_do_not_require_grad():
    scores = torch.tensor([0.1, 0.0], dtype=torch.float64, requires_grad=True)
    original = scores.detach().clone()
    advantages, diagnostic = compute_rebuttal_advantages(scores, torch.zeros_like(scores), scores, [0, 0], [0, 1])
    torch.testing.assert_close(scores, original)
    assert not advantages.requires_grad
    assert not any(value.requires_grad for value in diagnostic.values())


def test_rejects_inconsistent_trajectory_metadata():
    with pytest.raises(ValueError, match="one task group"):
        estimate([0, 0], groups=[0, 1], trajectories=["same", "same"])
    with pytest.raises(ValueError, match="constant within"):
        estimate([0, 10], task=[0, 10], trajectories=["same", "same"])


@pytest.mark.parametrize("kwargs", [
    {"estimator": "unknown"},
    {"estimator": "gdpo", "norm_by_std": False},
    {"estimator": "trajectory", "match_advantage_rms": True},
    {"norm_by_std": False, "match_advantage_rms": True},
    {"epsilon": 0},
    {"epsilon": float("nan")},
    {"prediction_weight": -0.1},
    {"response_weights": [0, 0]},
    {"response_weights": [-1, 2]},
    {"response_weights": [1, float("inf")]},
])
def test_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        estimate([0, 1], **kwargs)


def test_rejects_invalid_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match="active step"):
        estimate([])
    with pytest.raises(ValueError, match="finite"):
        estimate([0, float("nan")])
    with pytest.raises(ValueError, match="shape"):
        estimate([0, 1], prediction=[0])
    with pytest.raises(ValueError, match="entries"):
        estimate([0, 1], groups=[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_output_precision_is_safe(dtype):
    scores = torch.tensor([0.1, 0.0], dtype=dtype)
    advantages, _ = compute_rebuttal_advantages(scores, torch.zeros_like(scores), scores, [0, 0], [0, 1])
    assert advantages.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert torch.isfinite(advantages).all()
