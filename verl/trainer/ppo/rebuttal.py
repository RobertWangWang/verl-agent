# Copyright 2026 The verl-agent contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Scalar advantage estimators for explicitly enabled rebuttal experiments.

Inputs contain one row per *active* environment step, before response-token
broadcasting. No padding rows may enter this module. It deliberately has no
DataProto, Ray, or simulator dependencies so the estimator algebra can be tested
independently of training. All returned diagnostics have one value per input row.

The GDPO variant is a documented step-sample adaptation: sample-standardize each
of two channels within a task group, combine their advantages, then sample-
standardize over the complete step batch. This is not the older capped
``apply_decoupled_pred_advantage`` implementation.
"""

import math
from collections import defaultdict

import torch


def _ids(values, size, name):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = values.detach().cpu().tolist()
    else:
        values = list(values)
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} entries")
    for value in values:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} entries must be hashable scalars") from exc
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"{name} contains an invalid identifier")
    return values


def _vector(values, reference, name):
    values = torch.as_tensor(values, dtype=reference.dtype, device=reference.device).detach()
    if values.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values


def _groups(ids, device):
    grouped = defaultdict(list)
    for row, identifier in enumerate(ids):
        grouped[identifier].append(row)
    return [torch.tensor(rows, dtype=torch.long, device=device) for rows in grouped.values()]


def _normalize(values, groups, epsilon, norm_by_std=True, legacy_singleton=False):
    means, stds = torch.empty_like(values), torch.empty_like(values)
    for indices in groups:
        group = values[indices]
        if group.numel() == 1:
            # Step pooling retains the historical core_algos convention. New
            # trajectory/GDPO normalizers center singleton groups to zero.
            mean = group.new_zeros(()) if legacy_singleton else group[0]
            std = group.new_ones(()) if legacy_singleton else group.new_zeros(())
        else:
            mean, std = group.mean(), group.std(unbiased=True)
        means[indices], stds[indices] = mean, std
    centered = values - means
    return (centered / (stds + epsilon) if norm_by_std else centered), means, stds


@torch.no_grad()
def compute_rebuttal_advantages(
    scores,
    task_scores,
    prediction_scores,
    group_ids,
    trajectory_ids,
    *,
    estimator="step",
    norm_by_std=True,
    epsilon=1e-6,
    prediction_weight=0.1,
    match_advantage_rms=False,
    response_weights=None,
):
    """Return detached scalar advantages and the normalizers actually used.

    Args:
        scores: Floating tensor ``[steps]`` containing task + prediction - penalty.
        task_scores: The pure terminal task score, broadcast to every trajectory row.
        prediction_scores: The prediction component *already injected* in scores,
            including its coefficient and any success/first-step mask. Use zeros
            for prompt-only controls.
        group_ids, trajectory_ids: Stable identifiers for every active step.
            A trajectory must belong to exactly one task group and have one
            constant broadcast task score.
        estimator: ``step``, ``trajectory``, or ``gdpo``.
        norm_by_std: Whether to divide by the sample standard deviation; false
            selects mean-only. GDPO requires true.
        epsilon: Strictly positive additive numerical floor.
        prediction_weight: For GDPO, undo this coefficient in prediction_scores,
            normalize the raw masked prediction delta, and apply the coefficient
            after channel normalization. Zero is allowed only with a zero
            prediction component.
        match_advantage_rms: For standard step pooling only, multiply the whole
            batch by one scalar matching its weighted RMS to mean-only. This
            does not match post-clipping gradients or optimizer updates.
        response_weights: Nonnegative finite ``[steps]`` weights representing
            the policy loss reduction (e.g. valid response-token counts for
            token-mean). They affect RMS matching only, not normalization pools.

    Half/bfloat16 inputs are accumulated in float32; float64 inputs remain
    float64. Inputs are never modified. Trajectory normalization forms
    ``X_i = q_i + sum_t(scores_it - q_i)`` and broadcasts one weight per trajectory.
    """
    if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError("scores must be a one-dimensional floating tensor")
    if scores.numel() == 0:
        raise ValueError("at least one active step is required")
    if estimator not in {"step", "trajectory", "gdpo"}:
        raise ValueError(f"unknown rebuttal estimator: {estimator}")
    if not isinstance(norm_by_std, bool) or not isinstance(match_advantage_rms, bool):
        raise ValueError("normalization flags must be booleans")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    if not math.isfinite(prediction_weight) or prediction_weight < 0:
        raise ValueError("prediction_weight must be finite and nonnegative")
    if estimator == "gdpo" and not norm_by_std:
        raise ValueError("gdpo requires standard-deviation normalization")
    if match_advantage_rms and (estimator != "step" or not norm_by_std):
        raise ValueError("RMS matching requires the step estimator with standard-deviation normalization")

    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    scores = scores.detach().to(dtype=dtype)
    if not torch.isfinite(scores).all():
        raise ValueError("scores must contain finite values")
    task_scores = _vector(task_scores, scores, "task_scores")
    prediction_scores = _vector(prediction_scores, scores, "prediction_scores")
    groups = _ids(group_ids, scores.numel(), "group_ids")
    trajectories = _ids(trajectory_ids, scores.numel(), "trajectory_ids")
    group_rows = _groups(groups, scores.device)
    trajectory_rows = _groups(trajectories, scores.device)
    for indices in trajectory_rows:
        rows = indices.cpu().tolist()
        if any(groups[row] != groups[rows[0]] for row in rows[1:]):
            raise ValueError("a trajectory must belong to exactly one task group")
        if not torch.equal(task_scores[indices], task_scores[indices[0]].expand(indices.numel())):
            raise ValueError("task_scores must be constant within each trajectory")

    weights = torch.ones_like(scores) if response_weights is None else _vector(response_weights, scores, "response_weights")
    if (weights < 0).any() or not (weights > 0).any():
        raise ValueError("response_weights must be nonnegative with a positive total")

    diagnostics = {}
    if estimator == "trajectory":
        trajectory_scores = torch.stack([task_scores[rows[0]] + (scores[rows] - task_scores[rows[0]]).sum() for rows in trajectory_rows])
        trajectory_groups = [groups[int(rows[0])] for rows in trajectory_rows]
        trajectory_advantages, trajectory_means, trajectory_stds = _normalize(
            trajectory_scores, _groups(trajectory_groups, scores.device), epsilon, norm_by_std
        )
        advantages, normalization_score = torch.empty_like(scores), torch.empty_like(scores)
        means, stds = torch.empty_like(scores), torch.empty_like(scores)
        for index, rows in enumerate(trajectory_rows):
            advantages[rows] = trajectory_advantages[index]
            normalization_score[rows] = trajectory_scores[index]
            means[rows], stds[rows] = trajectory_means[index], trajectory_stds[index]
        diagnostics["trajectory_score"] = normalization_score
    elif estimator == "gdpo":
        if prediction_weight == 0:
            if (prediction_scores != 0).any():
                raise ValueError("zero prediction_weight requires zero prediction_scores")
            raw_prediction = torch.zeros_like(scores)
        else:
            raw_prediction = prediction_scores / prediction_weight
        task_advantage, task_mean, task_std = _normalize(scores - prediction_scores, group_rows, epsilon)
        prediction_advantage, prediction_mean, prediction_std = _normalize(raw_prediction, group_rows, epsilon)
        normalization_score = task_advantage + prediction_weight * prediction_advantage
        advantages, means, stds = _normalize(
            normalization_score, [torch.arange(scores.numel(), device=scores.device)], epsilon
        )
        diagnostics.update(
            gdpo_task_advantage=task_advantage,
            gdpo_prediction_advantage=prediction_advantage,
            gdpo_combined=normalization_score,
            gdpo_task_mean=task_mean,
            gdpo_task_std=task_std,
            gdpo_prediction_mean=prediction_mean,
            gdpo_prediction_std=prediction_std,
        )
    else:
        normalization_score = scores
        advantages, means, stds = _normalize(scores, group_rows, epsilon, norm_by_std, legacy_singleton=True)

    diagnostics.update(
        normalization_score=normalization_score,
        normalization_mean=means,
        normalization_std=stds,
        raw_advantages=advantages.clone(),
    )
    scale = scores.new_ones(())
    if match_advantage_rms:
        mean_only = scores - means
        # Relative weights define the same RMS; rescale before summing so large
        # but finite weights cannot overflow their normalizing denominator.
        relative_weights = weights / weights.max()
        normalized_weights = relative_weights / relative_weights.sum()
        raw_rms = (normalized_weights * advantages.square()).sum().sqrt()
        target_rms = (normalized_weights * mean_only.square()).sum().sqrt()
        scale = target_rms / raw_rms if raw_rms > 0 else scores.new_zeros(())
        advantages = advantages * scale
        diagnostics["rms_target"] = target_rms.expand_as(scores)
        diagnostics["rms_before"] = raw_rms.expand_as(scores)

    diagnostics["rms_scale"] = scale.expand_as(scores)
    diagnostics["advantages"] = advantages
    for name, values in diagnostics.items():
        if not torch.isfinite(values).all():
            raise ValueError(f"non-finite output in {name}; check score magnitudes and normalization settings")
    return advantages, diagnostics
