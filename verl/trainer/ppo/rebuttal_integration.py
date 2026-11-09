"""Opt-in integration of controlled estimator experiments with DataProto batches."""

import math

import numpy as np
import torch

from verl.trainer.ppo.rebuttal import compute_rebuttal_advantages


def validate_rebuttal_config(config):
    """Reject combinations whose meaning is not covered by the experiment protocol."""
    cfg = config.algorithm.get("rebuttal", {}) or {}
    pred = config.algorithm.get("pred_reward", {}) or {}
    enabled = cfg.get("enable", False)
    if pred.get("success_only", False) and pred.get("decoupled_advantage", False):
        raise ValueError("success_only cannot be combined with legacy decoupled_advantage")
    if not enabled:
        return
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("rebuttal estimators require algorithm.adv_estimator=grpo")
    if config.algorithm.get("use_kl_in_reward", False):
        raise ValueError("rebuttal component accounting requires use_kl_in_reward=False")
    if pred.get("decoupled_advantage", False):
        raise ValueError("select rebuttal.estimator=gdpo instead of legacy decoupled_advantage")
    if config.algorithm.get("filter_all_fail_groups", False):
        raise ValueError("rebuttal estimators do not support post-estimator all-fail filtering")
    if config.algorithm.get("aux_sft", {}).get("enable", False):
        raise ValueError("rebuttal reward experiments require aux_sft.enable=False")
    estimator = cfg.get("estimator", "step")
    if estimator not in {"step", "trajectory", "gdpo"}:
        raise ValueError(f"Unknown rebuttal estimator: {estimator}")
    standard = config.algorithm.get("norm_adv_by_std_in_grpo", True)
    if estimator == "gdpo" and not standard:
        raise ValueError("GDPO requires standard-deviation normalization")
    if cfg.get("match_advantage_rms", False) and (estimator != "step" or not standard):
        raise ValueError("RMS matching requires the step estimator with standard normalization")
    epsilon = float(cfg.get("epsilon", 1e-6))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("rebuttal.epsilon must be finite and positive")


def set_optimization_mask(data):
    """Exclude copied divisibility padding from policy, KL and estimator masks.

    Keep attention intact so model forwards remain valid. The actor consumes
    this separate mask only for the explicitly enabled rebuttal protocol.
    """
    mask = data.batch["response_mask"].clone()
    padding = data.non_tensor_batch.get("optimization_padding")
    if padding is not None:
        excluded = torch.as_tensor(np.asarray(padding, dtype=bool), device=mask.device)
        mask[excluded] = 0
    if "active_masks" in data.non_tensor_batch:
        active = torch.as_tensor(np.asarray(data.non_tensor_batch["active_masks"], dtype=bool), device=mask.device).reshape(-1)
        mask[~active] = 0
    if "loss_mask" in data.batch:
        mask = mask * data.batch["loss_mask"][:, -mask.shape[-1]:].to(mask.device)
    data.batch["response_mask"] = mask
    data.batch["optimization_mask"] = mask


def successful_trajectory_mask(task_scores, trajectory_ids):
    """Use pure task outcomes, never scores after shaping or action penalties."""
    scores = torch.as_tensor(task_scores).detach()
    if scores.ndim != 1 or len(scores) != len(trajectory_ids):
        raise ValueError("task_scores and trajectory_ids must be aligned one-dimensional arrays")
    if not torch.isfinite(scores).all():
        raise ValueError("task_scores must be finite")
    totals = {}
    for trajectory, score in zip(trajectory_ids, scores.tolist(), strict=True):
        totals[trajectory] = totals.get(trajectory, 0.0) + score
    return torch.tensor([totals[t] > 0 for t in trajectory_ids], device=scores.device, dtype=torch.bool)


def compute_rebuttal_batch(data, cfg, task_scores, prediction_scores, *, norm_by_std=True,
                           prediction_weight=0.1, multi_turn=False, loss_agg_mode="token-mean"):
    """Write only advantages/returns; return actual estimator diagnostics separately."""
    mask = data.batch["response_mask"]
    if "optimization_mask" in data.batch:
        mask = data.batch["optimization_mask"]
    elif multi_turn:
        mask = data.batch["loss_mask"][:, -mask.shape[-1]:]
    lengths = mask.sum(-1).to(torch.float32)
    active = lengths > 0
    if "active_masks" in data.non_tensor_batch:
        env_active = torch.as_tensor(np.asarray(data.non_tensor_batch["active_masks"], dtype=bool), device=mask.device)
        active &= env_active.reshape(-1)
    # Padding/empty responses must not enter group statistics or the policy loss.
    scores = data.batch["token_level_rewards"].sum(-1)
    tasks = torch.as_tensor(task_scores, device=scores.device, dtype=scores.dtype)
    predictions = torch.as_tensor(prediction_scores, device=scores.device, dtype=scores.dtype)
    active = active.to(scores.device)
    if tasks.shape != scores.shape or predictions.shape != scores.shape:
        raise ValueError("rebuttal score snapshots must align with response rows")
    # These are nominal full-batch loss weights. The legacy actor averages
    # microbatch losses separately, so this is not an exact gradient-weighted
    # RMS when response lengths/microbatch token totals differ.
    if loss_agg_mode in {"token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"}:
        weights = lengths.to(scores.device)[active]
    elif loss_agg_mode == "seq-mean-token-mean":
        weights = torch.ones_like(scores[active])
    else:
        raise ValueError(f"Unsupported loss reduction for RMS accounting: {loss_agg_mode}")
    scalar = torch.zeros_like(scores)
    diagnostics = {"active_row": active}
    if active.any():
        selected = active.detach().cpu().numpy()
        values, selected_diagnostics = compute_rebuttal_advantages(
            scores[active], tasks[active], predictions[active],
            np.asarray(data.non_tensor_batch["uid"])[selected],
            np.asarray(data.non_tensor_batch["traj_uid"])[selected],
            estimator=cfg.get("estimator", "step"), norm_by_std=norm_by_std,
            epsilon=float(cfg.get("epsilon", 1e-6)), prediction_weight=prediction_weight,
            match_advantage_rms=cfg.get("match_advantage_rms", False), response_weights=weights,
        )
        scalar[active] = values
        for name, values in selected_diagnostics.items():
            expanded = torch.full_like(scores, float("nan"))
            expanded[active] = values
            diagnostics[name] = expanded
    data.batch["advantages"] = scalar.unsqueeze(-1) * mask.to(scores.device)
    data.batch["returns"] = data.batch["advantages"].clone()
    return data, diagnostics
