"""
Group-wise variance decomposition telemetry (2026-09 data request).

Per-sample export of every quantity entering GRPO's within-group
normalization, so task / prediction / penalty variances and covariances can
be decomposed offline. One JSONL(.gz) file per update; raw float values.

Design constraints (from the request):
- one row per valid environment-step optimization sample;
- the exact normalization groups must be recoverable (uid + the replicated
  group statistics below);
- raw floats, never rounded summaries;
- unavailable fields are noted, not imputed.

Component decomposition is taken by snapshotting the per-sample scalar
score x_i = token_level_scores[i].sum() at three points of the fit loop:
    snap_task           after task reward set, before penalty injection
    snap_after_penalty  after apply_invalid_action_penalty
    x_final             token_level_rewards.sum(-1) fed to the estimator
so   penalty_i = snap_after_penalty_i - snap_task_i   (<= 0)
     pred_component_i = x_final_i - snap_after_penalty_i (= lambda * pred_delta_i,
     cross-checkable against the raw pred_rewards field).

Legacy group statistics replicate the step-pooled GRPO calculation. Opt-in
rebuttal estimators provide their actual statistics directly, including the
trajectory pool, GDPO channels/batch normalization and the final RMS scale.

Known limitations (documented for the README, per the request's "note, do
not reconstruct" rule):
- env_step_rank is only the occurrence rank in the current batch, which may
  have been reordered. Use the explicitly collected env_step/is_first_step.
- termination reason is not recorded by the framework -> field absent;
- old rollouts lack first-step and previous-potential metadata; leave it null.
- legacy reward-KL or post-estimator filters are not fully represented here;
  the rebuttal protocol rejects those combinations.
"""
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch


def snapshot_scores(batch):
    """Per-sample scalar score sum at this point of the fit loop."""
    return batch.batch['token_level_scores'].sum(-1).detach().cpu().clone()


def _replicate_group_stats(x, uid, epsilon):
    """Exact replica of the cross-step GRPO group statistics."""
    id2score = defaultdict(list)
    for i in range(len(x)):
        id2score[uid[i]].append(x[i])
    id2mean, id2std = {}, {}
    for idx, vals in id2score.items():
        if len(vals) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        else:
            id2mean[idx] = torch.mean(torch.tensor(vals))
            id2std[idx] = torch.std(torch.tensor([vals]))
    return id2mean, id2std


def dump_group_variance_rows(batch, out_dir, update_id, snap_task,
                             snap_after_penalty, lambda_pred, epsilon,
                             run_meta, diagnostics=None):
    """Write one JSONL.gz of per-sample rows for this update."""
    os.makedirs(out_dir, exist_ok=True)

    x_final = batch.batch['token_level_rewards'].sum(-1).detach().cpu()
    uid = batch.non_tensor_batch['uid']
    traj_uid = batch.non_tensor_batch['traj_uid']
    ntb = batch.non_tensor_batch

    pred_raw = ntb.get('pred_rewards', None)
    pred_acc = ntb.get('pred_accuracy', None)
    action_valid = ntb.get('is_action_valid', None)
    active = ntb.get('active_masks', None)
    diagnostics = {k: v.detach().cpu() for k, v in (diagnostics or {}).items()}
    estimator = run_meta.get('estimator', 'legacy_step')

    # advantage scalar at the last valid response token of each sample
    adv = batch.batch['advantages'].detach().cpu()
    prompt_len = batch.batch['prompts'].shape[-1]
    attn = batch.batch['attention_mask'][:, prompt_len:].detach().cpu()
    if 'optimization_mask' in batch.batch:
        attn = batch.batch['optimization_mask'].detach().cpu()
    valid_len = attn.sum(-1).long()

    id2mean, id2std = _replicate_group_stats(x_final, uid, epsilon)

    # trajectory-level context from the pure-task snapshot
    traj_task_sum = defaultdict(float)
    traj_len = defaultdict(int)
    order_in_traj = {}
    for i in range(len(uid)):
        if diagnostics and not bool(diagnostics['active_row'][i]):
            continue
        traj_task_sum[traj_uid[i]] += float(snap_task[i])
        order_in_traj[i] = traj_len[traj_uid[i]]
        traj_len[traj_uid[i]] += 1

    path = os.path.join(out_dir, f'update_{int(update_id):04d}.jsonl.gz')
    with gzip.open(path, 'wt') as f:
        for i in range(len(uid)):
            if diagnostics and not bool(diagnostics['active_row'][i]):
                continue
            gm = float(id2mean[uid[i]])
            gs = float(id2std[uid[i]])
            if 'normalization_mean' in diagnostics:
                gm = float(diagnostics['normalization_mean'][i])
                gs = float(diagnostics['normalization_std'][i])
            actual_position = torch.nonzero(attn[i], as_tuple=True)[0]
            row = {
                'run_id': run_meta.get('run_id'),
                'seed': run_meta.get('seed'),
                'update_id': int(update_id),
                'group_id': str(uid[i]),
                'trajectory_id': str(traj_uid[i]),
                'env_step_rank': order_in_traj[i],
                'env_step': int(ntb['pred_env_step'][i]) if 'pred_env_step' in ntb else None,
                'estimator': estimator,
                'normalization_scope': 'batch' if estimator == 'gdpo' else 'task_group',
                'norm_by_std': run_meta.get('norm_by_std'),
                'success_only': run_meta.get('success_only', False),
                'loss_agg_mode': run_meta.get('loss_agg_mode'),
                'rms_weighting': run_meta.get('rms_weighting'),
                'active_mask': bool(active[i]) if active is not None else None,
                'is_action_valid': (float(np.asarray(action_valid[i]).reshape(-1)[0])
                                    if action_valid is not None else None),
                'task_score_sample': float(snap_task[i]),
                'pred_delta_raw': (float(ntb['pred_raw_delta'][i]) if 'pred_raw_delta' in ntb else
                                   float(pred_raw[i]) if pred_raw is not None else None),
                'pred_delta_after_boundary': float(pred_raw[i]) if pred_raw is not None else None,
                'phi_prev': float(ntb['pred_phi_prev'][i]) if 'pred_phi_prev' in ntb else None,
                'is_first_step': bool(ntb['pred_is_first_step'][i]) if 'pred_is_first_step' in ntb else None,
                'phi_curr': float(ntb['pred_phi_curr'][i]) if 'pred_phi_curr' in ntb else None,
                'pred_accuracy': float(pred_acc[i]) if pred_acc is not None else None,
                'lambda': float(lambda_pred),
                'penalty_component': float(snap_after_penalty[i] - snap_task[i]),
                'pred_component': float(x_final[i] - snap_after_penalty[i]),
                'x_prenorm': float(x_final[i]),
                'group_mean': gm,
                'group_std': gs,
                'epsilon': float(epsilon),
                'advantage': float(adv[i, actual_position[-1]]) if len(actual_position) else None,
                'response_length': int(valid_len[i]),
                'traj_task_return': traj_task_sum[traj_uid[i]],
                'traj_task_return_is_step_sum': True, # legacy field sums the broadcast score
                'traj_success': traj_task_sum[traj_uid[i]] > 0,
                'traj_n_samples': traj_len[traj_uid[i]],
            }
            for key, values in diagnostics.items():
                value = float(values[i])
                row['estimator_' + key] = value if math.isfinite(value) else None
            f.write(json.dumps(row, allow_nan=False) + '\n')
    return path
