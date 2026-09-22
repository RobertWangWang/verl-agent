"""Tests for agent_system.telemetry.group_variance_logger.

The load-bearing property: the logger's replicated group statistics must
equal, bit-for-bit, what compute_grpo_outcome_advantage divides by, and the
component decomposition must close (task + penalty + pred == x_prenorm).
"""
import gzip
import json

import numpy as np
import torch

from agent_system.telemetry.group_variance_logger import _replicate_group_stats, dump_group_variance_rows, snapshot_scores
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _rand_batch(n_groups=3, n_traj=4, n_steps=5, seed=0):
    rng = np.random.default_rng(seed)
    bsz = n_groups * n_traj * n_steps
    resp_len = 7
    uid, traj_uid = [], []
    for g in range(n_groups):
        for t in range(n_traj):
            for s in range(n_steps):
                uid.append(f'g{g}')
                traj_uid.append(f'g{g}_t{t}')
    scores = torch.zeros(bsz, resp_len)
    # scalar per-sample reward on the last token, like the real pipeline
    scores[:, -1] = torch.tensor(rng.normal(0, 1, bsz), dtype=torch.float32)
    mask = torch.ones(bsz, resp_len)
    return scores, mask, np.array(uid), np.array(traj_uid)


def test_group_stats_match_estimator():
    scores, mask, uid, traj_uid = _rand_batch()
    x = scores.sum(-1)
    id2mean, id2std = _replicate_group_stats(x, uid, epsilon=1e-6)
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=scores, response_mask=mask, index=uid,
        traj_index=traj_uid, epsilon=1e-6, norm_adv_by_std_in_grpo=True,
        compute_mean_std_cross_steps=True)
    # reconstruct the estimator's z-score from the logger's stats
    for i in range(len(x)):
        expect = (x[i] - id2mean[uid[i]]) / (id2std[uid[i]] + 1e-6)
        got = adv[i, -1]  # broadcast to every response token
        assert torch.isclose(expect, got, atol=0, rtol=0), (i, expect, got)


def test_singleton_group_convention():
    x = torch.tensor([2.5])
    id2mean, id2std = _replicate_group_stats(x, np.array(['g0']), 1e-6)
    assert float(id2mean['g0']) == 0.0 and float(id2std['g0']) == 1.0


class _FakeBatch:
    def __init__(self, scores, mask, uid, traj_uid, adv, prompts_len=3):
        full_mask = torch.cat([torch.ones(scores.shape[0], prompts_len), mask], dim=-1)
        self.batch = {
            'token_level_scores': scores,
            'token_level_rewards': scores,
            'advantages': adv,
            'prompts': torch.zeros(scores.shape[0], prompts_len, dtype=torch.long),
            'attention_mask': full_mask,
        }
        self.non_tensor_batch = {
            'uid': uid, 'traj_uid': traj_uid,
            'pred_rewards': np.zeros(scores.shape[0], dtype=np.float32),
            'pred_accuracy': np.zeros(scores.shape[0], dtype=np.float32),
            'is_action_valid': np.ones(scores.shape[0], dtype=np.float32),
            'active_masks': np.ones(scores.shape[0], dtype=bool),
        }


def test_dump_decomposition_closes(tmp_path):
    scores, mask, uid, traj_uid = _rand_batch(seed=1)
    snap_task = scores.sum(-1).clone() - 0.3   # pretend task part
    snap_after_pen = scores.sum(-1).clone() - 0.1  # penalty added 0.2
    adv = torch.zeros(scores.shape[0], mask.shape[-1])
    batch = _FakeBatch(scores, mask, uid, traj_uid, adv)
    path = dump_group_variance_rows(
        batch, out_dir=str(tmp_path), update_id=7, snap_task=snap_task,
        snap_after_penalty=snap_after_pen, lambda_pred=0.1, epsilon=1e-6,
        run_meta={'run_id': 'unit', 'seed': 0})
    rows = [json.loads(line) for line in gzip.open(path, 'rt')]
    assert len(rows) == len(uid)
    for r in rows:
        total = r['task_score_sample'] + r['penalty_component'] + r['pred_component']
        assert abs(total - r['x_prenorm']) < 1e-5
        assert r['group_id'].startswith('g') and r['update_id'] == 7
    # env_step_rank counts 0..n_steps-1 within each trajectory
    per_traj = {}
    for r in rows:
        per_traj.setdefault(r['trajectory_id'], []).append(r['env_step_rank'])
    for ranks in per_traj.values():
        assert sorted(ranks) == list(range(len(ranks)))


def test_snapshot_is_independent_copy():
    scores, mask, uid, traj_uid = _rand_batch(seed=2)
    batch = _FakeBatch(scores, mask, uid, traj_uid, torch.zeros_like(scores))
    snap = snapshot_scores(batch)
    batch.batch['token_level_scores'][:, -1] += 99.0
    assert not torch.allclose(snap, batch.batch['token_level_scores'].sum(-1))
