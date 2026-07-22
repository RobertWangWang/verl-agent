"""全败组占比指标 + R43 过滤对照臂 (compute_group_outcome_metrics / apply_all_fail_group_filter)。"""
import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    apply_all_fail_group_filter,
    compute_group_outcome_metrics,
)

RESPONSE_LEN = 4


def make_batch(uids, traj_uids, traj_scores):
    """每行一个 step 样本; traj_scores 给出该样本携带的任务分 (放在最后 token)。"""
    bs = len(uids)
    scores = torch.zeros(bs, RESPONSE_LEN, dtype=torch.float32)
    for i, s in enumerate(traj_scores):
        scores[i, -1] = s
    tensors = {
        'token_level_scores': scores,
        'advantages': torch.ones(bs, RESPONSE_LEN, dtype=torch.float32),
    }
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={'uid': np.array(uids, dtype=object),
                     'traj_uid': np.array(traj_uids, dtype=object)},
    )


class TestGroupOutcomeMetrics:
    def test_all_fail_and_success_fractions(self):
        # 组 A: 两轨迹全败; 组 B: 一成一败 (mixed); 组 C: 全成
        data = make_batch(
            uids=['A', 'A', 'B', 'B', 'C'],
            traj_uids=['a1', 'a2', 'b1', 'b2', 'c1'],
            traj_scores=[0.0, 0.0, 10.0, 0.0, 10.0],
        )
        m, all_fail = compute_group_outcome_metrics(data)
        assert m['batch/all_fail_group_frac'] == pytest.approx(1 / 3)
        assert m['batch/all_success_group_frac'] == pytest.approx(1 / 3)
        assert m['batch/mixed_group_frac'] == pytest.approx(1 / 3)
        assert all_fail == {'A'}

    def test_multi_step_trajectory_sums_scores(self):
        """轨迹分跨多个 step 样本累加: 任一步携带成功分即非全败。"""
        data = make_batch(
            uids=['G', 'G', 'G', 'G'],
            traj_uids=['t1', 't1', 't2', 't2'],
            traj_scores=[0.0, 10.0, 0.0, 0.0],
        )
        m, all_fail = compute_group_outcome_metrics(data)
        assert m['batch/all_fail_group_frac'] == 0.0
        assert all_fail == set()

    def test_invalid_penalty_isolation(self):
        """负分 (如 invalid penalty 已注入时) 不该把失败轨迹判成成功——
        本指标要求在 penalty 前调用, 但阈值 >0 对负分也稳健。"""
        data = make_batch(
            uids=['G'], traj_uids=['t1'], traj_scores=[-0.1],
        )
        m, all_fail = compute_group_outcome_metrics(data)
        assert m['batch/all_fail_group_frac'] == 1.0


class TestAllFailGroupFilter:
    def test_zeroes_only_all_fail_group_advantages(self):
        data = make_batch(
            uids=['A', 'A', 'B', 'B'],
            traj_uids=['a1', 'a2', 'b1', 'b2'],
            traj_scores=[0.0, 0.0, 10.0, 0.0],
        )
        _, all_fail = compute_group_outcome_metrics(data)
        data, m = apply_all_fail_group_filter(data, all_fail)
        assert torch.all(data.batch['advantages'][0] == 0)
        assert torch.all(data.batch['advantages'][1] == 0)
        assert torch.all(data.batch['advantages'][2] == 1)
        assert torch.all(data.batch['advantages'][3] == 1)
        assert m['batch/filtered_all_fail_samples'] == 2
        assert m['batch/filtered_sample_frac'] == pytest.approx(0.5)

    def test_noop_when_no_all_fail_groups(self):
        data = make_batch(uids=['B', 'B'], traj_uids=['b1', 'b2'],
                          traj_scores=[10.0, 10.0])
        data, m = apply_all_fail_group_filter(data, set())
        assert torch.all(data.batch['advantages'] == 1)
        assert m['batch/filtered_all_fail_samples'] == 0
