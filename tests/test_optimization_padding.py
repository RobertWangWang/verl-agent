"""Distributed alignment copies must not become new optimization samples."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.utils import adjust_batch
from verl import DataProto
from verl.workers.actor import dp_actor


def padding_config(enabled):
    return OmegaConf.create({
        'trainer': {'n_gpus_per_node': 1, 'nnodes': 1},
        'algorithm': {'use_kl_in_reward': False, 'rebuttal': {'enable': enabled}},
        'actor_rollout_ref': {
            'actor': {'use_kl_loss': False, 'ppo_micro_batch_size_per_gpu': 4},
            'rollout': {'log_prob_micro_batch_size_per_gpu': 4},
        },
    })


@pytest.mark.parametrize('enabled', [False, True])
def test_adjust_batch_marks_only_real_copies_without_changing_legacy(enabled):
    source = DataProto.from_dict(
        tensors={'input_ids': torch.arange(3).unsqueeze(-1)},
        non_tensors={'traj_uid': np.array(['a', 'a', 'b']), 'active_masks': np.ones(3, dtype=bool)},
    )
    padded = adjust_batch(padding_config(enabled), source)
    assert len(padded) == 4
    assert padded.non_tensor_batch['active_masks'].all()  # Environment activity is a different concept.
    if enabled:
        assert padded.non_tensor_batch['optimization_padding'].tolist() == [False, False, False, True]
        padded.reorder(torch.tensor([3, 1, 2, 0]))
        assert padded.non_tensor_batch['optimization_padding'].tolist() == [True, False, False, False]
    else:
        assert 'optimization_padding' not in padded.non_tensor_batch


def test_divisible_batch_has_explicit_false_padding_markers():
    source = DataProto.from_dict(tensors={'input_ids': torch.ones(4, 1)})
    padded = adjust_batch(padding_config(True), source)
    assert not padded.non_tensor_batch['optimization_padding'].any()


def _actor_gradient(monkeypatch, reduction, *, padded=False, optimization_mask=True, all_padding=False):
    monkeypatch.setattr(dp_actor, 'get_torch_device', lambda: SimpleNamespace(current_device=lambda: 'cpu'))
    monkeypatch.setattr('verl.utils.debug.performance._get_current_mem_info', lambda: ('0',) * 4)
    config = OmegaConf.create({
        'use_remove_padding': False, 'use_fused_kernels': False, 'use_torch_compile': False,
        'ulysses_sequence_parallel_size': 1, 'use_kl_loss': True, 'kl_loss_coef': 0.2,
        'kl_loss_type': 'low_var_kl', 'ppo_epochs': 1, 'ppo_mini_batch_size': 4,
        'ppo_micro_batch_size_per_gpu': 4, 'use_dynamic_bsz': False,
        'clip_ratio': 0.2, 'clip_ratio_low': None, 'clip_ratio_high': None,
        'clip_ratio_c': 3.0, 'entropy_coeff': 0.05, 'loss_agg_mode': reduction,
        'policy_loss': {'loss_mode': 'vanilla'},
    })
    model = torch.nn.Embedding(4, 2)
    with torch.no_grad():
        model.weight.fill_(-0.2)
    actor = dp_actor.DataParallelPPOActor(config, model, torch.optim.SGD(model.parameters(), lr=0.1))
    gradients = []

    def forward(micro_batch, temperature, calculate_entropy):
        logits = model(micro_batch['input_ids'][:, 0])
        return 1.0 + 0.3 * logits, logits

    def step():
        gradients.append(model.weight.grad.detach().clone())
        return model.weight.grad.norm()

    monkeypatch.setattr(actor, '_forward_micro_batch', forward)
    monkeypatch.setattr(actor, '_optimizer_step', step)
    n = 4 if padded else 2
    attention = torch.ones(n, 3)
    attention[1, -1] = 0
    tensors = {
        'input_ids': torch.arange(n).unsqueeze(-1),
        'attention_mask': attention,
        'position_ids': torch.zeros(n, 3, dtype=torch.long),
        'responses': torch.ones(n, 2, dtype=torch.long),
        'old_log_probs': torch.full((n, 2), -0.2),
        'ref_log_prob': torch.full((n, 2), -0.5),
        'advantages': torch.tensor([[1., 1.], [-1., -1.], [99., 99.], [99., 99.]])[:n],
    }
    if optimization_mask:
        tensors['optimization_mask'] = attention[:, -2:].clone()
        if padded:
            tensors['optimization_mask'][2:] = 0
        if all_padding:
            tensors['optimization_mask'][:] = 0
    data = DataProto.from_dict(tensors=tensors, meta_info={'temperature': 1.0})
    metrics = actor.update_policy(data)
    return gradients[0], metrics


@pytest.mark.parametrize('reduction', [
    'token-mean', 'seq-mean-token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm',
])
def test_actor_padding_excluded_from_policy_entropy_and_kl_denominators(monkeypatch, reduction):
    expected, _ = _actor_gradient(monkeypatch, reduction, optimization_mask=False)
    actual, metrics = _actor_gradient(monkeypatch, reduction, padded=True)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[2:]) == 0
    assert all(np.isfinite(value).all() for value in metrics.values())


@pytest.mark.parametrize('reduction', ['token-mean', 'seq-mean-token-mean'])
def test_all_padding_microbatch_has_finite_differentiable_zero(monkeypatch, reduction):
    gradient, metrics = _actor_gradient(monkeypatch, reduction, padded=True, all_padding=True)
    assert torch.count_nonzero(gradient) == 0
    assert all(np.isfinite(value).all() for value in metrics.values())
