"""Boundary ablation and per-transition prediction audit metadata, CPU only."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.environments import env_manager
from agent_system.memory import HybridMemory, SimpleMemory
from agent_system.memory.predictive_memory import PredictiveMemory
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector, collect_prediction_info
from tests.test_ps_alfworld_env_manager import FakeAlfWorldEnvs, fake_projection, make_config, response
from verl import DataProto


@pytest.mark.parametrize('mode', PredictiveMemory.REWARD_MODES)
def test_drop_first_preserves_subsequent_reward_and_raw_delta(mode):
    original = PredictiveMemory(lambda_pred=0.3, reward_mode=mode)
    dropped = PredictiveMemory(lambda_pred=0.3, reward_mode=mode, drop_first_reward=True)
    for memory in (original, dropped):
        memory.reset(1)
    for step, accuracy in enumerate((0.8, 0.2, 0.6)):
        expected = original.compute_prediction_reward(0, step, accuracy)
        actual = dropped.compute_prediction_reward(0, step, accuracy)
        assert actual.shaped_reward == pytest.approx(0.0 if step == 0 else expected.shaped_reward)
        assert actual.phi_prev == expected.phi_prev
        assert actual.raw_delta == expected.raw_delta
        assert actual.prediction_accuracy == expected.prediction_accuracy
        assert actual.is_first_step is (step == 0)
        assert actual.env_step == step + 1


def test_zero_accuracy_consumes_first_step_independently_and_reset_restarts():
    memory = PredictiveMemory(lambda_pred=1.0, drop_first_reward=True)
    memory.reset(2)
    first = memory.compute_prediction_reward(0, 0, 0.0)
    second = memory.compute_prediction_reward(0, 1, 0.7)
    # Different call counts and turn indices cannot share a boundary flag.
    other_first = memory.compute_prediction_reward(1, 10, 0.5)
    third = memory.compute_prediction_reward(0, 2, 0.2)
    assert first.is_first_step and first.shaped_reward == 0.0
    assert not second.is_first_step and second.shaped_reward == pytest.approx(0.7)
    assert other_first.is_first_step and other_first.shaped_reward == 0.0
    assert other_first.raw_delta == 0.5 and other_first.env_step == 1
    assert third.raw_delta == pytest.approx(-0.5) and third.env_step == 3
    memory.reset(1)
    restarted = memory.compute_prediction_reward(0, 0, 0.4)
    assert restarted.is_first_step and restarted.env_step == 1
    assert restarted.phi_prev == 0.0 and restarted.raw_delta == 0.4
    assert restarted.shaped_reward == 0.0


def test_default_potential_rewards_keep_original_formula():
    memory = PredictiveMemory(lambda_pred=0.25)
    memory.reset(1)
    actual = [memory.compute_prediction_reward(0, i, phi) for i, phi in enumerate((0.8, 0.2, 0.6))]
    assert [r.shaped_reward for r in actual] == pytest.approx([0.2, -0.15, 0.1])
    assert [r.raw_delta for r in actual] == pytest.approx([0.8, -0.6, 0.4])


@pytest.mark.parametrize('env_name,manager_type', [
    ('alfworld', env_manager.AlfWorldEnvironmentManager),
    ('hiddenrule', env_manager.HiddenRuleEnvironmentManager),
    ('webshop', env_manager.WebshopEnvironmentManager),
    ('sciworld', env_manager.SciWorldEnvironmentManager),
])
def test_all_prediction_managers_forward_flag_and_keep_disabled_path(env_name, manager_type):
    config = OmegaConf.create({'env': {'history_length': 2, env_name: {
        'prediction': {'enable': True, 'drop_first_reward': True}}}})
    manager = manager_type(object(), fake_projection, config)
    assert isinstance(manager.memory, HybridMemory)
    assert manager.memory.memory.drop_first_reward is True
    config.env[env_name].prediction.enable = False
    disabled = manager_type(object(), fake_projection, config)
    assert isinstance(disabled.memory, SimpleMemory)


def test_alfworld_scores_first_prediction_and_updates_potential_before_reset():
    config = make_config(True)
    config.env.alfworld.prediction.drop_first_reward = True
    manager = env_manager.AlfWorldEnvironmentManager(FakeAlfWorldEnvs(), fake_projection, config)
    manager.reset(kwargs=None)
    text = response('<predict>next_location: cabinet 1; target_visible: no; task_done: no</predict>', 'go to cabinet 1')
    actions = [text]
    _, _, _, infos = manager.step(actions)
    first = infos[0]
    assert actions == [text]
    assert first['pred_accuracy'] == pytest.approx(1.0)
    assert first['pred_reward'] == 0.0
    assert first['pred_phi_prev'] == 0.0 and first['pred_raw_delta'] == pytest.approx(1.0)
    assert first['pred_is_first_step'] and first['pred_env_step'] == 1
    assert manager.memory.memory.predictions[0]  # Still parsed and recorded.
    _, _, _, infos = manager.step([response('', 'open cabinet 1')])
    second = infos[0]
    assert second['pred_reward'] == pytest.approx(-1.0)
    assert second['pred_phi_prev'] == pytest.approx(1.0)
    assert not second['pred_is_first_step'] and second['pred_env_step'] == 2
    manager.reset(kwargs=None)
    _, _, _, infos = manager.step(actions)
    assert infos[0]['pred_reward'] == 0.0 and infos[0]['pred_is_first_step']
    assert infos[0]['pred_env_step'] == 1


def test_random_potential_records_actual_potential_separately_from_accuracy():
    config = make_config(True)
    config.env.alfworld.prediction.reward_mode = 'random_potential'
    manager = env_manager.AlfWorldEnvironmentManager(FakeAlfWorldEnvs(), fake_projection, config)
    manager.reset(kwargs=None)
    _, _, _, infos = manager.step([response('', 'go to cabinet 1')])
    info = infos[0]
    assert info['pred_accuracy'] == 0.0
    assert info['pred_phi_curr'] > 0.0
    assert info['pred_phi_curr'] == pytest.approx(info['pred_phi_prev'] + info['pred_raw_delta'])
    collected = collect_prediction_info(infos)
    assert collected['pred_phi_curr'][0] == pytest.approx(info['pred_phi_curr'])


def test_prediction_info_accepts_legacy_and_rejects_partial_metadata():
    legacy = {'pred_reward': 0.2, 'pred_accuracy': 0.2, 'pred_parse_valid': True}
    assert set(collect_prediction_info([legacy])) == {'pred_rewards', 'pred_accuracy', 'pred_parse_valid'}
    assert collect_prediction_info([{}]) == {}
    with pytest.raises(ValueError, match='pred_phi_prev'):
        collect_prediction_info([dict(legacy, pred_phi_prev=0.0), legacy])


class _RolloutEnvs:
    """Two episodes of lengths one and two, sharing a single batched rollout."""

    def __init__(self):
        self.memory = PredictiveMemory(lambda_pred=1.0, drop_first_reward=True)

    def reset(self, kwargs=None):
        self.memory.reset(2)
        self.step_index = 0
        return {'text': ['obs', 'obs'], 'image': None}, [{}, {}]

    def step(self, text_actions):
        self.step_index += 1
        infos = []
        accuracies = (0.8, 0.4) if self.step_index == 1 else (0.9, 0.6)
        for i, accuracy in enumerate(accuracies):
            reward = self.memory.compute_prediction_reward(i, self.step_index, accuracy)
            infos.append({
                'pred_reward': reward.shaped_reward, 'pred_accuracy': accuracy, 'pred_parse_valid': True,
                'pred_phi_prev': reward.phi_prev, 'pred_raw_delta': reward.raw_delta,
                'pred_is_first_step': reward.is_first_step, 'pred_env_step': reward.env_step,
            })
        return {'text': ['obs', 'obs'], 'image': None}, np.zeros(2), np.array([True, self.step_index == 2]), infos

    def success_evaluator(self, **kwargs):
        return {'success_rate': np.zeros(2)}


class _RolloutActor:
    world_size = 1

    def generate_sequences(self, batch):
        tensors = dict(batch.batch.items())
        tensors['responses'] = torch.tensor([[9, 10]] * len(batch))
        return DataProto.from_dict(tensors=tensors)


class _Tokenizer:
    def batch_decode(self, responses, **kwargs):
        return ['<predict>unchanged</predict><action>look</action>'] * len(responses)


def test_full_rollout_keeps_metadata_and_response_aligned_after_filter_and_reorder(monkeypatch):
    config = OmegaConf.create({'env': {'rollout': {'n': 1}, 'max_steps': 3}})
    collector = TrajectoryCollector(config, _Tokenizer())
    initial = DataProto.from_dict(tensors={'input_ids': torch.ones((2, 2), dtype=torch.long)})

    def preprocess(**kwargs):
        return DataProto.from_dict(
            tensors={key: torch.ones((2, 2), dtype=torch.long) for key in ('input_ids', 'attention_mask', 'position_ids')},
            non_tensors={'raw_prompt_ids': np.array([[1, 1], [1, 1]], dtype=object)},
        )

    monkeypatch.setattr(collector, 'preprocess_batch', preprocess)
    parts = collector.vanilla_multi_turn_loop(initial, _RolloutActor(), _RolloutEnvs())
    output = collector.gather_rollout_data(*parts)
    assert len(output) == 3  # The finished environment's extra batch row is discarded.
    audit = output.non_tensor_batch
    assert audit['pred_env_step'].tolist() == [1, 1, 2]
    assert audit['pred_is_first_step'].tolist() == [True, True, False]
    assert audit['pred_raw_delta'].tolist() == pytest.approx([0.8, 0.4, 0.2])
    assert audit['pred_rewards'].tolist() == pytest.approx([0.0, 0.0, 0.2])
    assert audit['pred_phi_prev'].tolist() == pytest.approx([0.0, 0.0, 0.4])
    assert output.batch['responses'].tolist() == [[9, 10], [9, 10], [9, 10]]
    output.reorder(torch.tensor([2, 0, 1]))
    assert output.non_tensor_batch['pred_env_step'].tolist() == [2, 1, 1]
    assert output.non_tensor_batch['pred_raw_delta'].tolist() == pytest.approx([0.2, 0.8, 0.4])
    assert output.non_tensor_batch['traj_uid'][0] == output.non_tensor_batch['traj_uid'][2]


@pytest.mark.parametrize('eval_seed,expected', [('absent', 1007), (None, 1007), (123, 123), (0, 0)])
def test_eval_seed_only_overrides_validation_builder(monkeypatch, eval_seed, expected):
    import agent_system.environments.env_package.hiddenrule as hiddenrule

    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(hiddenrule, 'build_hiddenrule_envs', build)
    config = OmegaConf.create({
        'env': {'env_name': 'hiddenrule/HiddenRuleEnv', 'seed': 7, 'history_length': 2,
                'rollout': {'n': 4}, 'resources_per_worker': {}, 'max_steps': 3,
                'hiddenrule': {'require_think_tags': False}},
        'data': {'train_batch_size': 2, 'val_batch_size': 3},
    })
    if eval_seed != 'absent':
        config.env.eval_seed = eval_seed
    env_manager.make_envs(config)
    assert calls[0]['seed'] == 7 and calls[0]['is_train']
    assert calls[0]['group_n'] == 4
    assert calls[1]['seed'] == expected and not calls[1]['is_train']
    assert calls[1]['group_n'] == 1
