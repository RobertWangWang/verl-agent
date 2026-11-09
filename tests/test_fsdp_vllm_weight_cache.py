"""CPU lifecycle tests for reusing vLLM's level-1 sleep weight backup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl.utils.debug import performance
from verl.workers.sharding_manager import fsdp_vllm


class Device:
    def __init__(self):
        self.rng = 7

    def get_rng_state(self):
        return self.rng

    def set_rng_state(self, state):
        self.rng = state

    def manual_seed(self, seed):
        self.rng = seed

    def current_device(self):
        return "cpu"

    def empty_cache(self):
        pass


class Actor:
    def __init__(self):
        self._fsdp_wrapped_module = torch.nn.Linear(1, 1)
        self.weight = torch.tensor([1.0])
        self.exports = 0

    def state_dict(self):
        self.exports += 1
        return {"weight": self.weight.clone()}

    def train(self):
        pass


class Engine:
    def __init__(self):
        self.weight = None
        self.backup = None
        self.loads = 0
        self.wakes = []
        self.sleeps = 0
        self.model = SimpleNamespace(load_weights=self.load_weights)
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=self.model))
        self.llm_engine = SimpleNamespace(model_executor=SimpleNamespace(driver_worker=SimpleNamespace(worker=worker)))

    def load_weights(self, weights):
        self.loads += 1
        self.weight = dict(weights)["weight"].clone()
        return {"weight"}

    def wake_up(self, tags=None):
        self.wakes.append(tags)
        if (tags is None or "weights" in tags) and self.backup is not None:
            self.weight = self.backup.clone()
            self.backup = None

    def sleep(self, level):
        assert level == 1
        self.sleeps += 1
        self.backup = self.weight.clone()
        self.weight = None


class EngineWithoutTags(Engine):
    def wake_up(self):
        super().wake_up()


@pytest.fixture
def setup_manager(monkeypatch):
    device = Device()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fsdp_vllm, "get_torch_device", lambda: device)
    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: ("0",) * 4)
    monkeypatch.setattr(fsdp_vllm, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp_vllm, "fsdp_version", lambda module: 0)
    monkeypatch.setattr(fsdp_vllm, "is_version_ge", lambda **kwargs: True)
    monkeypatch.setattr(fsdp_vllm, "vllm_version", "0.11.0")
    monkeypatch.setattr(fsdp_vllm.VLLMHijack, "hijack", lambda: None)
    monkeypatch.setattr(fsdp_vllm, "patch_vllm_moe_model_weight_loader", lambda model: None)
    axis = SimpleNamespace(size=lambda: 1, get_local_rank=lambda: 0)

    def make(engine_class=Engine, offload=False):
        actor, engine = Actor(), engine_class()
        manager = fsdp_vllm.FSDPVLLMShardingManager(
            actor, engine, None, device_mesh={"infer_tp": axis, "dp": axis}, offload_param=offload,
        )
        return manager, actor, engine, device

    return make


@pytest.mark.parametrize("engine_class", [Engine, EngineWithoutTags])
def test_repeated_generation_restores_weights_and_preserves_rng(setup_manager, engine_class):
    manager, actor, engine, device = setup_manager(engine_class)
    for expected_rng in (1000, 1001, 1002):
        with manager:
            torch.testing.assert_close(engine.weight, actor.weight)
            assert device.rng == expected_rng
            device.rng += 1
        assert device.rng == 7
    assert actor.exports == engine.loads == 1
    assert engine.sleeps == 3
    assert manager.gen_random_states == 1003
    if engine_class is Engine:
        assert engine.wakes == [["weights"], ["kv_cache"], ["weights", "kv_cache"], ["weights", "kv_cache"]]


def test_invalidation_refreshes_changed_actor_before_generation(setup_manager):
    manager, actor, engine, _ = setup_manager()
    with manager:
        pass
    actor.weight.add_(5)
    manager.invalidate_weights()
    with manager:
        torch.testing.assert_close(engine.weight, torch.tensor([6.0]))
    with manager:
        torch.testing.assert_close(engine.weight, torch.tensor([6.0]))
    assert actor.exports == engine.loads == 2


def test_cached_generation_does_not_reload_offloaded_actor(setup_manager, monkeypatch):
    load, offload = Mock(), Mock()
    monkeypatch.setattr(fsdp_vllm, "load_fsdp_model_to_gpu", load)
    monkeypatch.setattr(fsdp_vllm, "offload_fsdp_model_to_cpu", offload)
    manager, actor, engine, _ = setup_manager(offload=True)
    for _ in range(2):
        with manager:
            pass
    load.assert_called_once_with(actor)
    offload.assert_called_once_with(actor)
    assert engine.loads == 1


def test_failed_sync_is_retried(setup_manager, monkeypatch):
    manager, actor, engine, _ = setup_manager()
    original = engine.model.load_weights
    monkeypatch.setattr(engine.model, "load_weights", Mock(side_effect=RuntimeError("partial load")))
    with pytest.raises(RuntimeError, match="partial load"):
        with manager:
            pass
    assert not manager._weights_synced
    monkeypatch.setattr(engine.model, "load_weights", original)
    with manager:
        torch.testing.assert_close(engine.weight, actor.weight)
    assert actor.exports == 2


def test_failed_restore_invalidates_cache(setup_manager, monkeypatch):
    manager, actor, engine, _ = setup_manager()
    with manager:
        pass
    original = engine.wake_up
    monkeypatch.setattr(engine, "wake_up", Mock(side_effect=RuntimeError("wake failed")))
    with pytest.raises(RuntimeError, match="wake failed"):
        with manager:
            pass
    assert not manager._weights_synced
    monkeypatch.setattr(engine, "wake_up", original)
    with manager:
        pass
    assert actor.exports == engine.loads == 2


def test_failed_sleep_invalidates_cache(setup_manager, monkeypatch):
    manager, _, engine, _ = setup_manager()
    monkeypatch.setattr(engine, "sleep", Mock(side_effect=RuntimeError("sleep failed")))
    with pytest.raises(RuntimeError, match="sleep failed"):
        with manager:
            pass
    assert not manager._weights_synced


def test_generation_error_invalidates_cache(setup_manager):
    manager, actor, engine, _ = setup_manager()
    with pytest.raises(RuntimeError, match="generation failed"):
        with manager:
            raise RuntimeError("generation failed")
    assert not manager._weights_synced
    with manager:
        pass
    assert actor.exports == engine.loads == 2


def test_lora_and_older_versions_do_not_enable_cache(setup_manager, monkeypatch):
    monkeypatch.setattr(fsdp_vllm, "PeftModel", torch.nn.Linear)
    manager, _, _, _ = setup_manager()
    assert not manager._cache_weights
    monkeypatch.setattr(fsdp_vllm, "PeftModel", type("FakePeft", (), {}))
    monkeypatch.setattr(fsdp_vllm, "is_version_ge", lambda **kwargs: False)
    manager, _, _, _ = setup_manager()
    assert not manager._cache_weights


def test_worker_invalidates_before_update_and_checkpoint_errors(setup_manager, monkeypatch):
    from verl.workers import fsdp_workers

    manager, _, _, device = setup_manager()
    monkeypatch.setattr(fsdp_workers, "get_torch_device", lambda: device)
    worker = SimpleNamespace(
        rollout_sharding_manager=manager, _is_offload_param=False,
        checkpoint_manager=SimpleNamespace(load_checkpoint=Mock(side_effect=RuntimeError("load failed"))),
    )
    for invoke, message in (
        (lambda: fsdp_workers.ActorRolloutRefWorker.update_actor(worker, SimpleNamespace(to=Mock(side_effect=RuntimeError("update failed")))), "update failed"),
        (lambda: fsdp_workers.ActorRolloutRefWorker.load_checkpoint(worker, "/unused"), "load failed"),
    ):
        manager._weights_synced = True
        with pytest.raises(RuntimeError, match=message):
            invoke()
        assert not manager._weights_synced


def test_legacy_sync_path_still_transfers_every_generation(setup_manager, monkeypatch):
    class LegacyEngine(Engine):
        def sync_model_weights(self, params, load_format):
            assert load_format == "dtensor"
            self.load_weights(params.items())

        def offload_model_weights(self):
            self.sleep(level=1)

    monkeypatch.setattr(fsdp_vllm, "vllm_version", "0.6.3")
    monkeypatch.setattr(fsdp_vllm, "is_version_ge", lambda **kwargs: False)
    manager, actor, engine, _ = setup_manager(LegacyEngine)
    for _ in range(2):
        with manager:
            torch.testing.assert_close(engine.weight, actor.weight)
    assert actor.exports == engine.loads == 2
    assert engine.wakes == []


def test_spin_dpo_invalidates_before_update_errors(setup_manager, monkeypatch):
    from recipe.spin.fsdp_workers import SPINRolloutRefWorker

    manager, _, _, _ = setup_manager()
    manager._weights_synced = True
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    worker = SimpleNamespace(rollout_sharding_manager=manager)
    data = SimpleNamespace(to=Mock(side_effect=RuntimeError("DPO update failed")))
    with pytest.raises(RuntimeError, match="DPO update failed"):
        SPINRolloutRefWorker.update_actor_dpo(worker, data)
    assert not manager._weights_synced
