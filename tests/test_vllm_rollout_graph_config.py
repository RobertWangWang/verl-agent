"""Exercise CUDA graph/cache compatibility without constructing a GPU engine."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd as rollout


@pytest.fixture
def construct(monkeypatch):
    engine = Mock()
    factory = Mock(return_value=engine)
    monkeypatch.setattr(rollout, "LLM", factory)
    monkeypatch.setattr(rollout.torch.distributed, "get_world_size", lambda: 8)

    def make(version, eager=False, free_cache=True):
        monkeypatch.setattr(rollout, "package_version", version)
        config = OmegaConf.create({
            "enforce_eager": eager,
            "free_cache_engine": free_cache,
            "tensor_model_parallel_size": 1,
            "prompt_length": 1536,
            "response_length": 512,
            "max_model_len": None,
            "enable_chunked_prefill": False,
            "load_format": "dummy_dtensor",
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.6,
            "max_num_seqs": 32,
            "disable_log_stats": True,
        })
        return rollout.vLLMRollout(
            "/unused/model", config, SimpleNamespace(pad_token_id=0),
            SimpleNamespace(max_position_embeddings=4096),
        )

    return make, factory, engine


@pytest.mark.parametrize("version", ["0.7.3", "0.11.0"])
def test_sleep_mode_allows_graphs_and_preserves_initial_sleep(construct, version):
    make, factory, engine = construct
    instance = make(version)
    assert instance.inference_engine is engine
    assert factory.call_args.kwargs["enforce_eager"] is False
    assert factory.call_args.kwargs["enable_sleep_mode"] is True
    engine.sleep.assert_called_once_with(level=1)


@pytest.mark.parametrize("version", ["0.5.4", "0.6.3", "0.6.3+rocm", "0.7.2"])
def test_legacy_graph_cache_conflict_rejected_before_engine_creation(construct, version):
    make, factory, _ = construct
    with pytest.raises(ValueError, match="require vLLM >= 0.7.3"):
        make(version)
    factory.assert_not_called()


@pytest.mark.parametrize("version", ["0.6.3", "0.11.0"])
@pytest.mark.parametrize("eager,free_cache", [(True, True), (False, False)])
def test_existing_nonconflicting_options_pass_through(construct, version, eager, free_cache):
    make, factory, _ = construct
    make(version, eager=eager, free_cache=free_cache)
    assert factory.call_args.kwargs["enforce_eager"] is eager
