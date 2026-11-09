"""Opt-in research seeding. Seeding does not imply bitwise CUDA determinism."""

import contextlib
import random


def derived_seed(seed, rank=0):
    """Keep NumPy and Torch seeds in their shared unsigned 32-bit range."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    return (seed + rank) % 2**32


def seed_everything(seed, rank=0):
    """Seed Python, NumPy, Torch CPU and CUDA; None preserves legacy behavior.

    PYTHONHASHSEED must be set by the launcher before Python starts; changing it
    here would not reseed this process's hash function.
    """
    if seed is None:
        return None
    import numpy as np
    import torch

    effective = derived_seed(seed, rank)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return {"base_seed": seed, "rank": rank, "effective_seed": effective}


def make_device_rng_state(device, seed):
    """Construct a CUDA/NPU RNG state without advancing the current stream."""
    previous = device.get_rng_state()
    try:
        device.manual_seed(derived_seed(seed))
        return device.get_rng_state()
    finally:
        device.set_rng_state(previous)


def rollout_sampling_kwargs(config, sampling_params):
    """Copy sampling options without reusing the engine seed per request.

    A fixed SamplingParams.seed can give identical prompts identical samples,
    collapsing the diversity of a GRPO group. ``rollout.seed`` initializes the
    engine only; request sampling consumes its advancing RNG stream.
    """
    return {key: config.get(key) for key in config.keys() if key != "seed" and hasattr(sampling_params, str(key))}


@contextlib.contextmanager
def isolated_generation_rng(manager, state):
    """Swap a sharding manager's generation stream; retain its advanced state.

    ``state`` is a one-element list so the caller can retain validation's stream
    across calls. The surrounding manager context advances gen_random_states.
    """
    training_state = manager.gen_random_states
    manager.gen_random_states = state[0]
    try:
        yield
    finally:
        state[0] = manager.gen_random_states
        manager.gen_random_states = training_state
