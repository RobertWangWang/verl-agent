#!/usr/bin/env python3
"""Small torchrun probes; wrap the launcher in shell timeout to bound driver hangs."""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from functools import partial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--mode", choices=("collectives", "fsdp"), default="collectives")
    parser.add_argument("--numel", type=int, default=1024, help="Float32 elements per collective input")
    parser.add_argument("--device-mesh", action="store_true", help="Use a 1D CUDA mesh and DTensor FSDP state dict")
    parser.add_argument("--width", type=int, default=64, help="FSDP linear-layer input/output width")
    parser.add_argument("--layers", type=int, default=4, help="Number of FSDP linear layers")
    args = parser.parse_args()
    if min(args.iterations, args.timeout_seconds, args.numel, args.width, args.layers) < 1:
        parser.error("iterations, timeout-seconds, numel, width, and layers must be positive")
    if args.mode == "fsdp" and args.backend != "nccl":
        parser.error("fsdp mode requires --backend nccl and CUDA")
    if args.device_mesh and args.mode != "fsdp":
        parser.error("--device-mesh requires --mode fsdp")

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    def log(event, **details):
        record = {"time": datetime.now(timezone.utc).isoformat(), "rank": rank, "event": event, **details}
        os.write(1, (json.dumps(record) + "\n").encode())

    device = torch.device("cpu")
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL requires an available CUDA device")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    env_names = (
        "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE",
        "NCCL_CUMEM_ENABLE", "NCCL_CUMEM_HOST_ENABLE", "NCCL_ALGO", "NCCL_PROTO",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_BLOCKING_WAIT",
    )
    nccl_version = torch.cuda.nccl.version() if dist.is_nccl_available() else None
    log("versions", torch=torch.__version__, cuda=torch.version.cuda, nccl=nccl_version, device=str(device),
        backend=args.backend, mode=args.mode, numel=args.numel, device_mesh=args.device_mesh, width=args.width, layers=args.layers,
        env={name: os.environ[name] for name in env_names if name in os.environ})
    try:
        log("init.begin")
        dist.init_process_group(backend=args.backend, timeout=timedelta(seconds=args.timeout_seconds))
        world = dist.get_world_size()
        log("init.end", world_size=world)
        if args.mode == "collectives":
            for iteration in range(args.iterations):
                values = torch.full((args.numel,), float(rank + 1), device=device)
                log("all_reduce.begin", iteration=iteration)
                dist.all_reduce(values)
                synchronize()
                torch.testing.assert_close(values, torch.full_like(values, world * (world + 1) / 2), rtol=0, atol=0)
                log("all_reduce.end", iteration=iteration)
                source = torch.full((args.numel,), float(rank + iteration), device=device)
                gathered = [torch.empty_like(source) for _ in range(world)]
                log("all_gather.begin", iteration=iteration)
                dist.all_gather(gathered, source)
                synchronize()
                for peer, values in enumerate(gathered):
                    torch.testing.assert_close(values, torch.full_like(values, peer + iteration), rtol=0, atol=0)
                log("all_gather.end", iteration=iteration)
        else:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            torch.manual_seed(0)
            model = torch.nn.Sequential(*(torch.nn.Sequential(torch.nn.Linear(args.width, args.width), torch.nn.GELU()) for _ in range(args.layers)))
            expected_keys = set(model.state_dict())
            mesh = None
            if args.device_mesh:
                from torch.distributed.device_mesh import init_device_mesh

                log("device_mesh.begin")
                mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
                log("device_mesh.end")
            log("fsdp.wrap.begin")
            model = FSDP(model, device_id=device, device_mesh=mesh, sync_module_states=True,
                         auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1))
            synchronize()
            wrappers = sum(isinstance(module, FSDP) for module in model.modules())
            if wrappers < 2:
                raise AssertionError("Expected nested FSDP wrappers")
            log("fsdp.wrap.end", wrappers=wrappers)
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=False)):
                for iteration in range(args.iterations):
                    log("state_dict.begin", iteration=iteration)
                    state = model.state_dict()
                    synchronize()
                    if set(state) != expected_keys:
                        raise AssertionError(f"Unexpected state_dict keys: {list(state)}")
                    for name, value in state.items():
                        if hasattr(value, "local_shards"):
                            local = [shard.tensor for shard in value.local_shards()]
                        else:
                            local = [value.to_local() if hasattr(value, "to_local") else value]
                        if not all(torch.isfinite(tensor).all().item() for tensor in local):
                            raise AssertionError(f"Non-finite state_dict tensor: {name}")
                    log("state_dict.end", iteration=iteration, tensors=len(state), types=sorted({type(value).__name__ for value in state.values()}))
                    del state
        log("barrier.begin")
        dist.barrier()
        synchronize()
        log("passed")
    except BaseException as error:
        log("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if dist.is_initialized():
            log("destroy.begin")
            dist.destroy_process_group()
            log("destroy.end")


if __name__ == "__main__":
    main()
