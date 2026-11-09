"""Shared implementation of run_smoke.sh and run_train.sh; no parallel jobs."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from examples.rebuttal import launch


def parser(mode):
    if mode not in {"smoke", "train"}:
        raise ValueError(f"Unknown run mode: {mode}")
    p = launch.parser()
    p.prog = f"bash examples/rebuttal/run_{mode}.sh"
    p.description = (
        "生成并检查 rebuttal 冒烟配置，--execute 后串行运行。默认 A01、seed=0、2 updates，每步验证和保存。"
        if mode == "smoke" else
        "生成并检查 rebuttal 正式训练配置，--execute 后按 arm、seed 顺序串行运行，失败立即停止。"
    )
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.add_argument("--dry-run", action="store_true", help="只生成完整 manifest；默认行为，不能与 --execute 同时使用")
    p._option_string_actions["--execute"].help = "串行执行全部选定 runs；不后台启动、不自动重试或恢复"
    p._option_string_actions["--manifest"].help = "外层完整配置清单的新文件；默认 OUTPUT_ROOT/_manifests/ 下自动生成"
    p._option_string_actions["--output-root"].help = "所有 run 的输出根目录；每个 arm/seed 使用独立子目录"
    p._option_string_actions["--gpus"].help = "每个 run 使用同一组显式 CUDA 索引，例如 0,1；不会自动并行分卡"
    p.epilog = """示例（替换为本机模型目录；PYTHON_BIN 可指定 Python 可执行文件）：
  bash examples/rebuttal/run_smoke.sh --model-4b /models/Qwen3-4B --gpus 0,1 --output-root /experiments/smoke
  bash examples/rebuttal/run_smoke.sh --model-4b /models/Qwen3-4B --gpus 0,1 --output-root /experiments/smoke --execute
  bash examples/rebuttal/run_train.sh --arms A00,A01,A02 --seeds 0,1,2 --model-4b /models/Qwen3-4B --gpus 0,1 --output-root /experiments/rebuttal --execute
  bash examples/rebuttal/run_train.sh --arms core --model-4b /models/Qwen3-4B --model-1.7b /models/Qwen3-1.7B --gpus 0,1,2,3 --output-root /experiments/rebuttal --execute

模型路径、GPU 和输出目录必须显式指定。默认只生成配置，不加载模型或启动 Ray。
正式 A 臂为 150 updates，B 臂为 60 updates；在线 val32 不是正式 140-game 终点评估。
所有 runs 先统一校验，再按清单顺序执行；每个 run 的日志在 <run>/train.log。
"""
    if mode == "smoke":
        p.set_defaults(arms="A01", seeds="0", smoke_steps=2)
    return p


def preflight(runs, manifest_path):
    """Check the complete queue before creating anything or launching one run."""
    roots = [Path(run["paths"]["root"]).resolve() for run in runs]
    if len(set(roots)) != len(roots):
        raise ValueError("选定 runs 的输出目录重复")
    for root in roots:
        if root.exists():
            raise ValueError(f"run 输出目录已存在，拒绝覆盖或自动恢复: {root}")
        if manifest_path == root or root in manifest_path.parents:
            raise ValueError(f"外层 manifest 不能位于 run 输出目录内: {root}")
        if manifest_path in root.parents:
            raise ValueError(f"manifest 文件路径与 run 的父目录冲突: {manifest_path}")
    if manifest_path.exists():
        raise ValueError(f"manifest 已存在，拒绝覆盖: {manifest_path}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"smoke", "train"}:
        mode_parser = argparse.ArgumentParser(description=__doc__)
        mode_parser.add_argument("mode", choices=["smoke", "train"])
        mode_parser.parse_args(argv)
        return None
    mode, remaining = argv[0], argv[1:]
    p = parser(mode)
    args = p.parse_args(remaining)
    try:
        if args.dry_run and args.execute:
            raise ValueError("--dry-run 和 --execute 不能同时使用")
        if mode == "train" and args.smoke_steps is not None:
            raise ValueError("正式训练不接受 --smoke-steps；请使用 run_smoke.sh")
        names = launch.select_arms(args.arms)
        seeds = launch.parse_seeds(args.seeds)
        runs = [launch.build_run(name, seed, args) for name in names for seed in seeds]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest_path = (
            args.manifest or args.output_root / "_manifests" / f"{mode}_{stamp}_{uuid4().hex[:8]}.json"
        ).expanduser().resolve()
        preflight(runs, manifest_path)
    except ValueError as exc:
        p.error(str(exc))

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dry_run": not args.execute,
        "execution_order": "serial, arm then seed; stop at first failure",
        "repository": str(launch.REPO_ROOT),
        "run_count": len(runs),
        "evidence": launch.evidence(),
        "limitations": [
            "H200 memory/throughput and simulator resource use require a real smoke run.",
            "Online val32 is not the formal endpoint eval140.",
            "W/X, fixed-game endpoint evaluation and gradient-vector measurement are not implemented.",
            "Each run uses the specified GPU set; no automatic concurrent scheduling is performed.",
        ],
        "runs": runs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"{'EXECUTE' if args.execute else 'DRY RUN'}: {len(runs)} runs; manifest: {manifest_path}", flush=True)
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run['run_id']} | GPUs={run['environment']['CUDA_VISIBLE_DEVICES']} | log: {run['paths']['log']}", flush=True)
        if not args.execute:
            print(run["prepare_command"], flush=True)
            print(run["train_command"], flush=True)

    if args.execute:
        # Execute exactly the JSON document that was persisted for inspection.
        # Do not regenerate config independently for each run.
        with manifest_path.open(encoding="utf-8") as stream:
            stored = json.load(stream)
        for index, run in enumerate(stored["runs"], start=1):
            print(f"START [{index}/{len(runs)}] {run['run_id']} | log: {run['paths']['log']}", flush=True)
            launch.execute_run(run, stored)
            print(f"DONE [{index}/{len(runs)}] {run['run_id']}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
