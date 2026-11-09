#!/usr/bin/env python3
"""Offline scalar diagnostics for M01/M02; no model, optimizer, or training run.

Examples (run from the repository root)::

    python scripts/rebuttal_analysis.py synthetic --output boundary.json
    python scripts/rebuttal_analysis.py replay runs/telemetry --output replay.json
    python scripts/rebuttal_analysis.py replay update_0050.jsonl.gz --include-groups

Each telemetry file must contain one complete update. Files are read one at a
time, and results are streamed into one JSON document. Replay cannot recreate
unlogged rollouts, gradients, clipping, Adam updates, or online training effects.
"""

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

# Support direct execution as well as ``python -m scripts.rebuttal_analysis``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verl.trainer.ppo.rebuttal import compute_rebuttal_advantages

LIMITATIONS = [
    "Frozen-rollout scalar replay is a computational counterfactual, not a replacement for independent training.",
    "Advantage RMS is not a gradient norm, clipped policy update, KL contribution, or Adam parameter step.",
    "Trajectory replay assumes the file contains every active optimization step of each trajectory.",
    "Row and response-token-weighted RMS are both reported; the latter need not equal the configured policy-loss reduction.",
    "In the legacy actor, token means are computed per microbatch and accumulated equally; whole-batch token weighting is only a nominal diagnostic.",
    "Group variance summaries are descriptive, not independent seed-level statistical evidence.",
    "Replays apply the named base estimator; subsequent advantage filtering or RMS rescaling from the original run is not reproduced.",
    "Replay arithmetic uses float64; it need not be bitwise identical to training-time float32 arithmetic.",
]
VARIANTS = {"step": ("step", True), "mean_only": ("step", False), "trajectory": ("trajectory", True), "gdpo": ("gdpo", True)}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _stats(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.numel() == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "std_sample": None, "rms": None}
    return {
        "count": values.numel(), "mean": values.mean().item(), "min": values.min().item(), "max": values.max().item(),
        "std_sample": values.std(unbiased=True).item() if values.numel() > 1 else None,
        "rms": values.square().mean().sqrt().item(),
    }


def _variance(values):
    return values.var(unbiased=False).item()


def _covariance(left, right):
    return ((left - left.mean()) * (right - right.mean())).mean().item()


def _partition_variance(values, first):
    """Population law of total variance, including between-class interaction."""
    if first is None:
        return {"status": "unavailable", "reason": "first-step identity unavailable"}
    weight = first.double().mean().item()
    first_values, later_values = values[first], values[~first]
    first_var = _variance(first_values) if first_values.numel() else 0.0
    later_var = _variance(later_values) if later_values.numel() else 0.0
    between = weight * (1 - weight) * (first_values.mean() - later_values.mean()).square().item() if 0 < weight < 1 else 0.0
    return {
        "status": "ok", "first_fraction": weight, "within_first": weight * first_var,
        "within_later": (1 - weight) * later_var, "between": between,
        "reconstructed_population_variance": weight * first_var + (1 - weight) * later_var + between,
        "population_variance": _variance(values),
    }


def _component_variance(task, prediction, penalty):
    components = {"task": task, "prediction": prediction, "penalty": penalty}
    result = {key: _variance(value) for key, value in components.items()}
    result.update(
        task_prediction_covariance=_covariance(task, prediction),
        task_penalty_covariance=_covariance(task, penalty),
        prediction_penalty_covariance=_covariance(prediction, penalty),
        total=_variance(task + prediction + penalty),
    )
    result["reconstructed_total"] = sum(result[key] for key in components) + 2 * sum(value for key, value in result.items() if key.endswith("covariance"))
    result["prediction_variance_over_total"] = result["prediction"] / result["total"] if result["total"] > 0 else None
    return result


def _optional_stats(rows, key):
    present = [_number(row[key], key) for row in rows if row.get(key) is not None]
    return {**_stats(present), "missing": len(rows) - len(present)}


def _first_mask(rows, infer_first):
    source = "is_first_step"
    if all(isinstance(row.get("is_first_step"), bool) for row in rows):
        flags = [row["is_first_step"] for row in rows]
    elif infer_first and all(isinstance(row.get("env_step_rank"), int) for row in rows):
        source = "inferred_from_env_step_rank"
        flags = [row["env_step_rank"] == 0 for row in rows]
    else:
        return None, "unavailable: missing explicit is_first_step; legacy rank inference requires --infer-first-from-rank"
    counts = defaultdict(int)
    for row, flag in zip(rows, flags, strict=True):
        counts[row["trajectory_id"]] += int(flag)
    if any(count != 1 for count in counts.values()):
        return None, "unavailable: expected exactly one first step per trajectory"
    return torch.tensor(flags, dtype=torch.bool), source


def _prepare(rows, infer_first=False):
    if not rows:
        raise ValueError("telemetry file is empty")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("every telemetry line must be a JSON object")
    identities = {(row.get("run_id"), row.get("seed"), row.get("update_id")) for row in rows}
    if len(identities) != 1:
        raise ValueError("one file must contain exactly one run/seed/update; split mixed updates before replay")
    active, inactive, empty_response = [], 0, 0
    for row in rows:
        marker = row.get("estimator_active_row", row.get("active_mask"))
        if marker is None or marker not in (True, False, 0.0, 1.0):
            raise ValueError("active_mask or estimator_active_row must identify padding explicitly")
        if not bool(marker):
            inactive += 1
            continue
        length = _number(row.get("response_length"), "response_length")
        if length < 0 or not length.is_integer():
            raise ValueError("response_length must be a nonnegative integer")
        if not length:
            empty_response += 1
            continue
        for key in ("group_id", "trajectory_id"):
            if not isinstance(row.get(key), (str, int)) or isinstance(row[key], bool):
                raise ValueError(f"{key} must be a string or integer")
        active.append(row)
    if not active:
        raise ValueError("no active rows with response tokens")
    arrays = {
        name: torch.tensor([_number(row.get(key), key) for row in active], dtype=torch.float64)
        for name, key in {"task": "task_score_sample", "prediction": "pred_component", "penalty": "penalty_component", "scores": "x_prenorm", "weights": "response_length"}.items()
    }
    reconstruction = arrays["task"] + arrays["prediction"] + arrays["penalty"]
    errors = (reconstruction - arrays["scores"]).abs()
    if not torch.allclose(reconstruction, arrays["scores"], rtol=2e-5, atol=2e-6):
        raise ValueError("x_prenorm does not reconstruct from task + pred_component + signed penalty_component")
    groups, trajectories = defaultdict(list), defaultdict(list)
    for i, row in enumerate(active):
        groups[row["group_id"]].append(i)
        trajectories[row["trajectory_id"]].append(i)
    for indices in trajectories.values():
        if len({active[i]["group_id"] for i in indices}) != 1:
            raise ValueError("a trajectory belongs to multiple task groups")
        if len({active[i]["task_score_sample"] for i in indices}) != 1:
            raise ValueError("task_score_sample is not constant within a trajectory; terminal q_i must be broadcast")
        successes = {active[i]["traj_success"] for i in indices if isinstance(active[i].get("traj_success"), bool)}
        if len(successes) > 1:
            raise ValueError("traj_success must be constant within a trajectory")
    lambda_values = {row.get("lambda") for row in active}
    prediction_weight = next(iter(lambda_values)) if len(lambda_values) == 1 else None
    if prediction_weight is not None:
        prediction_weight = _number(prediction_weight, "lambda")
        if prediction_weight < 0:
            raise ValueError("lambda must be nonnegative")
    first, first_source = _first_mask(active, infer_first)
    run_id, seed, update_id = next(iter(identities))
    warnings = []
    if prediction_weight is None:
        warnings.append("lambda is missing or inconsistent; GDPO replay is unavailable")
    if first is None or first_source.startswith("inferred"):
        warnings.append(first_source)
    logged_counts = [row.get("traj_n_samples") for row in active]
    if any(value is not None and value != len(trajectories[row["trajectory_id"]]) for row, value in zip(active, logged_counts, strict=True)):
        warnings.append("logged traj_n_samples differs from active row count; check padding and trajectory completeness")
    metadata = {
        "run_id": run_id, "seed": seed, "update_id": update_id, "input_rows": len(rows), "active_rows": len(active),
        "excluded_inactive_rows": inactive, "excluded_zero_response_rows": empty_response,
        "groups": len(groups), "trajectories": len(trajectories), "lambda": prediction_weight,
        "lambda_values": sorted(lambda_values, key=str), "component_reconstruction_max_error": errors.max().item(),
        "first_step_source": first_source, "warnings": warnings,
        "success_only_values": sorted({str(row.get("success_only", "unknown")) for row in active}),
        "loss_agg_mode_values": sorted({str(row.get("loss_agg_mode", "unknown")) for row in active}),
        "rms_weighting_values": sorted({str(row.get("rms_weighting", "unknown")) for row in active}),
    }
    return active, arrays, groups, trajectories, first, metadata


def _group_info(rows, arrays, groups):
    info = {}
    for identifier, indices in groups.items():
        trajectory_rows = defaultdict(list)
        for i in indices:
            trajectory_rows[rows[i]["trajectory_id"]].append(i)
        tasks = [arrays["task"][positions[0]].item() for positions in trajectory_rows.values()]
        successes = []
        for positions in trajectory_rows.values():
            recorded = [rows[i]["traj_success"] for i in positions if isinstance(rows[i].get("traj_success"), bool)]
            successes.append(recorded[0] if recorded else arrays["task"][positions[0]].item() > 0)
        category = "all_success" if all(successes) else "mixed" if any(successes) else "all_failure"
        info[identifier] = {
            "group_id": identifier, "category": category, "task_tied": len(set(tasks)) == 1,
            "rows": len(indices), "trajectories": len(trajectory_rows), "successful_trajectories": sum(successes),
        }
    return info


def _summary(indices, arrays, first, advantage, diagnostics, norm_by_std, epsilon):
    index = torch.tensor(indices, dtype=torch.long)
    values, weights = advantage[index], arrays["weights"][index]
    result = {
        "advantage": _stats(values),
        "response_token_weighted_advantage_rms": (values.square() * (weights / weights.sum())).sum().sqrt().item(),
        "normalization_score": _stats(diagnostics["normalization_score"][index]),
        "normalization_mean": _stats(diagnostics["normalization_mean"][index]),
        "normalization_std": _stats(diagnostics["normalization_std"][index]),
        "actual_denominator": _stats(diagnostics["normalization_std"][index] + epsilon if norm_by_std else torch.ones_like(values)),
        "epsilon_dominated_fraction": (diagnostics["normalization_std"][index] <= epsilon).double().mean().item() if norm_by_std else None,
    }
    if first is not None:
        result["first_advantage"], result["later_advantage"] = _stats(values[first[index]]), _stats(values[~first[index]])
    for channel in ("task", "prediction"):
        key = f"gdpo_{channel}_std"
        if key in diagnostics:
            result[f"{channel}_channel_denominator"] = _stats(diagnostics[key][index] + epsilon)
    return result


def replay_rows(rows, *, epsilons=None, infer_first=False, include_groups=False):
    """Analyze one complete update, retaining explicit unavailable variants."""
    rows, arrays, groups, trajectories, first, metadata = _prepare(rows, infer_first)
    info = _group_info(rows, arrays, groups)
    if epsilons is None:
        recorded_epsilons = {_number(row.get("epsilon"), "epsilon") for row in rows}
        if len(recorded_epsilons) != 1:
            raise ValueError("recorded epsilon is inconsistent; specify --epsilons explicitly")
        base = next(iter(recorded_epsilons))
        epsilons = [base / 100, base, base * 100]
    epsilons = list(dict.fromkeys(epsilons))
    if any(not math.isfinite(value) or value <= 0 for value in epsilons):
        raise ValueError("epsilons must be finite and positive")
    classes = {}
    for name in ("all_failure", "mixed", "all_success", "task_tied"):
        classes[name] = [i for group, indices in groups.items() if (info[group]["task_tied"] if name == "task_tied" else info[group]["category"] == name) for i in indices]
    source_groups = []
    for identifier, indices in groups.items():
        index = torch.tensor(indices)
        record = {**info[identifier], "component_population_variance": _component_variance(arrays["task"][index], arrays["prediction"][index], arrays["penalty"][index])}
        record["prediction_first_later_variance"] = _partition_variance(arrays["prediction"][index], first[index] if first is not None else None)
        source_groups.append(record)
    variance_classes = {}
    for name in classes:
        records = [row for row in source_groups if row["task_tied"]] if name == "task_tied" else [row for row in source_groups if row["category"] == name]
        terms = [row["component_population_variance"] for row in records]
        summed = {key: sum(term[key] for term in terms) for key in ("task", "prediction", "penalty", "task_prediction_covariance", "task_penalty_covariance", "prediction_penalty_covariance", "total", "reconstructed_total")}
        variance_classes[name] = {
            "group_count": len(records), "sum_group_population_variance_terms": summed,
            "ratio_of_summed_prediction_to_summed_total_variances": summed["prediction"] / summed["total"] if summed["total"] > 0 else None,
        }
    trajectory_prediction_sums = torch.stack([arrays["prediction"][indices].sum() for indices in trajectories.values()])
    raw_keys = ("pred_delta_raw", "pred_delta_after_boundary", "phi_prev", "phi_curr", "pred_accuracy", "pred_component", "penalty_component")
    observed_denominators, missing_denominators = [], 0
    for row in rows:
        if row.get("norm_by_std") is False:
            observed_denominators.append(1.0)
        elif row.get("norm_by_std") is True and row.get("group_std") is not None and row.get("epsilon") is not None:
            observed_denominators.append(_number(row["group_std"], "group_std") + _number(row["epsilon"], "epsilon"))
        else:
            missing_denominators += 1
    result = {
        "status": "ok", "source": metadata, "replay_dtype": "float64",
        "std_convention": "sample (ddof=1); variances/covariances below use population (ddof=0)",
        "classification": "Explicit traj_success when present, otherwise pure task_score_sample > 0; task_tied compares terminal q_i.",
        "observed": {
            "estimator_names": sorted({str(row.get("estimator", "unknown")) for row in rows}),
            "actual_denominator": {**_stats(observed_denominators), "missing": missing_denominators},
            "advantage": _optional_stats(rows, "advantage"),
            "raw_fields": {key: _optional_stats(rows, key) for key in raw_keys},
            "recorded_estimator_diagnostics": {key: _optional_stats(rows, key) for key in sorted({key for row in rows for key in row if key.startswith("estimator_")})},
            "trajectory_prediction_sum": _stats(trajectory_prediction_sums),
            "positive_prediction_rows": int((arrays["prediction"] > 0).sum()), "negative_prediction_rows": int((arrays["prediction"] < 0).sum()),
        },
        "variance_by_group_class": variance_classes, "replays": [],
    }
    if include_groups:
        result["source_groups"] = source_groups
    for dropfirst in (False, True):
        prediction = arrays["prediction"].clone()
        if dropfirst and first is not None:
            prediction[first] = 0
        scores = arrays["scores"] + prediction - arrays["prediction"]
        for epsilon in epsilons:
            for name, (estimator, normalize) in VARIANTS.items():
                record = {"estimator": name, "dropfirst": dropfirst, "epsilon": epsilon, "lambda": metadata["lambda"]}
                record["injected_prediction_sum"] = prediction.sum().item() if not dropfirst or first is not None else None
                if dropfirst and first is None:
                    record.update(status="unavailable", reason=metadata["first_step_source"])
                elif estimator == "gdpo" and metadata["lambda"] is None:
                    record.update(status="unavailable", reason="recorded lambda is required to undo prediction weighting")
                else:
                    try:
                        advantage, diagnostics = compute_rebuttal_advantages(
                            scores, arrays["task"], prediction,
                            [row["group_id"] for row in rows], [row["trajectory_id"] for row in rows],
                            estimator=estimator, norm_by_std=normalize, epsilon=epsilon,
                            prediction_weight=metadata["lambda"] if metadata["lambda"] is not None else 0,
                        )
                        record.update(status="ok", normalization_scope="batch" if estimator == "gdpo" else "task_group")
                        record["all_rows"] = _summary(list(range(len(rows))), arrays, first, advantage, diagnostics, normalize, epsilon)
                        record["by_group_class"] = {category: _summary(indices, arrays, first, advantage, diagnostics, normalize, epsilon) for category, indices in classes.items() if indices}
                        if include_groups:
                            record["groups"] = [{**info[group], **_summary(indices, arrays, first, advantage, diagnostics, normalize, epsilon)} for group, indices in groups.items()]
                    except ValueError as exc:
                        record.update(status="error", reason=str(exc))
                result["replays"].append(record)
    result["failed_replays"] = sum(row["status"] == "error" for row in result["replays"])
    result["unavailable_replays"] = sum(row["status"] == "unavailable" for row in result["replays"])
    if result["failed_replays"]:
        result["status"] = "partial"
    return result


def synthetic(*, group_size=4, lengths=None, lambdas=None, epsilons=None, penalty_scale=0.0):
    """Verify the boundary formula and zero-sum increments without training."""
    lengths = lengths or [50] * group_size
    lambdas = lambdas if lambdas is not None else [0.0, 1e-12, 1e-9, 1e-7, 1e-6, 1e-5, 1e-3, 0.1, 1.0]
    epsilons = epsilons if epsilons is not None else [1e-8, 1e-6, 1e-4]
    if group_size < 2 or len(lengths) != group_size or any(length < 2 for length in lengths):
        raise ValueError("use group_size >= 2 and one length >= 2 per trajectory")
    if any(not math.isfinite(value) or value < 0 for value in lambdas) or not math.isfinite(penalty_scale) or penalty_scale < 0:
        raise ValueError("lambdas and penalty_scale must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in epsilons):
        raise ValueError("epsilons must be finite and positive")
    trajectory_ids = [i for i, length in enumerate(lengths) for _ in range(length)]
    first = torch.tensor([step == 0 for length in lengths for step in range(length)])
    total = sum(lengths)
    task = torch.zeros(total, dtype=torch.float64)
    penalty = torch.tensor([-penalty_scale * (step % 2) for length in lengths for step in range(length)], dtype=torch.float64)
    records = []
    for pattern in ("saturated_boundary", "signed_zero_sum"):
        raw = torch.tensor([float(step == 0) - (float(step == 1) if pattern == "signed_zero_sum" else 0.0) for length in lengths for step in range(length)], dtype=torch.float64)
        for coefficient in lambdas:
            prediction = coefficient * raw
            expected_std = coefficient * math.sqrt(group_size * (total - group_size) / (total * (total - 1))) if pattern == "saturated_boundary" else None
            for epsilon in epsilons:
                for dropfirst in (False, True):
                    injected = prediction.clone()
                    if dropfirst:
                        injected[first] = 0
                    for name, (estimator, normalize) in VARIANTS.items():
                        advantages, diagnostics = compute_rebuttal_advantages(
                            task + injected + penalty, task, injected, ["g"] * total, trajectory_ids,
                            estimator=estimator, norm_by_std=normalize, epsilon=epsilon, prediction_weight=coefficient,
                        )
                        records.append({
                            "pattern": pattern, "lambda": coefficient, "epsilon": epsilon, "estimator": name, "dropfirst": dropfirst,
                            "boundary_prediction_only_expected_sample_std": expected_std,
                            "injected_prediction_sample_std": injected.std(unbiased=True).item(),
                            "actual_normalization_std": diagnostics["normalization_std"][0].item(),
                            "actual_denominator": diagnostics["normalization_std"][0].item() + epsilon if normalize else 1.0,
                            "first_advantage_mean": advantages[first].mean().item(), "later_advantage_mean": advantages[~first].mean().item(),
                            "advantage_rms": advantages.square().mean().sqrt().item(),
                            "component_population_variance": _component_variance(task, injected, penalty),
                        })
    return {
        "mode": "synthetic", "group_size": group_size, "lengths": lengths, "penalty_scale": penalty_scale,
        "formula_scope": "Expected boundary std is for one lambda per trajectory and all other prediction entries zero, before dropfirst; it excludes penalties.",
        "limitations": LIMITATIONS, "records": records,
    }


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
    return rows


def _input_files(paths):
    found = set()
    for value in paths:
        path = Path(value)
        candidates = sorted(path.rglob("*.jsonl*")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_dir() or not (candidate.name.endswith(".jsonl") or candidate.name.endswith(".jsonl.gz")):
                continue
            resolved = candidate.resolve()
            if resolved not in found:
                found.add(resolved)
                yield resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    simulated = subcommands.add_parser("synthetic", help="M01: G=4,T=50 boundary/zero-sum lambda and epsilon sweeps")
    simulated.add_argument("--group-size", type=int, default=4)
    simulated.add_argument("--lengths", nargs="+", type=int, help="One trajectory length per group member; default 50 each")
    simulated.add_argument("--lambdas", nargs="+", type=float)
    simulated.add_argument("--penalty-scale", type=float, default=0.0, help="Alternate zero / negative penalty to inspect signed covariance")
    replay = subcommands.add_parser("replay", help="M02: replay complete saved updates without training")
    replay.add_argument("paths", nargs="+", help="JSONL, JSONL.gz, or directories recursively containing one file per update")
    replay.add_argument("--infer-first-from-rank", action="store_true", help="Explicitly permit legacy env_step_rank==0 as first-step identity")
    replay.add_argument("--include-groups", action="store_true", help="Include per-group source decomposition and all estimator statistics")
    for command in (simulated, replay):
        command.add_argument("--epsilons", nargs="+", type=float, help="Positive additive floors; replay default is recorded epsilon times [0.01,1,100]")
        command.add_argument("--output", type=Path, help="JSON destination; default stdout")
    args = parser.parse_args(argv)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") if args.output else nullcontext(sys.stdout) as output:
            if args.mode == "synthetic":
                result = synthetic(group_size=args.group_size, lengths=args.lengths, lambdas=args.lambdas, epsilons=args.epsilons, penalty_scale=args.penalty_scale)
                json.dump(result, output, allow_nan=False, indent=2)
                output.write("\n")
                return 0
            output.write('{"mode":"replay","limitations":')
            json.dump(LIMITATIONS, output)
            output.write(',"files":[')
            count, failures = 0, 0
            for path in _input_files(args.paths):
                if count:
                    output.write(",\n")
                try:
                    result = replay_rows(read_rows(path), epsilons=args.epsilons, infer_first=args.infer_first_from_rank, include_groups=args.include_groups)
                    result["path"] = str(path)
                    serialized = json.dumps(result, allow_nan=False)
                except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
                    result = {"status": "error", "reason": str(exc)}
                    result["path"] = str(path)
                    serialized = json.dumps(result, allow_nan=False)
                failures += result["status"] in {"error", "partial"}
                output.write(serialized)
                output.flush()
                count += 1
            output.write('],"summary":')
            json.dump({"file_count": count, "failed_files": failures, "status": "error" if failures or not count else "ok"}, output)
            output.write("}\n")
            return 1 if failures or not count else 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
