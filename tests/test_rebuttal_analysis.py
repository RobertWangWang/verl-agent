"""Tests for frozen-rollout analysis, using the on-disk telemetry schema."""

import copy
import gzip
import json
import math

import pytest

from scripts.rebuttal_analysis import main, read_rows, replay_rows, synthetic


def telemetry(tasks=(0.0, 0.0, 1.0, 0.0, 1.0, 1.0), lengths=None):
    """Three two-trajectory groups: all-fail, mixed, all-success."""
    lengths = lengths or [3] * len(tasks)
    rows = []
    for trajectory, (task, length) in enumerate(zip(tasks, lengths, strict=True)):
        for step in range(length):
            prediction = 0.1 if step == 0 else 0.0
            rows.append({
                "run_id": "fixture", "seed": 7, "update_id": 50,
                "group_id": f"g{trajectory // 2}", "trajectory_id": f"t{trajectory}",
                "env_step_rank": step, "env_step": step, "is_first_step": step == 0,
                "active_mask": True, "response_length": 2 + step,
                "task_score_sample": task, "pred_component": prediction,
                "penalty_component": 0.0, "x_prenorm": task + prediction,
                "pred_delta_raw": prediction / 0.1, "pred_delta_after_boundary": prediction / 0.1,
                "phi_prev": 0.0 if step == 0 else 1.0, "phi_curr": 1.0,
                "lambda": 0.1, "epsilon": 1e-6, "norm_by_std": True,
                "group_std": 0.5, "group_mean": 0.0, "advantage": 0.25,
                "estimator": "step", "traj_success": task > 0,
                "traj_n_samples": length,
            })
    return rows


def variant(result, estimator, dropfirst=False, epsilon=1e-6):
    return next(row for row in result["replays"] if row["estimator"] == estimator and row["dropfirst"] == dropfirst and row["epsilon"] == epsilon)


def test_replay_classification_denominators_and_trajectory_q_once():
    result = replay_rows(telemetry(), epsilons=[1e-6], include_groups=True)
    assert result["status"] == "ok"
    assert result["source"]["lambda"] == 0.1
    assert len(result["replays"]) == 8
    counts = result["variance_by_group_class"]
    assert [counts[key]["group_count"] for key in ("all_failure", "mixed", "all_success", "task_tied")] == [1, 1, 1, 2]
    assert result["observed"]["actual_denominator"]["mean"] == pytest.approx(0.500001)
    trajectory = variant(result, "trajectory")
    assert trajectory["by_group_class"]["all_failure"]["normalization_score"]["mean"] == pytest.approx(0.1)
    assert trajectory["by_group_class"]["all_success"]["normalization_score"]["mean"] == pytest.approx(1.1)
    assert trajectory["by_group_class"]["all_failure"]["advantage"]["rms"] == 0
    assert trajectory["by_group_class"]["mixed"]["normalization_std"]["mean"] == pytest.approx(math.sqrt(0.5))
    step = variant(result, "step")
    assert step["by_group_class"]["all_failure"]["advantage"]["rms"] > 0
    assert variant(result, "mean_only")["all_rows"]["actual_denominator"]["mean"] == 1
    gdpo = variant(result, "gdpo")
    assert gdpo["normalization_scope"] == "batch"
    assert "prediction_channel_denominator" in gdpo["all_rows"]
    assert gdpo["all_rows"]["actual_denominator"]["min"] == gdpo["all_rows"]["actual_denominator"]["max"]


def test_dropfirst_removes_only_injected_prediction_and_keeps_task():
    rows = telemetry()
    original = copy.deepcopy(rows)
    result = replay_rows(rows, epsilons=[1e-6])
    dropped = variant(result, "step", True)
    assert dropped["by_group_class"]["all_failure"]["advantage"]["rms"] == 0
    assert dropped["by_group_class"]["all_success"]["advantage"]["rms"] == 0
    assert dropped["by_group_class"]["mixed"]["advantage"]["rms"] > 0
    assert rows == original
    assert result["observed"]["raw_fields"]["pred_delta_raw"]["max"] == 1


def test_first_later_variance_includes_between_class_term():
    result = replay_rows(telemetry(tasks=(0, 0), lengths=[2, 5]), epsilons=[1e-6], include_groups=True)
    decomposition = result["source_groups"][0]["prediction_first_later_variance"]
    assert decomposition["first_fraction"] == pytest.approx(2 / 7)
    assert decomposition["within_first"] == 0
    assert decomposition["within_later"] == 0
    assert decomposition["between"] > 0
    assert decomposition["reconstructed_population_variance"] == pytest.approx(decomposition["population_variance"])


def test_signed_covariances_reconstruct_and_are_not_clipped():
    rows = telemetry(tasks=(0, 0))
    for row in rows:
        row["penalty_component"] = -0.5 * row["pred_component"]
        row["x_prenorm"] += row["penalty_component"]
    result = replay_rows(rows, epsilons=[1e-6], include_groups=True)
    decomposition = result["source_groups"][0]["component_population_variance"]
    assert decomposition["prediction_penalty_covariance"] < 0
    assert decomposition["prediction_variance_over_total"] == pytest.approx(4)
    assert decomposition["reconstructed_total"] == pytest.approx(decomposition["total"])


def test_missing_boundary_metadata_is_explicit_and_opt_in_rank_works():
    rows = telemetry()
    for row in rows:
        row.pop("is_first_step")
        row.pop("phi_prev")
    result = replay_rows(rows, epsilons=[1e-6])
    assert variant(result, "step", True)["status"] == "unavailable"
    assert variant(result, "step")["status"] == "ok"
    assert result["unavailable_replays"] == 4
    assert result["observed"]["raw_fields"]["phi_prev"]["missing"] == len(rows)
    inferred = replay_rows(rows, epsilons=[1e-6], infer_first=True)
    assert inferred["source"]["first_step_source"] == "inferred_from_env_step_rank"
    assert variant(inferred, "step", True)["status"] == "ok"


def test_missing_or_zero_lambda_does_not_silently_remove_prediction():
    rows = telemetry()
    for row in rows:
        row["lambda"] = None
    result = replay_rows(rows, epsilons=[1e-6])
    assert variant(result, "gdpo")["status"] == "unavailable"
    assert variant(result, "step")["status"] == "ok"
    for row in rows:
        row["lambda"] = 0.0
    result = replay_rows(rows, epsilons=[1e-6])
    assert result["status"] == "partial"
    assert variant(result, "gdpo")["status"] == "error"
    assert "zero prediction_weight" in variant(result, "gdpo")["reason"]


def test_active_selection_and_token_weighted_rms():
    rows = telemetry(tasks=(0, 0))
    inactive = copy.deepcopy(rows[0])
    inactive.update(active_mask=False, task_score_sample=10000, trajectory_id="padding")
    zero = copy.deepcopy(rows[0])
    zero.update(response_length=0, trajectory_id="zero_tokens")
    result = replay_rows(rows + [inactive, zero], epsilons=[1e-6])
    assert result["source"]["active_rows"] == 6
    assert result["source"]["excluded_inactive_rows"] == 1
    assert result["source"]["excluded_zero_response_rows"] == 1
    summary = variant(result, "step")["all_rows"]
    assert summary["response_token_weighted_advantage_rms"] < summary["advantage"]["rms"]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[0].update(update_id=49), "exactly one"),
    (lambda rows: rows[0].update(x_prenorm=999), "reconstruct"),
    (lambda rows: rows[0].update(active_mask=None), "padding explicitly"),
    (lambda rows: rows[0].update(response_length=-1), "nonnegative integer"),
    (lambda rows: rows[0].update(pred_component=float("nan")), "finite number"),
    (lambda rows: rows[0].update(trajectory_id=rows[-1]["trajectory_id"]), "multiple task groups"),
    (lambda rows: rows[0].update(task_score_sample=0.2, x_prenorm=0.3), "constant within"),
])
def test_inconsistent_input_is_rejected(mutation, match):
    rows = telemetry()
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        replay_rows(rows, epsilons=[1e-6])


def test_row_shuffle_does_not_change_replay():
    rows = telemetry(lengths=[2, 3, 4, 5, 6, 7])
    expected = replay_rows(rows, epsilons=[1e-6])
    actual = replay_rows(list(reversed(rows)), epsilons=[1e-6])
    for before, after in zip(expected["replays"], actual["replays"], strict=True):
        assert before["all_rows"]["advantage"]["rms"] == pytest.approx(after["all_rows"]["advantage"]["rms"])
        assert before["all_rows"]["actual_denominator"]["mean"] == pytest.approx(after["all_rows"]["actual_denominator"]["mean"])


def test_synthetic_formula_epsilon_floor_and_zero_sum():
    result = synthetic(lambdas=[0.0, 1e-12, 0.1], epsilons=[1e-6])
    def select(pattern, coefficient, estimator="step", dropfirst=False):
        return next(row for row in result["records"] if row["pattern"] == pattern and row["lambda"] == coefficient and row["estimator"] == estimator and row["dropfirst"] == dropfirst)
    boundary = select("saturated_boundary", 0.1)
    assert boundary["actual_normalization_std"] == pytest.approx(0.014035131799278395)
    assert boundary["actual_normalization_std"] == pytest.approx(boundary["boundary_prediction_only_expected_sample_std"])
    assert boundary["first_advantage_mean"] == pytest.approx(6.98198060558524)
    assert select("saturated_boundary", 1e-12)["first_advantage_mean"] < 1e-6
    assert select("saturated_boundary", 0)["advantage_rms"] == 0
    assert select("saturated_boundary", 0.1, dropfirst=True)["advantage_rms"] == 0
    assert select("signed_zero_sum", 0.1)["advantage_rms"] > 0
    assert select("signed_zero_sum", 0.1, "trajectory")["advantage_rms"] == 0


def test_synthetic_unequal_lengths_and_penalty_covariance():
    result = synthetic(group_size=4, lengths=[2, 3, 5, 8], lambdas=[0.1], epsilons=[1e-6], penalty_scale=0.2)
    row = result["records"][0]
    expected = 0.1 * math.sqrt(4 * (18 - 4) / (18 * 17))
    assert row["injected_prediction_sample_std"] == pytest.approx(expected)
    assert row["actual_normalization_std"] != pytest.approx(expected)
    assert row["component_population_variance"]["prediction_penalty_covariance"] > 0


def test_cli_streams_gzip_and_reports_bad_file_without_losing_good_file(tmp_path):
    good = tmp_path / "update_0050.jsonl.gz"
    with gzip.open(good, "wt", encoding="utf-8") as stream:
        for row in telemetry():
            stream.write(json.dumps(row) + "\n")
    assert len(read_rows(good)) == 18
    bad = tmp_path / "update_0051.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    output = tmp_path / "result.json"
    status = main(["replay", str(tmp_path), "--epsilons", "0.000001", "--output", str(output)])
    result = json.loads(output.read_text())
    assert status == 1
    assert result["summary"] == {"file_count": 2, "failed_files": 1, "status": "error"}
    assert result["files"][0]["status"] == "ok"
    assert result["files"][1]["status"] == "error"
    assert "line 1" in result["files"][1]["reason"]


def test_cli_synthetic_and_empty_input(tmp_path):
    output = tmp_path / "synthetic.json"
    assert main(["synthetic", "--lambdas", "0.1", "--epsilons", "1e-6", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())["records"]) == 16
    assert main(["replay", str(tmp_path / "missing.jsonl"), "--output", str(tmp_path / "failed.json")]) == 1


def test_missing_recorded_denominator_is_not_imputed():
    rows = telemetry()
    for row in rows:
        row["norm_by_std"] = None
    result = replay_rows(rows, epsilons=[1e-6])
    assert result["observed"]["actual_denominator"]["count"] == 0
    assert result["observed"]["actual_denominator"]["missing"] == len(rows)
