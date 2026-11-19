"""CLI smoke tests for uplift-ml benchmark and CSV execution."""

from pathlib import Path
import subprocess
import sys

from main import main
from src.utils import generate_synthetic_uplift_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_cli_default_benchmark_smoke_run(capsys):
    exit_code = main(["--demo", "--n-samples", "600"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Causal Uplift Meta-Learner Benchmark" in captured.out
    assert "S-learner" in captured.out
    assert "T-learner" in captured.out
    assert "X-learner" in captured.out
    assert "persuadables" in captured.out
    assert "sleeping_dogs" in captured.out
    assert "Uplift-by-Decile" in captured.out


def test_cli_csv_mode_smoke_run(tmp_path):
    df = generate_synthetic_uplift_data(n_samples=500, random_state=7).rename(
        columns={"outcome": "converted", "treatment": "campaign"}
    )
    csv_path = tmp_path / "campaign.csv"
    df.to_csv(csv_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--data-path",
            str(csv_path),
            "--model-type",
            "X-learner",
            "--outcome-var",
            "converted",
            "--treatment-var",
            "campaign",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Uplift Report (X-learner)" in proc.stdout
    assert "Qini=" in proc.stdout
    assert "AUUC=" in proc.stdout


def _report(qini: float):
    from src.evaluation import EvaluationReport

    return EvaluationReport(
        n_samples=10,
        n_treated=5,
        n_control=5,
        control_mean=0.0,
        treated_mean=0.1,
        observed_ate=0.1,
        predicted_ate=0.1,
        qini_coefficient=qini,
        qini_auc=0.0,
        random_qini_auc=0.0,
        auuc=0.0,
        random_auuc=0.0,
        auuc_gain=0.0,
    )


def test_exit_code_no_threshold_always_succeeds():
    from main import exit_code_for

    assert exit_code_for([_report(-1.0)], None) == 0


def test_exit_code_gates_on_min_qini():
    from main import exit_code_for

    assert exit_code_for([_report(0.1)], 0.5) == 1
    assert exit_code_for([_report(0.5)], 0.5) == 0
    assert exit_code_for([_report(0.9), _report(0.1)], 0.5) == 1
    assert exit_code_for([], 0.5) == 0


def test_exit_code_rejects_non_finite_threshold():
    import math

    import pytest

    from main import exit_code_for

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="min_qini must be finite"):
            exit_code_for([_report(1.0)], bad)


def test_cli_min_qini_gates_the_run():
    assert main(["--n-samples", "400", "--min-qini", "-1.0"]) == 0
    assert main(["--n-samples", "400", "--min-qini", "10.0"]) == 1
