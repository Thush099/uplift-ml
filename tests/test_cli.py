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
