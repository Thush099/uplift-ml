"""CLI entry point for causal uplift modeling and S/T/X-learner benchmarking."""

import argparse
import math
from typing import Dict, Optional

import pandas as pd

from config import Config, VALID_MODEL_TYPES
from src.evaluation import EvaluationReport
from src.pipeline import PipelineResult, UpliftPipeline, run_benchmark
from src.utils import load_data


def _format_benchmark_table(results: Dict[str, PipelineResult]) -> str:
    lines = [
        "=== Causal Uplift Meta-Learner Benchmark (Synthetic Heterogeneous CATE) ===",
        (
            f"{'Model':<12} {'Pred ATE':>10} {'True ATE':>10} {'Obs ATE':>10} "
            f"{'PEHE':>8} {'Qini':>8} {'AUUC':>8} {'Top Decile':>12} {'Bot Decile':>12}"
        ),
        "-" * 96,
    ]
    for model_name, res in results.items():
        r = res.report
        true_ate_str = f"{r.true_ate:+.4f}" if r.true_ate is not None else "n/a"
        pehe_str = f"{r.pehe:.4f}" if r.pehe is not None else "n/a"
        top_u = r.uplift_by_decile[0].observed_uplift
        bot_u = r.uplift_by_decile[-1].observed_uplift
        top_u_str = f"{top_u:+.4f}" if top_u is not None else "n/a"
        bot_u_str = f"{bot_u:+.4f}" if bot_u is not None else "n/a"
        lines.append(
            f"{model_name:<12} {r.predicted_ate:>+10.4f} {true_ate_str:>10} {r.observed_ate:>+10.4f} "
            f"{pehe_str:>8} {r.qini_coefficient:>8.4f} {r.auuc:>8.4f} {top_u_str:>12} {bot_u_str:>12}"
        )
    return "\n".join(lines)


def _format_subgroup_summary(df: pd.DataFrame) -> str:
    lines = ["", "Synthetic Subgroup Ground Truth:"]
    summary = (
        df.groupby("subgroup", sort=True)
        .agg(
            count=("true_cate", "size"),
            mean_true_cate=("true_cate", "mean"),
        )
        .sort_values("mean_true_cate", ascending=False)
    )
    for name, row in summary.iterrows():
        lines.append(f"  - {name:<14} n={int(row['count']):>4} | true_cate={row['mean_true_cate']:+.4f}")
    return "\n".join(lines)


def _format_deciles(model_name: str, report: EvaluationReport) -> str:
    lines = [
        "",
        f"Uplift-by-Decile ({model_name}, test n={report.n_samples}):",
        f"{'Decile':>6} {'N':>6} {'Treated':>8} {'Control':>8} {'Pred Uplift':>12} {'Obs Uplift':>12}",
        "-" * 58,
    ]
    for d in report.uplift_by_decile:
        observed = f"{d.observed_uplift:+.4f}" if d.observed_uplift is not None else "n/a"
        lines.append(
            f"{d.decile:>6} {d.n_samples:>6} {d.n_treated:>8} {d.n_control:>8} "
            f"{d.mean_predicted_uplift:>+12.4f} {observed:>12}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser supporting both benchmark demo and CSV inputs."""
    parser = argparse.ArgumentParser(
        description="Causal Uplift Modeling Toolkit (S-Learner, T-Learner, X-Learner)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run synthetic heterogeneous CATE benchmark comparing S-, T-, and X-learner.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3000,
        help="Number of synthetic samples for benchmark mode (default: 3000).",
    )
    parser.add_argument(
        "--data-path",
        "--data_path",
        dest="data_path",
        default=None,
        help="Path to input CSV dataset (omit for synthetic benchmark).",
    )
    parser.add_argument(
        "--model-type",
        "--model_type",
        dest="model_type",
        default="X-learner",
        choices=[*VALID_MODEL_TYPES, "all"],
        help="Meta-learner type: S-learner, T-learner, X-learner, or all (default: X-learner).",
    )
    parser.add_argument(
        "--outcome-var",
        "--outcome_var",
        dest="outcome_var",
        default="outcome",
        help="Name of outcome column in CSV input (default: outcome).",
    )
    parser.add_argument(
        "--treatment-var",
        "--treatment_var",
        dest="treatment_var",
        default="treatment",
        help="Name of binary treatment column in CSV input (default: treatment).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--min-qini",
        type=float,
        default=None,
        help=(
            "Exit non-zero when any evaluated model's Qini coefficient is below "
            "this threshold, so the benchmark can gate a CI job."
        ),
    )
    return parser


def exit_code_for(reports: list[EvaluationReport], min_qini: Optional[float]) -> int:
    """Map evaluation reports to a process exit code under a Qini gate.

    Returns 0 when no threshold is given or every model meets it, and 1 when any
    model's Qini coefficient is below the threshold. The threshold must be a
    finite number; anything else raises ``ValueError`` so a typo cannot silently
    disable gating. Qini coefficients are normalized against an oracle ordering,
    so unlike a probability they may legitimately be negative.
    """
    if min_qini is None:
        return 0
    if not math.isfinite(min_qini):
        raise ValueError(f"min_qini must be finite; got {min_qini!r}")
    return 1 if any(r.qini_coefficient < min_qini for r in reports) else 0


def main(argv: Optional[list[str]] = None) -> int:
    """Execute synthetic benchmark or CSV uplift evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.min_qini is not None and not math.isfinite(args.min_qini):
        parser.error(f"--min-qini must be finite; got {args.min_qini!r}")

    if args.demo or args.data_path is None:
        config = Config(model_type="all", random_state=args.seed)
        df, results = run_benchmark(n_samples=args.n_samples, config=config)
        print(_format_benchmark_table(results))
        print(_format_subgroup_summary(df))
        print(_format_deciles("X-learner", results["X-learner"].report))
        return exit_code_for([res.report for res in results.values()], args.min_qini)

    try:
        raw_df = load_data(args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    true_cate = raw_df["true_cate"].to_numpy(dtype=float) if "true_cate" in raw_df.columns else None

    if args.model_type == "all":
        config = Config(
            data_path=args.data_path,
            model_type="all",
            outcome_var=args.outcome_var,
            treatment_var=args.treatment_var,
            random_state=args.seed,
        )
        pipeline = UpliftPipeline(config=config)
        results = {
            m: pipeline.run_on_dataframe(raw_df, model_type=m, true_cate=true_cate)
            for m in VALID_MODEL_TYPES
        }
        print(_format_benchmark_table(results))
        print(_format_deciles("X-learner", results["X-learner"].report))
        return exit_code_for([res.report for res in results.values()], args.min_qini)

    config = Config(
        data_path=args.data_path,
        model_type=args.model_type,
        outcome_var=args.outcome_var,
        treatment_var=args.treatment_var,
        random_state=args.seed,
    )
    pipeline = UpliftPipeline(config=config)
    res = pipeline.run_on_dataframe(raw_df, model_type=args.model_type, true_cate=true_cate)
    r = res.report
    print(f"=== Uplift Report ({args.model_type}) ===")
    print(
        f"Test n={r.n_samples} (treated={r.n_treated}, control={r.n_control}) | "
        f"Control Mean={r.control_mean:.4f} | Treated Mean={r.treated_mean:.4f}"
    )
    print(
        f"Predicted ATE={r.predicted_ate:+.4f} | Observed ATE={r.observed_ate:+.4f} | "
        f"Qini={r.qini_coefficient:.4f} | AUUC={r.auuc:.4f} (gain={r.auuc_gain:+.4f})"
    )
    if r.pehe is not None:
        print(f"PEHE (RMSE vs true CATE)={r.pehe:.4f}")
    print(_format_deciles(args.model_type, r))
    return exit_code_for([r], args.min_qini)


if __name__ == "__main__":
    raise SystemExit(main())
