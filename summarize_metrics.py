#!/usr/bin/env python3
"""Print metric summaries from an output_dir produced by run_compare.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "deletion_auc",
    "insertion_auc",
    "confidence_drop",
    "confidence_increase",
    "road_combined",
    "runtime_sec",
]

LOWER_IS_BETTER = {"deletion_auc", "runtime_sec"}
HIGHER_IS_BETTER = {
    "insertion_auc",
    "road_combined",
    "confidence_increase",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize metrics.csv from a CAM comparison output directory."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory containing metrics.csv, e.g. outputs_cub_resnet50",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="method",
        help=(
            "Comma-separated grouping columns. "
            "Default: method. Examples: method,target_mode or model_mode,method"
        ),
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="",
        help="Optional metric used to sort the summary table, e.g. insertion_auc.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows with non-empty error field when computing summaries.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to save the summary table as CSV.",
    )
    return parser.parse_args()


def load_metrics(output_dir: Path) -> pd.DataFrame:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Cannot find metrics.csv at: {metrics_path}")

    df = pd.read_csv(metrics_path)

    if df.empty:
        raise ValueError(f"metrics.csv is empty: {metrics_path}")

    for col in METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "error" not in df.columns:
        df["error"] = ""

    df["error"] = df["error"].fillna("").astype(str)

    return df


def print_header(title: str):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_basic_info(df: pd.DataFrame, valid_df: pd.DataFrame):
    print_header("Basic information")

    print(f"Total rows              : {len(df)}")
    print(f"Valid rows for summary  : {len(valid_df)}")
    print(f"Failed rows             : {len(df) - len(valid_df)}")

    if "image_id" in df.columns:
        print(f"Unique images           : {df['image_id'].nunique()}")

    if "method" in df.columns:
        methods = sorted(str(x) for x in df["method"].dropna().unique())
        print(f"Methods                 : {', '.join(methods)}")

    if "model_mode" in df.columns:
        modes = sorted(str(x) for x in df["model_mode"].dropna().unique())
        print(f"Model modes             : {', '.join(modes)}")

    if "target_mode" in df.columns:
        target_modes = sorted(str(x) for x in df["target_mode"].dropna().unique())
        print(f"Target modes            : {', '.join(target_modes)}")


def print_failure_summary(df: pd.DataFrame):
    if "error" not in df.columns:
        return

    failed = df[df["error"].str.len() > 0]
    if failed.empty:
        print_header("Failure summary")
        print("No failed rows.")
        return

    print_header("Failure summary")

    group_cols = [c for c in ["model_mode", "method"] if c in failed.columns]
    if group_cols:
        failure_counts = (
            failed.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name="failed_count")
            .sort_values("failed_count", ascending=False)
        )
        print(failure_counts.to_string(index=False))

    print()
    print("Example errors:")
    example_cols = [c for c in ["image_id", "method", "error"] if c in failed.columns]
    print(failed[example_cols].head(10).to_string(index=False))


def build_summary(valid_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    available_metrics = [c for c in METRIC_COLUMNS if c in valid_df.columns]

    if not available_metrics:
        raise ValueError("No metric columns found in metrics.csv.")

    agg_spec = {}
    for metric in available_metrics:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
        agg_spec[f"{metric}_median"] = (metric, "median")
        agg_spec[f"{metric}_min"] = (metric, "min")
        agg_spec[f"{metric}_max"] = (metric, "max")
        agg_spec[f"{metric}_count"] = (metric, "count")

    summary = (
        valid_df.groupby(group_cols, dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )

    return summary


def print_compact_summary(summary: pd.DataFrame, group_cols: list[str], sort_by: str):
    print_header("Compact summary")

    preferred_cols = list(group_cols)

    for metric in METRIC_COLUMNS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        if mean_col in summary.columns:
            preferred_cols.extend([mean_col, std_col, count_col])

    compact = summary[preferred_cols].copy()

    if sort_by:
        sort_col = f"{sort_by}_mean"
        if sort_col not in compact.columns:
            raise ValueError(
                f"--sort-by {sort_by!r} is invalid. "
                f"Available metrics: {', '.join(METRIC_COLUMNS)}"
            )
        ascending = sort_by in LOWER_IS_BETTER
        compact = compact.sort_values(sort_col, ascending=ascending)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        "{:.6f}".format,
    ):
        print(compact.to_string(index=False))


def print_ranking(summary: pd.DataFrame, group_cols: list[str]):
    print_header("Metric rankings by mean")

    label_col = "__group_label__"
    ranked = summary.copy()
    ranked[label_col] = ranked[group_cols].astype(str).agg(" | ".join, axis=1)

    for metric in METRIC_COLUMNS:
        mean_col = f"{metric}_mean"
        if mean_col not in ranked.columns:
            continue

        tmp = ranked[[label_col, mean_col]].dropna()
        if tmp.empty:
            continue

        if metric in LOWER_IS_BETTER:
            tmp = tmp.sort_values(mean_col, ascending=True)
            direction = "lower is usually better"
        elif metric in HIGHER_IS_BETTER:
            tmp = tmp.sort_values(mean_col, ascending=False)
            direction = "higher is usually better"
        else:
            tmp = tmp.sort_values(mean_col, ascending=False)
            direction = "interpret with care"

        print()
        print(f"{metric} ({direction})")
        print("-" * 88)
        print(tmp.head(10).to_string(index=False))


def print_overall_metric_stats(valid_df: pd.DataFrame):
    print_header("Overall metric statistics")

    available_metrics = [c for c in METRIC_COLUMNS if c in valid_df.columns]

    rows = []
    for metric in available_metrics:
        values = valid_df[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "count": int(values.count()),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }
        )

    overall = pd.DataFrame(rows)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        160,
        "display.float_format",
        "{:.6f}".format,
    ):
        print(overall.to_string(index=False))


def main():
    args = parse_args()

    df = load_metrics(args.output_dir)

    if args.include_failed:
        valid_df = df.copy()
    else:
        valid_df = df[df["error"].str.len() == 0].copy()

    if valid_df.empty:
        print_basic_info(df, valid_df)
        print_failure_summary(df)
        raise ValueError(
            "No valid rows available for summary. "
            "Use --include-failed to include failed rows, or inspect the error column."
        )

    group_cols = [x.strip() for x in args.group_by.split(",") if x.strip()]
    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        raise ValueError(
            f"Grouping columns not found: {missing_group_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    print_basic_info(df, valid_df)
    print_failure_summary(df)
    print_overall_metric_stats(valid_df)

    summary = build_summary(valid_df, group_cols)
    print_compact_summary(summary, group_cols, args.sort_by)
    print_ranking(summary, group_cols)

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.save_csv, index=False)
        print()
        print(f"Saved summary CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()
