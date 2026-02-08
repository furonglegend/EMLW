#!/usr/bin/env python3
"""
scripts/analyze_ablation.py

Simple analysis utilities for ablation experiment CSV outputs.
- Loads CSV files from a directory
- Aggregates metric traces and produces plots + a summary CSV

Expected CSV format (one of the accepted schemas):
- experiment, ablation_tag, round/epoch, metric_name, metric_value
or
- experiment, ablation_tag, epoch, acc, loss, precision, recall, ...
"""

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "analyze_ablation.log", mode="a")],
    )


def load_experiment_csvs(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    df_list = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df["source_file"] = p.name
            df_list.append(df)
        except Exception as e:
            logging.warning("Failed to load %s: %s", p, e)
    combined = pd.concat(df_list, ignore_index=True)
    logging.info("Loaded %d rows from %d csv files.", combined.shape[0], len(df_list))
    return combined


def standardize_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide metric table into long format if needed.
    If the DataFrame already has columns ['metric_name', 'metric_value'], return it unchanged.
    """
    cols = set(df.columns)
    if {"metric_name", "metric_value"}.issubset(cols):
        return df
    # Detect numeric metric columns beyond (experiment, ablation_tag, epoch)
    id_vars = [c for c in ["experiment", "ablation_tag", "epoch", "round", "source_file"] if c in cols]
    metric_cols = [c for c in df.columns if c not in id_vars and c not in ["source_file"]]
    if not metric_cols:
        raise ValueError("No metric columns detected to analyze.")
    long = df.melt(id_vars=id_vars, value_vars=metric_cols, var_name="metric_name", value_name="metric_value")
    return long


def plot_metric_traces(df_long: pd.DataFrame, metric: str, output_dir: Path):
    """
    Plot metric traces per ablation_tag and experiment for the given metric_name.
    Saves a PNG file per metric.
    """
    plot_path = output_dir / f"{metric}_traces.png"
    plt.figure(figsize=(10, 6))
    grouped = df_long[df_long["metric_name"] == metric].groupby(["ablation_tag", "epoch"])
    # compute mean and std across sources/experiments
    agg = grouped["metric_value"].agg(["mean", "std"]).reset_index()
    for tag, sub in agg.groupby("ablation_tag"):
        plt.plot(sub["epoch"], sub["mean"], label=str(tag))
        plt.fill_between(sub["epoch"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.2)
    plt.xlabel("epoch/round")
    plt.ylabel(metric)
    plt.title(f"{metric} traces by ablation tag")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logging.info("Saved metric trace plot to %s", plot_path)


def summarize_metrics(df_long: pd.DataFrame, output_dir: Path) -> Path:
    """
    Produce a summary CSV with final metrics (by taking last epoch).
    Returns path to saved CSV.
    """
    # Use epoch column if present, otherwise try 'round'
    epoch_col = "epoch" if "epoch" in df_long.columns else ("round" if "round" in df_long.columns else None)
    if epoch_col is None:
        # fallback: take mean across whatever is available
        summary = df_long.groupby(["experiment", "ablation_tag", "metric_name"])["metric_value"].agg(["mean", "std", "count"]).reset_index()
    else:
        last_epoch_df = df_long.sort_values(epoch_col).groupby(["experiment", "ablation_tag", "metric_name"]).tail(1)
        summary = last_epoch_df.groupby(["experiment", "ablation_tag", "metric_name"])["metric_value"].agg(["mean", "std", "count"]).reset_index()
    out_csv = output_dir / "ablation_summary.csv"
    summary.to_csv(out_csv, index=False)
    logging.info("Saved ablation summary to %s", out_csv)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation experiment CSV outputs.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with CSV experiment outputs.")
    parser.add_argument("--output-dir", type=str, default="results/ablation_analysis", help="Output folder for plots and summaries.")
    parser.add_argument("--metric", type=str, default="accuracy", help="Primary metric to plot (metric_name).")
    parser.add_argument("--list-metrics", action="store_true", help="List unique metric names found and exit.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    setup_logging(out_dir)

    logging.info("Analyzing ablation results in %s", in_dir)
    df = load_experiment_csvs(in_dir)
    df_long = standardize_long_format(df)

    unique_metrics = sorted(df_long["metric_name"].unique().tolist())
    logging.info("Found metrics: %s", unique_metrics)
    if args.list_metrics:
        print("\n".join(unique_metrics))
        return

    metric_to_plot = args.metric
    if metric_to_plot not in unique_metrics:
        logging.warning("Requested metric '%s' not found. Defaulting to first metric: %s", metric_to_plot, unique_metrics[0])
        metric_to_plot = unique_metrics[0]

    # Ensure epoch column exists and is numeric
    if "epoch" in df_long.columns:
        df_long["epoch"] = pd.to_numeric(df_long["epoch"], errors="coerce").fillna(0).astype(int)
    elif "round" in df_long.columns:
        df_long["epoch"] = pd.to_numeric(df_long["round"], errors="coerce").fillna(0).astype(int)
    else:
        # add pseudo-epoch as index per source_file
        df_long["epoch"] = df_long.groupby(["source_file"]).cumcount()

    # Plot the chosen metric
    plot_metric_traces(df_long, metric_to_plot, out_dir)

    # Summarize and save CSV
    summary_path = summarize_metrics(df_long, out_dir)
    logging.info("Analysis complete. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
