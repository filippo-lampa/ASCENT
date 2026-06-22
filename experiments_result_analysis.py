import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import argparse
import logging

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["executions"]

def flatten_executions(executions):
    rows = []
    for ex in executions:
        base = {
            "execution_id": ex["execution_id"],
            "sut_name": ex["sut_name"],
            "total_tests_executed_on_killable_mutants": ex["total_tests_executed_on_killable_mutants"]
        }
        # merge parameters
        row = {**base, **ex["parameters"]}
        rows.append(row)
    return pd.DataFrame(rows)

def compute_correlations(df, target):
    correlations = {}
    p_values = {}

    for col in df.columns:
        if col not in ["execution_id", "sut_name", target]:
            corr, p = pearsonr(df[col], df[target])
            correlations[col] = corr
            p_values[col] = p

    return correlations, p_values

def plot_parameters(df, target, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    for col in df.columns:
        if col in ["execution_id", "sut_name", target]:
            continue

        x = df[col].values
        y = df[target].values

        plt.figure()
        plt.scatter(x, y, label="Data points")

        # Trend line: linear regression
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        plt.plot(x, y_pred, label="Trend line", linewidth=2, color='red')

        plt.xlabel(col)
        plt.ylabel(target)
        plt.title(f"{col} vs {target}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{col}_vs_{target}.png")
        plt.close()

def main():
    
    parser = argparse.ArgumentParser(description='ASCENT experiments statistical analyzer')
    parser.add_argument('--experiments_file_path', type=str, help='Path to the json experiment file.')

    args = parser.parse_args()
    experiments_file_path = args.experiments_file_path

    executions = load_json(experiments_file_path)
    df = flatten_executions(executions)

    target = "total_tests_executed_on_killable_mutants"

    correlations, p_values = compute_correlations(df, target)

    logging.info("\nPearson Correlation Coefficients:")
    for k, v in correlations.items():
        logging.info(f"{k}: {v}")

    logging.info("\nP-values:")
    for k, v in p_values.items():
        logging.info(f"{k}: {v}")

    logging.info("\nGenerating plots...")
    plot_parameters(df, target)
    logging.info("Plots saved in 'plots/' directory.")

if __name__ == "__main__":
    main()
