import json
import os

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import logging

from experiment_tracker import (analyze_experiment, analyze_pair_disagreement, analyze_disagreement_vs_reward)

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["executions"]


def extract_parameters(executions):
    param_names = list(executions[0]["parameters"].keys())
    X = {p: [] for p in param_names}
    y = []

    for ex in executions:
        for p in param_names:
            X[p].append(ex["parameters"][p])
        y.append(ex["total_tests_executed_on_killable_mutants"])

    return param_names, X, y


def compute_correlations(param_names, X, y):
    results = {}
    for p in param_names:
        corr, pval = pearsonr(X[p], y)
        results[p] = (float(corr), float(pval))
    return results


def plot_parameters(param_names, X, y):
    for p in param_names:
        plt.figure()
        x_vals = np.array(X[p])
        y_vals = np.array(y)

        plt.scatter(x_vals, y_vals)
        coeffs = np.polyfit(x_vals, y_vals, 1)
        trend = np.poly1d(coeffs)

        plt.plot(x_vals, trend(x_vals), color='red', label='Trend')
        plt.title(f"{p} vs Killable Mutant Tests")
        plt.xlabel(p)
        plt.ylabel("Tests executed on killable mutants")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if not os.path.exists("plots"):
            os.makedirs("plots")

        plt.savefig(os.path.join("plots", f"{p}_vs_killable_mutants.png"))


def best_single_run(executions):
    best = max(executions, key=lambda e: e["percentual_improvement_on_killable_mutants"])
    return best["parameters"], float(best["percentual_improvement_on_killable_mutants"])


def top_percent_runs(executions, top_percent=10):
    n_top = max(1, int(len(executions) * top_percent / 100))

    sorted_ex = sorted(
        executions,
        key=lambda e: e["percentual_improvement_on_killable_mutants"],
        reverse=True
    )
    return sorted_ex[:n_top]


def top_percent_improvement(executions, top_percent=10):
    top_ex = top_percent_runs(executions, top_percent)
    improvements = [e["percentual_improvement_on_killable_mutants"] for e in top_ex]

    return {
        "mean_top": float(np.mean(improvements)),
        "max_top": float(np.max(improvements)),
        "count_top": len(top_ex),
    }


def compute_global_statistics(executions):
    improvements = [e["percentual_improvement_on_killable_mutants"] for e in executions]
    baseline_tests = [e["baseline_total_tests_executed_on_killable_mutants"] for e in executions]
    prioritized_tests = [e["total_tests_executed_on_killable_mutants"] for e in executions]

    return {
        "mean_improvement": float(np.mean(improvements)),
        "std_improvement": float(np.std(improvements)),
        "var_improvement": float(np.var(improvements)),
        "best_improvement": float(np.max(improvements)),
        "avg_baseline": float(np.mean(baseline_tests)),
        "avg_prioritized": float(np.mean(prioritized_tests)),
    }


def snap_value_to_grid(value, allowed):
    allowed = np.array(allowed)
    idx = np.argmin(np.abs(allowed - value))
    return float(allowed[idx])

def compute_consensus_parameters(executions, allowed_grid, top_percent=10):
    top_ex = top_percent_runs(executions, top_percent)
    sample = top_ex[0]["parameters"].keys()
    consensus = {}

    for param in sample:
        allowed = allowed_grid[param]
        values = np.array([e["parameters"][param] for e in top_ex])

        if np.issubdtype(np.array(allowed).dtype, np.number):

            if max(allowed) / max(1, min(allowed)) > 50:
                med = np.exp(np.median(np.log(values + 1e-9)))
            else:
                med = np.median(values)

            snapped = snap_value_to_grid(med, allowed)
            consensus[param] = snapped

        else:
            vals, counts = np.unique(values, return_counts=True)
            consensus[param] = vals[np.argmax(counts)]

    return consensus


def evaluate_consensus_configuration(executions, consensus_params):
    matching = [
        e for e in executions
        if all(e["parameters"][p] == consensus_params[p] for p in consensus_params)
    ]

    if not matching:
        return None

    improvements = [e["percentual_improvement_on_killable_mutants"] for e in matching]
    baseline_tests = [e["baseline_total_tests_executed_on_killable_mutants"] for e in matching]
    prioritized_tests = [e["total_tests_executed_on_killable_mutants"] for e in matching]

    return {
        "count": len(matching),
        "mean_improvement": float(np.mean(improvements)),
        "std_improvement": float(np.std(improvements)),
        "best_improvement": float(np.max(improvements)),
        "avg_baseline": float(np.mean(baseline_tests)),
        "avg_prioritized": float(np.mean(prioritized_tests)),
    }

def analyze_committee(tracker_json_path):
    analyze_experiment(tracker_json_path)
    analyze_pair_disagreement(tracker_json_path)
    analyze_disagreement_vs_reward(tracker_json_path)

def main(path, allowed_grid, top_percent=10):

    executions = load_data(path)

    param_names, X, y = extract_parameters(executions)

    logging.info("\nCorrelation results")
    logging.info("------------------")
    correlations = compute_correlations(param_names, X, y)
    for p, (corr, pval) in correlations.items():
        logging.info(f"{p}: corr={corr:.4f}, p={pval:.4f}")

    logging.info("\nGlobal improvement statistics")
    logging.info("-----------------------------")
    global_stats = compute_global_statistics(executions)
    for k, v in global_stats.items():
        logging.info(f"{k}: {v}")

    logging.info(f"\nTop {top_percent} percent improvement")
    logging.info("--------------------------------------")
    top_stats = top_percent_improvement(executions, top_percent)
    for k, v in top_stats.items():
        logging.info(f"{k}: {v}")

    logging.info("\nConsensus parameters from top percent")
    logging.info("-------------------------------------")
    consensus_params = compute_consensus_parameters(executions, allowed_grid, top_percent)
    for k, v in consensus_params.items():
        logging.info(f"{k}: {v}")

    consensus_eval = evaluate_consensus_configuration(executions, consensus_params)

    if consensus_eval:
        logging.info("\nConsensus configuration evaluation")
        logging.info("----------------------------------")
        for k, v in consensus_eval.items():
            logging.info(f"{k}: {v}")
    else:
        logging.info("\nNo runs matched the consensus configuration")

    logging.info("\nBest single run")
    logging.info("----------------")
    best_params, best_value = best_single_run(executions)
    logging.info(f"Best improvement: {best_value}")
    for k, v in best_params.items():
        logging.info(f"{k}: {v}")

    plot_parameters(param_names, X, y)


if __name__ == "__main__":

    path = r""

    allowed_grid = {
        "buffer_size": [50, 100, 200, 400, 800, 1600],
        "batch_size": [16, 32, 40, 64, 128],
        "update_delta": [1, 3, 5, 7, 10],
        "observation_network_update_delta": [1, 3, 5, 7, 10],
        "observation_network_buffer_size": [5, 10, 15, 25, 35, 50],
        "rollout_after": [0, 25, 45, 50, 75, 100],
        "asymmetric_loss_alpha": [1.0, 3.25, 5.5, 6.0, 7.75, 10.0],
        "value_network_learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "policy_network_learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "observation_network_learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "c_parameter": [0.1, 1.0, 2.0, 3.0, 5.0]
    }

    main(path, allowed_grid, top_percent=10)
