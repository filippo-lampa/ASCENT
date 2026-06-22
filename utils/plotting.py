import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import logging

import pandas as pd


def plot_mutant_prioritization_results(rewards, moving_average, v_losses, p_losses, networks_update_freq, moving_average_window, sut_name,
                                       should_save=False, save_path=None, execution_id=None):

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='blue', label='Individual Rewards', alpha=0.7)
    plt.plot(moving_average, color='orange', label='Average Reward', linewidth=2)

    # Labeling
    plt.title('Individual Rewards and Average Reward Over Time for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.legend()
    if should_save and save_path:
        plt.savefig(save_path + '/rewards_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()

    plt.plot(v_losses)
    plt.title('Value Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    if should_save and save_path:
        plt.savefig(save_path + '/value_losses_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()

    plt.plot(p_losses)
    plt.title('Policy Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    if should_save and save_path:
        plt.savefig(save_path + '/policy_losses_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()


def _stack_runs_with_nan_padding(all_runs_series):
    """Pad run series to max length with NaN so we can compute column-wise stats."""
    if not all_runs_series:
        return np.empty((0, 0), dtype=float)

    cleaned_runs = []
    for run in all_runs_series:
        values = []
        for value in (run or []):
            if value is None:
                values.append(np.nan)
            else:
                values.append(float(value))
        cleaned_runs.append(np.array(values, dtype=float))

    max_len = max((len(run) for run in cleaned_runs), default=0)
    matrix = np.full((len(cleaned_runs), max_len), np.nan, dtype=float)
    for i, run in enumerate(cleaned_runs):
        matrix[i, :len(run)] = run
    return matrix


def _running_mean_ignore_nan(series):
    """Cumulative mean that skips NaN values."""
    out = []
    cumulative_sum = 0.0
    count = 0
    for value in series:
        if not np.isnan(value):
            cumulative_sum += value
            count += 1
        out.append(cumulative_sum / count if count else np.nan)
    return np.array(out, dtype=float)


def _plot_metric_over_time(ax, runs_matrix, title, ylabel, color):
    """Plot per-run traces, mean +/- std, and cumulative mean for one metric."""
    if runs_matrix.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center', va='center')
        return

    mean_values = np.nanmean(runs_matrix, axis=0)
    std_values = np.nanstd(runs_matrix, axis=0)
    cumulative_mean = _running_mean_ignore_nan(mean_values)
    x = np.arange(runs_matrix.shape[1])

    valid_mask = ~np.isnan(mean_values)

    for run in runs_matrix:
        run_mask = ~np.isnan(run)
        ax.plot(x[run_mask], run[run_mask], color='gray', alpha=0.15, linewidth=0.6)

    ax.plot(x[valid_mask], mean_values[valid_mask], color=color, linewidth=2.2, label='Average')

    ax.fill_between(
        x[valid_mask],
        (mean_values - std_values)[valid_mask],
        (mean_values + std_values)[valid_mask],
        color=color,
        alpha=0.2,
        label='Std Dev'
    )

    valid_cum_mask = ~np.isnan(cumulative_mean)
    ax.plot(x[valid_cum_mask], cumulative_mean[valid_cum_mask], color=color, linestyle='--', linewidth=2.0,
            label='Cumulative Average')

    ax.set_title(title)
    ax.set_xlabel('Mutant Index')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_multiple_runs_results(all_runs_rewards, sut_name, should_save=False, save_path=None,
                               all_runs_divergencies=None, all_runs_rank_correlations=None):
    rewards_matrix = _stack_runs_with_nan_padding(all_runs_rewards)
    divergencies_matrix = _stack_runs_with_nan_padding(all_runs_divergencies or [])
    rank_corr_matrix = _stack_runs_with_nan_padding(all_runs_rank_correlations or [])

    metrics_to_plot = [
        {
            "matrix": rewards_matrix,
            "title": f'Average Reward Over Time ({len(all_runs_rewards)} runs) - {sut_name}',
            "ylabel": 'Reward',
            "color": 'tab:blue',
            "filename_suffix": 'rewards'
        },
        {
            "matrix": divergencies_matrix,
            "title": f'Average Divergency Among Agents Over Time - {sut_name}',
            "ylabel": 'Symmetric KL Divergence',
            "color": 'tab:green',
            "filename_suffix": 'divergencies'
        },
        {
            "matrix": rank_corr_matrix,
            "title": f'Average Rank Correlation Among Agents Over Time - {sut_name}',
            "ylabel": 'Spearman Correlation',
            "color": 'tab:purple',
            "filename_suffix": 'rank_correlations'
        }
    ]

    for metric in metrics_to_plot:

        fig, ax = plt.subplots(figsize=(10, 5))

        _plot_metric_over_time(
            ax,
            metric["matrix"],
            metric["title"],
            metric["ylabel"],
            metric["color"]
        )

        ax.legend(loc='upper right', framealpha=1)
        fig.tight_layout()

        if should_save and save_path:
            filename = f'{save_path}/multi_run_{metric["filename_suffix"]}_{sut_name}.png'
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            logging.info(f"Saved: {filename}")

        plt.show()



def plot_broken_y_axis(experiments_path=None):
    prior_tests_best = []
    prior_tests_avg = []

    projects = ['bakerfi', 'quadrata', 'secondswap', 'nextgen', 'thorwallet']

    for project in projects:
        best_value = None
        best_file = None
        best_execution_id = None
        all_values = []

        for json_file in experiments_path.glob(f'{project}*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                    if 'executions' in data and isinstance(data['executions'], list):
                        for execution in data['executions']:
                            # Estrai total_tests_executed_on_killable_mutants
                            if 'total_tests_executed_on_killable_mutants' in execution:
                                value = execution['total_tests_executed_on_killable_mutants']
                                exec_id = execution.get('execution_id', 'N/A')

                                all_values.append(value)

                                # Trova il valore minimo (migliore)
                                if best_value is None or value < best_value:
                                    best_value = value
                                    best_file = json_file.name
                                    best_execution_id = exec_id
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logging.debug(f"Error reading {json_file}: {e}")
                continue

        if best_value is not None:
            prior_tests_best.append(best_value)
            avg_value = np.mean(all_values) if all_values else best_value
            prior_tests_avg.append(avg_value)
            logging.debug(f"{project}: best={best_value}, avg={avg_value:.0f} (da {best_file}, execution_id: {best_execution_id})")
        else:
            logging.debug(f"{project}: Number of tests not found")

    default = [322161, 23917, 16472, 8380, 484]
    # sort prioritization by default descending
    sorted_indices = sorted(range(len(default)), key=lambda i: default[i], reverse=True)
    prior_tests_best = [prior_tests_best[i] for i in sorted_indices]
    prior_tests_avg = [prior_tests_avg[i] for i in sorted_indices]

    data = {
        "Project": ["BakerFi", "Quadrata", "SecondSwap", "NextGen", "THORWallet"],
        "Best Prioritization": prior_tests_best,
        "Avg Prioritization": prior_tests_avg,
        "Baseline": [322161, 23917, 16472, 8380, 484],
    }

    plt.rcParams.update({'font.size': 15})

    df = pd.DataFrame(data)
    x = range(len(df))
    bar_width = 0.28  # Ridotto per fare spazio a 3 barre

    # Custom colors - tonalità di verde
    baseline_color = "#a9dfbf"      # Verde pastello chiaro
    avg_prior_color = "#52be80"     # Verde medio
    best_prior_color = "#229954"    # Verde scuro

    # Create two subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 6), gridspec_kw={'height_ratios': [1, 2]},
                                   constrained_layout=True)

    # Set y-axis limits for both
    ax1.set_ylim(105000, 390000)  # upper range
    ax2.set_ylim(0, 28000)  # lower range

    # Bar plots in both axes - 3 barre per progetto
    bars1_top = ax1.bar([i - bar_width for i in x], df["Baseline"], width=bar_width,
                        color=baseline_color)
    bars2_top = ax1.bar([i for i in x], df["Avg Prioritization"], width=bar_width,
                        color=avg_prior_color)
    bars3_top = ax1.bar([i + bar_width for i in x], df["Best Prioritization"], width=bar_width,
                        color=best_prior_color)

    bars1_bot = ax2.bar([i - bar_width for i in x], df["Baseline"], width=bar_width,
                        color=baseline_color, label="Baseline")
    bars2_bot = ax2.bar([i for i in x], df["Avg Prioritization"], width=bar_width,
                        color=avg_prior_color, label="Avg Prioritization")
    bars3_bot = ax2.bar([i + bar_width for i in x], df["Best Prioritization"], width=bar_width,
                        color=best_prior_color, label="Best Prioritization")

    # Add values to the bars in the bottom axis (ax2) only
    for bar in bars1_bot:
        height = bar.get_height()
        if height < 105000:
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 800, f'{int(height):,}', ha='center', va='bottom',
                     fontsize=12, color="black", zorder=3)

    for bar in bars2_bot:
        height = bar.get_height()
        if height < 105000:
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 800, f'{int(height):,}', ha='center', va='bottom',
                     fontsize=12, color="black", zorder=3)

    for bar in bars3_bot:
        height = bar.get_height()
        if height < 105000:
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 800, f'{int(height):,}', ha='center', va='bottom',
                     fontsize=12, color="black", zorder=3)

    # Add values to the bars in the upper axis (ax1) only
    for bar in bars1_top:
        height = bar.get_height()
        if height > 105000:
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 5000, f'{int(height):,}', ha='center', va='bottom',
                     fontsize=12, color="black", zorder=3)

    for bar in bars2_top:
        height = bar.get_height()
        if height > 105000:
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 5000, f'{int(height):,}', ha='center', va='bottom',
                     fontsize=12, color="black", zorder=3)

    for bar in bars3_top:
        height = bar.get_height()
        if height > 105000:
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 5000, f'{int(height):,}', ha='center', va='bottom',
                     fontsize=12, color="black", zorder=3)

    for ax in (ax1, ax2):
        ax.axhline(y=28000, linestyle='--', color='gray', linewidth=3)
        ax.axhline(y=105000, linestyle='--', color='gray', linewidth=3)

    # Hide spines between ax1 and ax2 and draw break marks
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False, top=False, bottom=False)
    ax2.tick_params(labeltop=False, top=False)
    ax2.xaxis.tick_bottom()

    # Diagonal lines to show break
    d = .010
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d * 1.7, +d * 1.7), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d * 1.7, +d * 1.7), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, +1 + d), **kwargs)

    # Labels and titles
    ax2.set_xlabel("Project", fontsize=18, labelpad=20)
    ax2.set_ylabel("                     Number of Tests Executed", fontsize=18, labelpad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Project"], rotation=0, fontsize=15)
    ax1.set_yticks([150000, 250000, 350000])
    ax1.legend(fontsize=14, loc='upper right')
    ax2.legend(fontsize=14, loc='upper right')

    ax1.grid(zorder=0, axis='y')
    ax2.grid(zorder=0, axis='y')

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    plt.show()

if __name__ == '__main__':
    plot_broken_y_axis(experiments_path=Path(r'C:\Users\Filippo\Projects\ASCENT\experiments\randomized_params'))