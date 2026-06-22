import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

import pandas as pd


def plot_mutant_prioritization_results(rewards, moving_average, v_losses, p_losses, networks_update_freq, moving_average_window, sut_name,
                                       should_save=False, save_path=None, execution_id=None):

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='blue', label='Individual Rewards', alpha=0.7)
    plt.plot(moving_average, color='orange', label='Average Reward', linewidth=2)

    # Labeling
    if parameters_set_id is not None:
        plt.title('Individual Rewards and Average Reward Over Time for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ', Parameters Set ID: ' + str(parameters_set_id) + ')')
    else:
        plt.title('Individual Rewards and Average Reward Over Time for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.legend()
    if should_save and save_path:
        plt.savefig(save_path + '/rewards_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()

    plt.plot(v_losses)
    if parameters_set_id is not None:
        plt.title('Value Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ', Parameters Set ID: ' + str(parameters_set_id) + ')')
    else:
        plt.title('Value Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    if should_save and save_path:
        plt.savefig(save_path + '/value_losses_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()

    plt.plot(p_losses)
    if parameters_set_id is not None:
        plt.title('Policy Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ', Parameters Set ID: ' + str(parameters_set_id) + ')')
    else:
        plt.title('Policy Losses for SUT: ' + sut_name + ' (Execution ID: ' + str(execution_id) + ')')
    plt.xlabel('Update (every ' + str(networks_update_freq) + ' episodes)')
    plt.ylabel('Loss')
    if should_save and save_path:
        plt.savefig(save_path + '/policy_losses_plot_' + sut_name + "_" + str(execution_id) + '.png', bbox_inches='tight')
    plt.show()


def plot_multiple_runs_results(all_runs_rewards, sut_name, should_save=False, save_path=None):

    num_runs = len(all_runs_rewards)

    rewards_matrix = []
    for run_rewards in all_runs_rewards:
        killable_rewards = [r for r in run_rewards if r is not None]
        rewards_matrix.append(killable_rewards)

    min_length = min(len(r) for r in rewards_matrix)
    rewards_matrix = [r[:min_length] for r in rewards_matrix]
    rewards_array = np.array(rewards_matrix)

    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)

    cumulative_means = []
    cumulative_stds = []
    for i in range(1, len(mean_rewards) + 1):
        cumulative_means.append(np.mean(mean_rewards[:i]))
        cum_run_means = [np.mean(run[:i]) for run in rewards_matrix]
        cumulative_stds.append(np.std(cum_run_means))

    cumulative_means = np.array(cumulative_means)
    cumulative_stds = np.array(cumulative_stds)

    x = np.arange(len(mean_rewards))

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Push axis patches behind everything
    ax1.set_zorder(0)
    ax1.patch.set_visible(False)

    # First axis (mean rewards)
    color1 = 'tab:blue'
    ax1.set_xlabel('Mutant Index')
    ax1.set_ylabel('Mean Reward', color=color1)

    # Individual runs
    for run in rewards_matrix:
        ax1.plot(x, run, color='gray', alpha=0.15, linewidth=0.5, zorder=1)

    # Mean reward
    line_mean = ax1.plot(x, mean_rewards, color=color1, linewidth=2.5, label='Mean Reward', zorder=4)

    # Std shaded
    ax1.fill_between(x,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     color=color1,
                     alpha=0.25,
                     label='Mean Std Dev',
                     zorder=2)

    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Second axis
    ax2 = ax1.twinx()
    ax2.set_zorder(0)
    ax2.patch.set_visible(False)

    color2 = 'tab:orange'
    ax2.set_ylabel('Cumulative Average Reward', color=color2)

    line_cum = ax2.plot(x, cumulative_means, color=color2,
                        linewidth=2.5, linestyle='--',
                        label='Cumulative Mean Reward',
                        zorder=5)

    ax2.fill_between(x,
                     cumulative_means - cumulative_stds,
                     cumulative_means + cumulative_stds,
                     color=color2,
                     alpha=0.25,
                     label='Cumulative Std Dev',
                     zorder=3)

    ax2.tick_params(axis='y', labelcolor=color2)

    # Gather legend items
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # LEGEND ATTACHED TO FIGURE (most important fix)
    fig_legend = fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        framealpha=1
    )

    fig_legend.set_zorder(1000)

    plt.title(f'Reward Trends Across {num_runs} Runs for SUT: {sut_name}',
              fontsize=14, pad=20)

    fig.tight_layout()

    if should_save and save_path:
        fig.savefig(f'{save_path}/multi_run_combined_{sut_name}.png',
                    bbox_inches='tight', dpi=300)

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
                print(f"Error reading {json_file}: {e}")
                continue

        if best_value is not None:
            prior_tests_best.append(best_value)
            avg_value = np.mean(all_values) if all_values else best_value
            prior_tests_avg.append(avg_value)
            print(f"{project}: best={best_value}, avg={avg_value:.0f} (da {best_file}, execution_id: {best_execution_id})")
        else:
            print(f"{project}: Number of tests not found")

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