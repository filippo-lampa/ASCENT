import json
import os
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def load_executions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["executions"]

def load_default_parameters(json_path, df=None):
    # Default parameters as fallback
    defaults = {
        'c_parameter': 1.0,
        'batch_size': 40,
        'asymmetric_loss_alpha': 6.0,
        'rollout_after': 45,
        'observation_network_buffer_size': 10,
        'observation_network_update_delta': 1,
        'update_delta': 1,
        'buffer_size': None
    }

    with open(json_path, "r") as f:
        data = json.load(f)

    # Use JSON defaults if available, otherwise use hardcoded defaults
    json_defaults = data.get("default_parameters", {})
    defaults.update(json_defaults)

    # If buffer_size is still None and we have a dataframe, use the most frequent value
    if defaults['buffer_size'] is None and df is not None:
        if 'parameters.buffer_size' in df.columns:
            defaults['buffer_size'] = df['parameters.buffer_size'].mode()[0]
            print(f"Determined buffer_size default value (most frequent): {defaults['buffer_size']}")

    return defaults


def df_from_executions(executions):
    rows = []
    for exe in executions:
        row = exe.copy()
        params = row.pop("parameters")
        for k, v in params.items():
            row[f"parameters.{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_executions(df, improvement_col):
    '''Aggregate repeated executions with same parameter values'''
    param_cols = [c for c in df.columns if c.startswith("parameters.")]
    df_agg = df.groupby(param_cols, as_index=False)[improvement_col].mean()
    return df_agg


def global_correlations(df, improvement_col):
    param_cols = [c for c in df.columns if c.startswith("parameters.")]
    results = {}
    for col in param_cols:
        corr, p = pearsonr(df[col], df[improvement_col])
        results[col] = (corr, p)
    return results


def best_parameters(df, improvement_col):
    idx = df[improvement_col].idxmax()
    best_row = df.loc[idx]
    return best_row


def average_top_parameters(df, improvement_col, top_percent=10):
    n_top = max(1, int(len(df) * top_percent / 100))
    top_runs = df.nlargest(n_top, improvement_col)
    param_cols = [c for c in df.columns if c.startswith("parameters.")]
    avg_top_params = top_runs[param_cols].mean()
    return avg_top_params


def best_parameter_values(df, default_params, metric_col="total_tests_executed_on_killable_mutants"):
    param_cols = [c for c in df.columns if c.startswith("parameters.")]
    best_values = {}

    for param_col in param_cols:
        param_name = param_col.replace("parameters.", "")
        default_value = default_params.get(param_name)

        if default_value is None:
            print(f"Warning: No default value found for {param_name}, skipping...")
            continue

        print(f"\n--- Analyzing {param_name} (default value: {default_value}) ---")

        # Print total executions and value distribution
        print(f"Total executions: {len(df)}")
        print(f"Value distribution:")
        value_counts = df[param_col].value_counts().sort_index()
        for val, count in value_counts.items():
            marker = " <- DEFAULT" if (isinstance(val, (int, float)) and abs(val - default_value) < 1e-9) or val == default_value else ""
            print(f"  {val}: {count} executions{marker}")

        # Filter executions where parameter value is different from default
        df_filtered = df[~df[param_col].apply(lambda x: abs(x - default_value) < 1e-9 if isinstance(x, (int, float)) else x == default_value)].copy()

        print(f"Executions after filtering (non-default only): {len(df_filtered)}")

        if len(df_filtered) == 0:
            print(f"Warning: No non-default executions for {param_name}")
            continue

        print(f"Filtered value distribution:")
        filtered_value_counts = df_filtered[param_col].value_counts().sort_index()
        for val, count in filtered_value_counts.items():
            print(f"  {val}: {count} executions")

        # Group by parameter value and average the metric
        param_averages = df_filtered.groupby(param_col)[metric_col].mean()

        print(f"Average {metric_col} by parameter value:")
        for val, avg in param_averages.items():
            print(f"  {val}: {avg:.2f}")

        # Select the parameter value with the lowest average number of tests executed
        best_value = param_averages.idxmin()
        best_avg = param_averages.min()

        best_values[param_name] = {
            'value': best_value,
            'avg_tests': best_avg,
            'n_executions': len(df_filtered[df_filtered[param_col] == best_value])
        }

        print(f"Best value: {best_value} (avg tests: {best_avg:.2f})")

    return best_values


def analyze_parameter_changes(df, param_name, default_value=None, metric_col="total_tests_executed_on_killable_mutants"):
    param_col = f"parameters.{param_name}"

    if param_col not in df.columns:
        print(f"Error: {param_col} not found in dataframe")
        return None

    print(f"\n--- Analyzing {param_name} changes ---")

    # If default_value is None, use the most frequent value
    if default_value is None:
        default_value = df[param_col].mode()[0]
        print(f"Default value (most frequent): {default_value}")
    else:
        print(f"Default value: {default_value}")

    # Step 1: Filter executions where parameter value is different from default
    df_filtered = df[~df[param_col].apply(lambda x: abs(x - default_value) < 1e-9 if isinstance(x, (int, float)) else x == default_value)].copy()

    # Get all unique values (including defaults)
    unique_values = df[param_col].unique()
    print(f"Unique values found: {sorted(unique_values)}")

    if len(df_filtered) == 0:
        print("Warning: No non-default executions found")
        return None

    # Average the metric for non-default executions
    avg_tests = df_filtered[metric_col].mean()

    print(f"Total executions: {len(df)}")
    print(f"Non-default executions: {len(df_filtered)}")
    print(f"Average {metric_col} (non-default only): {avg_tests:.2f}")

    print(f"\nBreakdown by {param_name} value:")
    for val in sorted(unique_values):
        df_val = df[df[param_col] == val]
        avg_val = df_val[metric_col].mean()
        marker = " <- DEFAULT" if (isinstance(val, (int, float)) and abs(val - default_value) < 1e-9) or val == default_value else ""
        print(f"  Value {val}: {len(df_val)} executions, avg tests: {avg_val:.2f}{marker}")

    return avg_tests


def analyze_buffer_size_changes(df, metric_col="total_tests_executed_on_killable_mutants"):
    return analyze_parameter_changes(df, "buffer_size", default_value=None, metric_col=metric_col)


def plot_parameters(df, improvement_col, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    param_cols = [c for c in df.columns if c.startswith("parameters.")]
    for col in param_cols:
        plt.figure(figsize=(7,5))
        sns.scatterplot(data=df, x=col, y=improvement_col)
        sns.regplot(data=df, x=col, y=improvement_col, scatter=False, color='red', line_kws={'label':'Trend'})
        plt.xlabel(col)
        plt.ylabel(improvement_col)
        plt.title(f"{col} vs {improvement_col}")
        plt.legend()
        plt.tight_layout()
        filename = f"{col.replace('.', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def isolated_parameter_analysis(df, improvement_col, output_dir="isolated_param_plots"):
    os.makedirs(output_dir, exist_ok=True)
    param_cols = [c for c in df.columns if c.startswith("parameters.")]
    results = {}

    for target_param in param_cols:
        other_params = [c for c in param_cols if c != target_param]
        grouped = df.groupby(other_params)
        param_results = []

        for _, group in grouped:
            if group[target_param].nunique() > 1:
                group_sorted = group.sort_values(target_param)
                plt.figure(figsize=(7,5))
                sns.scatterplot(data=group_sorted, x=target_param, y=improvement_col)
                sns.regplot(data=group_sorted, x=target_param, y=improvement_col,
                            scatter=False, color='red', line_kws={'label':'Trend'})
                plt.xlabel(target_param)
                plt.ylabel(improvement_col)
                plt.title(f"{target_param} isolated")
                plt.legend()
                plt.tight_layout()
                group_hash = hashlib.md5(str(group[other_params].iloc[0].values).encode()).hexdigest()[:8]
                filename = f"{target_param.replace('.', '_')}_isolated_{group_hash}.png"
                plt.savefig(os.path.join(output_dir, filename))
                plt.close()
                # Correlation and p-value for this subset
                corr, p = pearsonr(group_sorted[target_param], group_sorted[improvement_col])
                param_results.append((corr, p))

        if param_results:
            avg_corr = sum(r[0] for r in param_results)/len(param_results)
            avg_p = sum(r[1] for r in param_results)/len(param_results)
            results[target_param] = (avg_corr, avg_p)
    return results


def main(json_file, top_percent=10):
    improvement_col = "percentual_improvement_on_killable_mutants"
    metric_col = "total_tests_executed_on_killable_mutants"

    executions = load_executions(json_file)
    df = df_from_executions(executions)

    # Load default parameters, passing df to determine buffer_size default
    default_params = load_default_parameters(json_file, df=df)

    df_agg = aggregate_executions(df, improvement_col)

    # Global correlations
    correlations = global_correlations(df_agg, improvement_col)
    print("Global correlations and p-values:")
    for k, v in correlations.items():
        print(f"{k}: corr={v[0]:.4f}, p={v[1]:.4f}")

    # Best parameters (single top run)
    best_row = best_parameters(df_agg, improvement_col)
    print("\nBest parameter combination (single run):")
    for col in [c for c in df_agg.columns if c.startswith("parameters.")]:
        print(f"{col}: {best_row[col]}")
    print(f"{improvement_col}: {best_row[improvement_col]}")

    # Average top-N% parameters
    avg_top_params = average_top_parameters(df_agg, improvement_col, top_percent=top_percent)
    print(f"\nAverage best parameters across top {top_percent}% of runs:")
    for col, val in avg_top_params.items():
        print(f"{col}: {val:.4f}")

    # Best parameter values based on non-default executions
    best_vals = best_parameter_values(df, default_params, metric_col=metric_col)
    print("\nBest parameter values (based on non-default executions):")
    for param_name, info in best_vals.items():
        print(f"{param_name}: {info['value']} (avg tests: {info['avg_tests']:.2f}, n_executions: {info['n_executions']})")


    # Analyze buffer_size changes
    analyze_buffer_size_changes(df, metric_col=metric_col)

    # General plots
    plot_parameters(df_agg, improvement_col)

    # Isolated-parameter plots
    isolated_results = isolated_parameter_analysis(df_agg, improvement_col)
    print("\nIsolated-parameter average correlations and p-values:")
    for k, v in isolated_results.items():
        print(f"{k}: avg_corr={v[0]:.4f}, avg_p={v[1]:.4f}")


if __name__ == "__main__":
    json_file = r"/experiments/isolated_params/secondswap.json"
    main(json_file, top_percent=10)
