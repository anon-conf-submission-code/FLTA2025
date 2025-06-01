import numpy as np
import pandas as pd # Though not directly used in funcs, often useful for data prep before calling
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

# Import functions from analysis_orchestration for data generation if needed for plots
from analysis_orchestration import repeat_experiments_parallel, compute_confidence_intervals, compute_p_values
from survival_core import kaplan_meier # For original curve plotting
from data_handling import assign_data_to_nodes # For setting up nodes if visualizing partitions
from multiprocessing import Manager # For creating shared cache for visualization repetitions
from config import (
    TIME_POINTS_COUNT,
    PLOT_DPI,
    PLOT_FIGSIZE,
    NUM_NODES_FOR_VIZ,
    DIST_NON_UNIFORM_MAJOR_FRAC,
    DIST_HIGHLY_IMBALANCED_MAJOR_FRAC
)

plt.switch_backend('Agg') # Set backend for non-interactive plotting

def visualize_km_comparisons_grid(
    df: pd.DataFrame, 
    partitioner, # data_handling.DataPartitioner instance
    methods: list[str], 
    epsilons: list[float], 
    repetitions: int = 5, 
    save_dir: str = '.',
    num_nodes_for_dist: int = NUM_NODES_FOR_VIZ, # Number of nodes for visualization partitions
    dist_non_uniform_major_frac: float = DIST_NON_UNIFORM_MAJOR_FRAC, # For non-uniform partition viz
    dist_highly_imbalanced_major_frac: float = DIST_HIGHLY_IMBALANCED_MAJOR_FRAC # For highly imbalanced partition viz
):
    """Generates and saves a grid of plots comparing DP-KM curves to centralized KM."""
    fig, axes = plt.subplots(3, 4, figsize=PLOT_FIGSIZE)
    max_duration = df['duration'].max() if not df.empty else 100 # Default max_duration if df is empty
    time_points = np.linspace(0, max_duration, TIME_POINTS_COUNT, dtype=np.float32)
    
    distributions = [
        ('Uniform', partitioner.uniform_partition(num_nodes_for_dist)),
        ('Non-uniform', partitioner.nonuniform_partition(num_nodes_for_dist, fraction_major=dist_non_uniform_major_frac)),
        ('Highly Imbalanced', partitioner.highly_imbalanced_partition(num_nodes_for_dist, fraction_major=dist_highly_imbalanced_major_frac))
    ]

    original_curve = kaplan_meier(df, time_points)

    for row, (dist_name, partitions) in enumerate(distributions):
        nodes = assign_data_to_nodes(partitions)
        with Manager() as manager: # Manager for this distribution's set of method/epsilon plots
            shared_sensitivity_cache_for_dist_plots = manager.dict()

            for col, method in enumerate(methods): # Expects 4 methods now
                ax = axes[row, col]
                ax.step(time_points, original_curve, where='post', label='Centralized KM', linestyle='--', color='black')

                for epsilon in epsilons:
                    # Pass the shared cache to repeat_experiments_parallel for this set of calls
                    repeated_curves = repeat_experiments_parallel(
                        nodes, time_points, method, epsilon, repetitions, 
                        external_sensitivity_cache=shared_sensitivity_cache_for_dist_plots,
                        distribution_name=dist_name
                    )
                    
                    if repeated_curves.size == 0:
                        continue

                    mean_curve, lower_ci, upper_ci = compute_confidence_intervals(repeated_curves)

                    if lower_ci.size > 0 and upper_ci.size > 0 and mean_curve.size > 0:
                        ir_lower = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
                        lower_ci_monotone = ir_lower.fit_transform(time_points, lower_ci)
                        
                        ir_upper = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
                        upper_ci_monotone = ir_upper.fit_transform(time_points, upper_ci)
                        
                        ir_mean = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
                        mean_curve_monotone = ir_mean.fit_transform(time_points, mean_curve)

                        ax.step(time_points, mean_curve_monotone, where='post', label=f'{method} ε={epsilon}')
                        ax.fill_between(time_points, lower_ci_monotone, upper_ci_monotone, step='post', alpha=0.2)
                    elif mean_curve.size > 0 :
                        ir_mean = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
                        mean_curve_monotone = ir_mean.fit_transform(time_points, mean_curve)
                        ax.step(time_points, mean_curve_monotone, where='post', label=f'{method} ε={epsilon} (no CI)')

                ax.set_xlabel('Time')
                ax.set_ylabel('Survival Probability')
                ax.set_title(f'{dist_name} - {method}')
                ax.legend(fontsize='small')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_ylim([-0.05, 1.05])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Federated DP Kaplan-Meier Curves (Repetitions: {repetitions})', fontsize=16)
    plot_path = f'{save_dir}/federated_dp_km_visualization.png'
    plt.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"Saved KM visualization: {plot_path}")

def visualize_mae_with_ci(
    df: pd.DataFrame, 
    partitioner, 
    methods: list[str], 
    epsilons: list[float], 
    repetitions: int = 5, 
    save_dir: str = '.',
    num_nodes_for_dist: int = NUM_NODES_FOR_VIZ,
    dist_non_uniform_major_frac: float = DIST_NON_UNIFORM_MAJOR_FRAC,
    dist_highly_imbalanced_major_frac: float = DIST_HIGHLY_IMBALANCED_MAJOR_FRAC
) -> pd.DataFrame:
    """Generates and saves bar plots of Mean Absolute Error with CIs.
    Also returns a DataFrame of individual MAE values per run.
    """
    mae_records = []

    fig, axes = plt.subplots(3, 4, figsize=PLOT_FIGSIZE, sharey=True)
    max_duration = df['duration'].max() if not df.empty else 100
    time_points = np.linspace(0, max_duration, TIME_POINTS_COUNT, dtype=np.float32)
    
    distributions = [
        ('Uniform', partitioner.uniform_partition(num_nodes_for_dist)),
        ('Non-uniform', partitioner.nonuniform_partition(num_nodes_for_dist, fraction_major=dist_non_uniform_major_frac)),
        ('Highly Imbalanced', partitioner.highly_imbalanced_partition(num_nodes_for_dist, fraction_major=dist_highly_imbalanced_major_frac))
    ]
    original_curve = kaplan_meier(df, time_points)

    for row, (dist_name, partitions) in enumerate(distributions):
        nodes = assign_data_to_nodes(partitions)
        with Manager() as manager:
            shared_sensitivity_cache_for_dist_plots = manager.dict()

            for col, method in enumerate(methods):
                ax = axes[row, col]
                mae_means = []
                mae_lower_cis = []
                mae_upper_cis = []

                for epsilon in epsilons:
                    repeated_curves = repeat_experiments_parallel(
                        nodes, time_points, method, epsilon, repetitions,
                        external_sensitivity_cache=shared_sensitivity_cache_for_dist_plots,
                        distribution_name=dist_name
                    )
                    if repeated_curves.size == 0 or repeated_curves.shape[0] < 1:
                        mae_means.append(np.nan)
                        mae_lower_cis.append(np.nan)
                        mae_upper_cis.append(np.nan)
                        for _ in range(repetitions):
                            mae_records.append({
                                "Partition":   dist_name,
                                "Method":      method,
                                "Epsilon":     epsilon,
                                "MAE":         np.nan
                            })
                        continue
                    
                    mae_values_for_epsilon = np.mean(np.abs(
                        repeated_curves - original_curve.reshape(1, -1)
                    ), axis=1, dtype=np.float32)
                    
                    for val in mae_values_for_epsilon:
                        mae_records.append({
                            "Partition":   dist_name,
                            "Method":      method,
                            "Epsilon":     epsilon,
                            "MAE":         val
                        })
                        
                    if not mae_values_for_epsilon.size:
                        mae_means.append(np.nan); mae_lower_cis.append(np.nan); mae_upper_cis.append(np.nan)
                        continue

                    mean_mae = np.mean(mae_values_for_epsilon)
                    lower_ci_mae = np.percentile(mae_values_for_epsilon, 2.5)
                    upper_ci_mae = np.percentile(mae_values_for_epsilon, 97.5)
                    
                    del repeated_curves

                    mae_means.append(mean_mae)
                    mae_lower_cis.append(mean_mae - lower_ci_mae)
                    mae_upper_cis.append(upper_ci_mae - mean_mae)

                x_labels = [str(e) for e in epsilons]
                ax.bar(x_labels, mae_means, yerr=[mae_lower_cis, mae_upper_cis], capsize=5, color='skyblue', ecolor='gray')
                ax.set_xlabel('Epsilon (ε)')
                ax.set_ylabel('Mean Absolute Error (MAE)' if col == 0 else '')
                ax.set_title(f'{dist_name} - {method}')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.suptitle(f'Mean Absolute Error (MAE) vs Centralized KM (Repetitions: {repetitions})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = f'{save_dir}/mae_with_ci_visualization.png'
    plt.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"Saved MAE visualization: {plot_path}")

    mae_runs_df = pd.DataFrame(mae_records)
    if not mae_runs_df.empty:
        mae_runs_df["Epsilon"] = mae_runs_df["Epsilon"].astype(float)
        mae_runs_df["MAE"] = mae_runs_df["MAE"].astype(float)

    return mae_runs_df

def visualize_p_values_ttest(
    df: pd.DataFrame, 
    partitioner, 
    methods: list[str], 
    epsilons: list[float], 
    repetitions: int = 5, 
    save_dir: str = '.',
    num_nodes_for_dist: int = NUM_NODES_FOR_VIZ,
    dist_non_uniform_major_frac: float = DIST_NON_UNIFORM_MAJOR_FRAC,
    dist_highly_imbalanced_major_frac: float = DIST_HIGHLY_IMBALANCED_MAJOR_FRAC
):
    """Generates and saves plots of p-values from t-tests over time points."""
    fig, axes = plt.subplots(3, 4, figsize=PLOT_FIGSIZE, sharey=True)
    max_duration = df['duration'].max() if not df.empty else 100
    time_points = np.linspace(0, max_duration, TIME_POINTS_COUNT, dtype=np.float32)

    distributions = [
        ('Uniform', partitioner.uniform_partition(num_nodes_for_dist)),
        ('Non-uniform', partitioner.nonuniform_partition(num_nodes_for_dist, fraction_major=dist_non_uniform_major_frac)),
        ('Highly Imbalanced', partitioner.highly_imbalanced_partition(num_nodes_for_dist, fraction_major=dist_highly_imbalanced_major_frac))
    ]
    original_curve = kaplan_meier(df, time_points)

    for row, (dist_name, partitions) in enumerate(distributions):
        nodes = assign_data_to_nodes(partitions)
        with Manager() as manager:
            shared_sensitivity_cache_for_dist_plots = manager.dict()

            for col, method in enumerate(methods): # Expects 4 methods now
                ax = axes[row, col]
                for epsilon in epsilons:
                    repeated_curves = repeat_experiments_parallel(
                        nodes, time_points, method, epsilon, repetitions,
                        external_sensitivity_cache=shared_sensitivity_cache_for_dist_plots,
                        distribution_name=dist_name
                    )
                    if repeated_curves.size == 0 or repeated_curves.shape[0] < 2:
                        continue 
                    
                    p_values = compute_p_values(original_curve, repeated_curves)
                    if p_values.size > 0:
                        ax.plot(time_points, p_values, label=f'ε={epsilon}', alpha=0.8)

                ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05', alpha=0.9)
                ax.set_xlabel('Time')
                ax.set_ylabel('p-value' if col == 0 else '')
                ax.set_title(f'{dist_name} - {method}')
                ax.set_yscale('log')
                ax.legend(fontsize='small')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_ylim([1e-5, 1.1])

    fig.suptitle(f'Pointwise p-values (Paired t-test vs Centralized KM) (Repetitions: {repetitions})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = f'{save_dir}/p_values_ttest_visualization.png'
    plt.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"Saved p-value t-test visualization: {plot_path}")

def visualize_data_distribution(
    df: pd.DataFrame, 
    partitioner, 
    save_dir: str = '.',
    num_nodes_for_dist: int = NUM_NODES_FOR_VIZ,
    dist_non_uniform_major_frac: float = DIST_NON_UNIFORM_MAJOR_FRAC,
    dist_highly_imbalanced_major_frac: float = DIST_HIGHLY_IMBALANCED_MAJOR_FRAC
):
    """Visualizes data distribution across nodes for different partitioning strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    strategies = [
        ('Uniform', partitioner.uniform_partition(num_nodes_for_dist)),
        ('Non-uniform', partitioner.nonuniform_partition(num_nodes_for_dist, fraction_major=dist_non_uniform_major_frac)),
        ('Highly Imbalanced', partitioner.highly_imbalanced_partition(num_nodes_for_dist, fraction_major=dist_highly_imbalanced_major_frac))
    ]
    
    # Get the colormap names for each strategy
    colormaps_for_strategies = ['viridis', 'plasma', 'inferno'] 

    for idx, (strategy_name, partitions) in enumerate(strategies):
        sizes = [len(p) for p in partitions if p is not None and not p.empty]
        total_data = sum(sizes)
        current_cmap_name = colormaps_for_strategies[idx % len(colormaps_for_strategies)]
        
        # Determine number of actual partitions to avoid cmap errors if less than num_nodes_for_dist
        actual_num_partitions = len(partitions)
        if actual_num_partitions == 0:
             bar_colors = [] # No partitions, no colors
        else:
            cmap = plt.cm.get_cmap(current_cmap_name, actual_num_partitions) # Get cmap with actual number of partitions
            bar_colors = [cmap(i) for i in range(actual_num_partitions)]

        if total_data == 0: 
            percentages = [0] * actual_num_partitions
            if not bar_colors and actual_num_partitions > 0: # if cmap failed but we have partitions
                bar_colors = ['gray'] * actual_num_partitions
        else: 
            percentages = [(s / total_data) * 100 for s in sizes]
        
        ax = axes[idx]
        node_labels = [f'Node {i+1}' for i in range(actual_num_partitions)]
        
        if len(node_labels) > 0 and len(percentages) == len(node_labels):
            bars = ax.bar(node_labels, percentages, color=bar_colors if bar_colors else 'gray')
            for bar_idx, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, "No data to display", ha='center', va='center', transform=ax.transAxes)

        ax.set_title(f'{strategy_name} Distribution ({actual_num_partitions} of {num_nodes_for_dist} nodes shown)')
        ax.set_xlabel('Node')
        ax.set_ylabel('Percentage of Total Data (%)')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
            
    fig.suptitle('Data Distribution Across Nodes for Visualization Strategies', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f'{save_dir}/data_distribution_visualization.png'
    plt.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"Saved data distribution visualization: {plot_path}")

def visualize_event_distribution(
    df: pd.DataFrame, 
    dataset_name: str, 
    save_dir: str = '.'
):
    """Visualizes the event status distribution (0s and 1s) for a dataset."""
    if df.empty:
        print(f"Cannot visualize event distribution for {dataset_name}: DataFrame is empty.")
        return

    plt.figure(figsize=(8, 6))
    event_counts = df['event'].value_counts().sort_index()
    
    labels = []
    counts = []
    # Ensure we have labels for 0 and 1, even if one is missing in data
    if 0 in event_counts.index:
        labels.append(f'Censored/No Event (0)\n(n={event_counts[0]})')
        counts.append(event_counts[0])
    else:
        labels.append(f'Censored/No Event (0)\n(n=0)')
        counts.append(0)
        
    if 1 in event_counts.index:
        labels.append(f'Event (1)\n(n={event_counts[1]})')
        counts.append(event_counts[1])
    else:
        labels.append(f'Event (1)\n(n=0)')
        counts.append(0)

    bars = plt.bar(labels, counts, color=['skyblue', 'salmon'])
    
    plt.title(f'{dataset_name} - Event Status Distribution (Total N={len(df)})')
    plt.xlabel('Event Status')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = f'{save_dir}/{dataset_name}_event_distribution.png'
    plt.savefig(plot_path, dpi=PLOT_DPI)
    plt.close()
    print(f"Saved event distribution plot: {plot_path}") 