import os
import sys
import warnings
from time import time
from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing import cpu_count # Manager is not directly used here
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Add script's directory to sys.path ---
# This helps worker processes (and this script) find the new modules.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir) # Insert at beginning

# --- Configuration ---
try:
    from config import VERBOSE_DEBUG, SEED_OFFSET
except ImportError as e:
    print(f"Fatal Error: config.py not found or import error: {e}. Ensure it's in the Python path.")
    sys.exit(1)

# --- Core Modules ---
try:
    from data_handling import load_ncctg_lung_data, DataPartitioner, assign_data_to_nodes
    from analysis_orchestration import perform_logrank_analysis_corrected
    from plotting_utils import (
        visualize_event_distribution, 
        visualize_data_distribution,
        visualize_km_comparisons_grid,
        visualize_mae_with_ci, 
        visualize_p_values_ttest
    )
    from reporting_utils import save_logrank_results, save_timing_results, save_mae_table, save_logrank_fp_rate_table, save_method_ranking_tables
except ImportError as e:
    print(f"Fatal Error: Could not import one or more core modules: {e}. "
          f"Ensure all .py files (config.py, data_handling.py, dp_core.py, survival_core.py, "
          f"aggregation_core.py, parallel_logic.py, analysis_orchestration.py, plotting_utils.py, "
          f"reporting_utils.py) are present in the directory: {script_dir} or in the Python path.")
    sys.exit(1)

# --- Global Settings ---
warnings.filterwarnings('ignore') # Suppress warnings globally

# Global cache for sensitivity results IN THE MAIN PROCESS.
sensitivity_cache_main_process = {}

def analyze_dataset_parallel(args_tuple: tuple):
    """Analyzes a single dataset: partitions data, runs analyses, generates plots and reports."""
    main_df, dataset_name_str, analysis_methods, epsilon_values, num_repetitions = args_tuple
    
    print(f"\n🚀 Starting analysis for dataset: {dataset_name_str} 🚀")
    overall_dataset_analysis_start_time = time()

    # Output directory for this dataset
    dataset_output_dir = dataset_name_str 
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Analyze TV parameters for different data sizes
    print(f"\n📊 [{dataset_name_str}] Analyzing TV smoothing parameters...")
    from dp_core import analyze_tv_parameters
    data_sizes = [10, 50, 200]  # Example data sizes for analysis
    tv_analysis = {size: analyze_tv_parameters(size) for size in data_sizes}
    
    # Save TV parameter analysis
    tv_analysis_file = os.path.join(dataset_output_dir, "tv_parameter_analysis.txt")
    with open(tv_analysis_file, "w") as f:
        f.write("TV Parameter Analysis\n")
        f.write("====================\n\n")
        for size, analysis in tv_analysis.items():
            f.write(f"Dataset Size: {size}\n")
            f.write(f"  Base value: {analysis['base_value']:.3f}\n")
            f.write(f"  Size scaling factor: {analysis['size_scaling']:.3f}\n")
            f.write(f"  Theory adjustment: {analysis['theory_adjustment']:.3f}\n")
            f.write(f"  Final lambda: {analysis['final_lambda']:.3f}\n")
            f.write(f"  Component effects:\n")
            f.write(f"    - Base effect: {analysis['components']['base_effect']:.3f}\n")
            f.write(f"    - With size scaling: {analysis['components']['scaling_effect']:.3f}\n")
            f.write(f"    - Total effect: {analysis['components']['total_effect']:.3f}\n")
            f.write("\n")
    print(f"  Saved TV parameter analysis to: {tv_analysis_file}")

    # Instantiate DataPartitioner for this dataset
    partitioner_instance = DataPartitioner(main_df)

    # --- Initial Data Visualizations ---
    print(f"\n📊 [{dataset_name_str}] Generating event distribution plot...")
    visualize_event_distribution(main_df, dataset_name_str, save_dir=dataset_output_dir)

    # --- Main Comparative Visualizations (KM curves, MAE, P-value plots) ---
    # These use `repeat_experiments_parallel` which internally handles its own multiprocessing pool
    # and sensitivity cache management for that scope of repetitions.
    print(f"\n📈 [{dataset_name_str}] Generating comparative KM, MAE, and P-value plots...")
    num_nodes_for_viz = 3 # Standard number of nodes for these overview plots
    viz_non_uniform_frac = 0.6
    viz_highly_imbalanced_frac = 0.9

    # visualize_all calls repeat_experiments_parallel internally
    visualize_km_comparisons_grid(
        main_df, partitioner_instance, analysis_methods, epsilon_values, 
        repetitions=num_repetitions, save_dir=dataset_output_dir, 
        num_nodes_for_dist=num_nodes_for_viz, 
        dist_non_uniform_major_frac=viz_non_uniform_frac, 
        dist_highly_imbalanced_major_frac=viz_highly_imbalanced_frac
    )
    # visualize_mae_with_ci calls repeat_experiments_parallel internally
    # It now returns a DataFrame of MAE runs
    mae_runs_df = visualize_mae_with_ci(
        main_df, partitioner_instance, analysis_methods, epsilon_values, 
        repetitions=num_repetitions, save_dir=dataset_output_dir, 
        num_nodes_for_dist=num_nodes_for_viz, 
        dist_non_uniform_major_frac=viz_non_uniform_frac, 
        dist_highly_imbalanced_major_frac=viz_highly_imbalanced_frac
    )
    # Save the MAE tables
    save_mae_table(mae_runs_df, dataset_name_str, base_save_dir='.')
    save_method_ranking_tables(mae_runs_df, dataset_name_str, base_save_dir='.')

    # visualize_p_values_ttest calls repeat_experiments_parallel internally
    visualize_p_values_ttest(
        main_df, partitioner_instance, analysis_methods, epsilon_values, 
        repetitions=num_repetitions, save_dir=dataset_output_dir, 
        num_nodes_for_dist=num_nodes_for_viz, 
        dist_non_uniform_major_frac=viz_non_uniform_frac, 
        dist_highly_imbalanced_major_frac=viz_highly_imbalanced_frac
    )
    # Plot showing data distribution for the visualization partitions
    visualize_data_distribution(
        main_df, partitioner_instance, save_dir=dataset_output_dir, 
        num_nodes_for_dist=num_nodes_for_viz, 
        dist_non_uniform_major_frac=viz_non_uniform_frac, 
        dist_highly_imbalanced_major_frac=viz_highly_imbalanced_frac
    )

    # --- Log-Rank Analysis for Different Partitioning Strategies ---
    print(f"\n🔬 [{dataset_name_str}] Performing Log-Rank Analysis across partitioning strategies...")
    num_nodes_logrank = 3 
    logrank_non_uniform_frac = 0.6
    logrank_highly_imbalanced_frac = 0.9

    # Define the actual partition configurations for log-rank analysis
    distributions_for_logrank = [
        ('Uniform', partitioner_instance.uniform_partition(num_nodes_logrank)),
        ('Non-uniform', partitioner_instance.nonuniform_partition(num_nodes_logrank, fraction_major=logrank_non_uniform_frac)),
        ('Highly Imbalanced', partitioner_instance.highly_imbalanced_partition(num_nodes_logrank, fraction_major=logrank_highly_imbalanced_frac))
    ]
    
    all_logrank_results_dfs = []
    all_timing_results_dfs = []

    for dist_name, actual_partitions in distributions_for_logrank:
        print(f"\n--- [{dataset_name_str}] Log-rank for {dist_name} partitioning ({num_nodes_logrank} nodes) ---")
        nodes_for_analysis = assign_data_to_nodes(actual_partitions)
        
        # perform_logrank_analysis_corrected runs serially for each (method, epsilon) config.
        # It uses the sensitivity_cache_main_process for its sensitivity calculations.
        logrank_df, timing_df = perform_logrank_analysis_corrected(
            central_df=main_df, 
            nodes=nodes_for_analysis, 
            methods=analysis_methods, 
            epsilons=epsilon_values, 
            repetitions=num_repetitions, 
            distribution_name=dist_name,
            sensitivity_cache_main_process=sensitivity_cache_main_process # Pass the main process cache
        )
        all_logrank_results_dfs.append(logrank_df)
        all_timing_results_dfs.append(timing_df)

    # Combine and save results from all distributions for this dataset
    if all_logrank_results_dfs:
        final_logrank_results_df = pd.concat(all_logrank_results_dfs, ignore_index=True)
        save_logrank_results(final_logrank_results_df, dataset_name_str, base_save_dir='.')
        save_logrank_fp_rate_table(final_logrank_results_df, dataset_name_str, base_save_dir='.')
    if all_timing_results_dfs:
        final_timing_results_df = pd.concat(all_timing_results_dfs, ignore_index=True)
        save_timing_results(final_timing_results_df, dataset_name_str, base_save_dir='.')

    overall_dataset_analysis_end_time = time()
    total_duration_dataset = overall_dataset_analysis_end_time - overall_dataset_analysis_start_time
    
    # Save overall timing for this dataset's analysis
    time_summary_path = os.path.join(dataset_output_dir, 'dataset_total_execution_time.txt')
    with open(time_summary_path, 'w') as f:
        f.write(f"Total execution time for {dataset_name_str} analysis: {total_duration_dataset:.2f} seconds\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n🏁 Finished analysis for {dataset_name_str}. Total time: {total_duration_dataset:.2f} seconds. 🏁")
    return dataset_name_str, "Analysis Completed Successfully", total_duration_dataset

def main():
    """Main function to drive the federated Kaplan-Meier analysis."""
    try:
        import tabulate # Check for tabulate, used by reporting_utils
    except ImportError:
        print("Info: 'tabulate' package not found. Text reports might be less formatted. "
              "Consider installing it via: pip install tabulate")

    print("\n========= Federated Kaplan-Meier DP Analysis (Modularized Version) =========")
    overall_start_time = time()

    # --- Load Dataset ---
    print("\n💿 Loading dataset...")
    try:
        lung_df = load_ncctg_lung_data()
        print(f"  Loaded NCCTG Lung Cancer dataset: {len(lung_df)} records.")
    except Exception as e:
        print(f"Fatal Error: Could not load dataset - {e}")
        sys.exit(1)

    # --- Analysis Configuration ---
    analysis_methods = ['Dct', 'Wavelet', 'Tv', 'Weibull'] # Updated methods
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0] # Example epsilon values
    num_repetitions = 2  # Repetitions for visualizations and log-rank statistical power

    # --- Prepare arguments for dataset analysis ---
    # Tuple: (DataFrame, output_directory_name, methods_list, epsilons_list, repetitions_count)
    dataset_args = (lung_df, "lung_cancer_results", analysis_methods, epsilon_values, num_repetitions)
    
    print(f"\n⚙️ Starting analysis for lung cancer dataset...")
    
    try:
        analyze_dataset_parallel(dataset_args)
    except Exception as e:
        print(f"\n❌ ERROR: Analysis for lung cancer dataset generated an unhandled exception: {e}")
    
    overall_end_time = time()
    total_script_duration = overall_end_time - overall_start_time
    print("\n🎉 ========= Analysis Finished ========= 🎉")
    print(f"Total script execution time: {total_script_duration:.2f} seconds.")
    
    # Summary of experimental parameters used
    num_distribution_strategies = 3 # Based on current setup (Uniform, Non-uniform, Highly Imbalanced)
    print(f"\nExperimental Setup:")
    print(f"  • Analysis Methods: {', '.join(analysis_methods)}")
    print(f"  • Privacy Budgets (ε): {epsilon_values}")
    print(f"  • Repetitions per configuration: {num_repetitions}")
    print(f"  • Distribution strategies tested: {num_distribution_strategies}")

if __name__ == '__main__':
    main()
