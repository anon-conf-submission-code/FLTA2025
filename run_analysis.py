import os
import sys
from typing import Dict, Tuple
import pandas as pd

# Add script's directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.insert(0, script_dir)

# Import necessary modules
from data_handling import load_ncctg_lung_data, DataPartitioner
from plotting_utils import (
    visualize_km_comparisons_grid,
    visualize_mae_with_ci,
    visualize_p_values_ttest,
    visualize_data_distribution,
    visualize_event_distribution
)
from reporting_utils import (
    save_mae_table,
    save_method_ranking_tables,
    save_logrank_results,
    save_logrank_fp_rate_table
)
from analysis_orchestration import perform_logrank_analysis_corrected
from config import (
    ANALYSIS_METHODS,
    EPSILON_VALUES,
    NUM_REPETITIONS,
    NUM_NODES_FOR_VIZ,
    NUM_NODES_LOGRANK,
    DIST_NON_UNIFORM_MAJOR_FRAC,
    DIST_HIGHLY_IMBALANCED_MAJOR_FRAC,
    ENABLE_SENSITIVITY_CACHE
)

# Analysis options
ANALYSIS_OPTIONS: Dict[int, Tuple[str, str]] = {
    1: ('km-viz', 'Kaplan-Meier Curve Visualization'),
    2: ('mae-viz', 'Mean Absolute Error Visualization'),
    3: ('mae-tables', 'MAE Comparison Tables'),
    4: ('method-rankings', 'Method Ranking Tables'),
    5: ('pvalue-viz', 'P-value Distribution Visualization'),
    6: ('data-dist', 'Data Distribution Visualization'),
    7: ('event-dist', 'Event Distribution Visualization'),
    8: ('logrank', 'Log-rank Tests'),
    9: ('all', 'Run All Analyses')
}

def print_menu() -> None:
    """Display the analysis options menu."""
    print("\n=== Available Analyses ===")
    for key, (_, description) in ANALYSIS_OPTIONS.items():
        print(f"{key}. {description}")
    print("\n0. Exit")

def run_selected_analysis(choice: int, df: pd.DataFrame, dataset_name: str = "lung_cancer") -> None:
    """Run the selected analysis based on user choice."""
    output_dir = f"{dataset_name}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize partitioner and sensitivity cache
    partitioner = DataPartitioner(df)
    sensitivity_cache = {} if ENABLE_SENSITIVITY_CACHE else None
    
    if choice == 1:  # Kaplan-Meier Visualization
        print("\nGenerating Kaplan-Meier curve visualizations...")
        visualize_km_comparisons_grid(
            df, partitioner, ANALYSIS_METHODS, EPSILON_VALUES,
            repetitions=NUM_REPETITIONS, save_dir=output_dir
        )
        
    elif choice == 2:  # MAE Visualization
        print("\nGenerating MAE visualizations...")
        visualize_mae_with_ci(
            df, partitioner, ANALYSIS_METHODS, EPSILON_VALUES,
            repetitions=NUM_REPETITIONS, save_dir=output_dir
        )
        
    elif choice == 3:  # MAE Tables
        print("\nGenerating MAE comparison tables...")
        mae_runs = visualize_mae_with_ci(
            df, partitioner, ANALYSIS_METHODS, EPSILON_VALUES,
            repetitions=NUM_REPETITIONS, save_dir=output_dir
        )
        save_mae_table(mae_runs, dataset_name, base_save_dir='.')
        
    elif choice == 4:  # Method Rankings
        print("\nGenerating method ranking tables...")
        mae_runs = visualize_mae_with_ci(
            df, partitioner, ANALYSIS_METHODS, EPSILON_VALUES,
            repetitions=NUM_REPETITIONS, save_dir=output_dir
        )
        save_method_ranking_tables(mae_runs, dataset_name, base_save_dir='.')
        
    elif choice == 5:  # P-value Distribution
        print("\nGenerating p-value distribution visualizations...")
        visualize_p_values_ttest(
            df, partitioner, ANALYSIS_METHODS, EPSILON_VALUES,
            repetitions=NUM_REPETITIONS, save_dir=output_dir
        )
        
    elif choice == 6:  # Data Distribution
        print("\nGenerating data distribution visualizations...")
        visualize_data_distribution(
            df, partitioner, save_dir=output_dir
        )
        
    elif choice == 7:  # Event Distribution
        print("\nGenerating event distribution visualizations...")
        visualize_event_distribution(
            df, dataset_name, save_dir=output_dir
        )
        
    elif choice == 8:  # Log-rank Tests
        print("\nPerforming log-rank tests...")
        
        # List of all partitioning strategies
        distributions = [
            ('Uniform', partitioner.uniform_partition(NUM_NODES_LOGRANK)),
            ('Non-uniform', partitioner.nonuniform_partition(NUM_NODES_LOGRANK, fraction_major=DIST_NON_UNIFORM_MAJOR_FRAC)),
            ('Highly Imbalanced', partitioner.highly_imbalanced_partition(NUM_NODES_LOGRANK, fraction_major=DIST_HIGHLY_IMBALANCED_MAJOR_FRAC))
        ]
        
        all_logrank_results = []
        all_timing_results = []
        
        # Run analysis for each distribution strategy
        for dist_name, partitions in distributions:
            print(f"\nAnalyzing {dist_name} partitioning strategy...")
            nodes = {f"node_{i}": partition for i, partition in enumerate(partitions, 1)}
            logrank_results, timing_results = perform_logrank_analysis_corrected(
                df, nodes, ANALYSIS_METHODS, EPSILON_VALUES,
                NUM_REPETITIONS, dist_name, sensitivity_cache
            )
            all_logrank_results.append(logrank_results)
            all_timing_results.append(timing_results)
        
        # Combine results from all distributions
        if all_logrank_results:
            final_logrank_results = pd.concat(all_logrank_results, ignore_index=True)
            save_logrank_results(final_logrank_results, dataset_name, base_save_dir='.')
            save_logrank_fp_rate_table(final_logrank_results, dataset_name, base_save_dir='.')
        
    elif choice == 9:  # Run All Analyses
        print("\nRunning all analyses...")
        for i in range(1, 9):  # Run all analyses except 'all'
            run_selected_analysis(i, df, dataset_name)

def main():
    """Main function to run the interactive analysis menu."""
    print("\n=== Lung Cancer Dataset Analysis ===")
    
    try:
        # Load the lung cancer dataset
        print("\nLoading lung cancer dataset...")
        df = load_ncctg_lung_data()
        print(f"Dataset loaded successfully: {len(df)} records")
        
        while True:
            print_menu()
            try:
                choice = int(input("\nEnter your choice (0-9): "))
                if choice == 0:
                    print("\nExiting program...")
                    break
                elif choice in ANALYSIS_OPTIONS:
                    run_selected_analysis(choice, df)
                    input("\nPress Enter to continue...")
                else:
                    print("\nInvalid choice. Please try again.")
            except ValueError:
                print("\nInvalid input. Please enter a number between 0 and 9.")
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 