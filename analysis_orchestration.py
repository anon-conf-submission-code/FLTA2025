import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from lifelines.statistics import logrank_test
from sklearn.isotonic import IsotonicRegression
from multiprocessing import Pool, Manager, cpu_count
from time import time

# Configuration and core logic imports
from config import (
    VERBOSE_DEBUG,
    WORKER_VERBOSE_DEBUG,
    SEED_OFFSET,
    TIME_POINTS_COUNT,
    MAX_WORKERS
)
from parallel_logic import execute_federated_repetition_task
from survival_core import kaplan_meier, compute_km_sensitivity, generate_dp_surrogate_df
# DP_Smoother will be used within perform_logrank_analysis_corrected directly from dp_core
from dp_core import DP_Smoother 

# Global cache for sensitivity results in this module, if needed for non-parallel parts.
# perform_logrank_analysis_corrected uses its own instance of sensitivity_cache_main_process (from fedKMDP.py)
# or rather, it should now use the one from survival_core or be passed one.
# For now, let it use the compute_km_sensitivity from survival_core which handles its own caching or accepts a cache.

def compute_confidence_intervals(curves: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes mean curve and confidence intervals from repeated curve estimations."""
    if curves.ndim == 1: # Handle case where only one curve might be passed accidentally
        # Or if repeat_experiments_parallel somehow returns a 1D array for 1 repetition
        # This indicates an issue upstream if curves is not 2D for multiple repetitions.
        # For now, treat as a single sample for CI (lower=upper=mean=curve)
        if curves.size == 0:
             return np.array([]), np.array([]), np.array([])
        return curves, curves, curves 

    if curves.shape[0] == 0: # No curves to process
        # Determine length of a typical curve if possible, else return empty arrays of appropriate shape
        # Assuming time_points length could be inferred or curves is (0, num_time_points)
        num_time_points = curves.shape[1] if curves.ndim > 1 and curves.shape[1] > 0 else 0
        empty_curve = np.array([]).reshape(0, num_time_points) if num_time_points > 0 else np.array([])
        return empty_curve, empty_curve, empty_curve
        
    lower_bound = np.percentile(curves, (1 - confidence) / 2 * 100, axis=0)
    upper_bound = np.percentile(curves, (1 + confidence) / 2 * 100, axis=0)
    mean_curve = np.mean(curves, axis=0)
    return mean_curve, lower_bound, upper_bound

def compute_p_values(original_curve: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """Computes p-values using paired t-test between original curve and set of curves."""
    if original_curve.size == 0 or curves.size == 0 or curves.shape[1] != original_curve.shape[0]:
        return np.array([])
    if curves.ndim == 1 : # if only one curve in curves, reshape for iteration
        curves = curves.reshape(1, -1)
    if curves.shape[0] == 0: # No curves to compare against
        return np.array([])

    p_values = []
    for i in range(len(original_curve)):
        # Ensure there are enough samples for ttest_rel (at least 2)
        if curves.shape[0] < 2:
            p_values.append(np.nan) # Not enough data for t-test
            continue
        try:
            # ttest_rel requires two arrays of same shape. Here, original_curve[i] is scalar.
            # We compare a list of [original_curve[i], original_curve[i], ...] vs curves[:, i]
            _, p = ttest_rel([original_curve[i]] * curves.shape[0], curves[:, i], nan_policy='omit')
            p_values.append(p)
        except ValueError: # Catches issues like not enough data after nan_policy='omit'
            p_values.append(np.nan)
    return np.array(p_values)

def repeat_experiments_parallel(
    nodes: dict, 
    time_points: np.ndarray, 
    method: str, 
    epsilon: float, 
    repetitions: int,
    external_sensitivity_cache: dict = None,
    distribution_name: str = "Unknown"
) -> np.ndarray:
    """Runs experiments in parallel using multiprocessing.Pool.
    Manages a shared sensitivity cache for the duration of this function call if not provided externally.
    """
    if VERBOSE_DEBUG:
        print(f"\nOrchestrator: Running {method} experiments (ε={epsilon}) with {repetitions} reps (multiprocessing.Pool)")

    if repetitions == 0:
        if VERBOSE_DEBUG:
            print(f"Orchestrator: No repetitions for {method} (ε={epsilon}).")
        return np.array([])
    
    # Decide on cache: use external if provided, else create a new Manager().dict()
    if external_sensitivity_cache is not None:
        shared_cache_for_this_run = external_sensitivity_cache
        manager_context = None
    else:
        manager = Manager()
        shared_cache_for_this_run = manager.dict()
        manager_context = manager

    tasks_args = [
        (nodes, time_points, method, epsilon, i, repetitions, shared_cache_for_this_run, distribution_name) 
        for i in range(repetitions)
    ]

    num_workers = min(cpu_count() if MAX_WORKERS is None else MAX_WORKERS, repetitions) if repetitions > 0 else 1
    if num_workers == 0: num_workers = 1

    results = []
    if tasks_args:
        try:
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(execute_federated_repetition_task, tasks_args)
        finally:
            if manager_context:
                manager_context.shutdown()
        
    if VERBOSE_DEBUG:
        print(f"Orchestrator: Completed {repetitions} reps for {method} (ε={epsilon}).")
    return np.array(results, dtype=np.float32)

def perform_logrank_analysis_corrected(
    central_df: pd.DataFrame, 
    nodes: dict, 
    methods: list[str], 
    epsilons: list[float], 
    repetitions: int, 
    distribution_name: str,
    sensitivity_cache_main_process: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Performs log-rank analysis, comparing centralized data to federated surrogate data.
    Generates node-specific DP curves for surrogate data generation.
    Uses a sensitivity cache passed from the main process for non-parallel part.
    """
    results_list = []
    timing_results_list = []
    time_points_logrank = np.linspace(0, central_df['duration'].max(), TIME_POINTS_COUNT, dtype=np.float32)
    
    total_configurations = len(methods) * len(epsilons)
    print(f"\nPerforming log-rank tests for {distribution_name}. Repetitions per config: {repetitions}")
    print(f"Total configurations: {total_configurations}")

    for method_idx, current_method in enumerate(methods):
        for epsilon_idx, current_epsilon in enumerate(epsilons):
            config_seed_base = SEED_OFFSET + method_idx * len(epsilons) + epsilon_idx + hash(distribution_name) % 1000
            
            print(f"\nProcessing {distribution_name} - Method: {current_method}, ε={current_epsilon} (Seed Base: {config_seed_base})")
            config_start_time = time()
            p_values_for_config_repetitions = []
            analyzed_reps_count = 0

            for rep_idx in range(repetitions):
                rep_seed = config_seed_base + rep_idx
                np.random.seed(rep_seed)
                print(f"  Repetition {rep_idx + 1}/{repetitions} for {current_method} (ε={current_epsilon}) starting (seed: {rep_seed})...")
                
                federated_surrogates_for_this_rep = []
                all_nodes_empty_for_this_rep = True

                for node_name, node_df in nodes.items():
                    if node_df.empty:
                        if VERBOSE_DEBUG:
                            print(f"    Skipping empty node {node_name} for rep {rep_idx + 1}")
                        continue
                    all_nodes_empty_for_this_rep = False # Found at least one non-empty node

                    # Sensitivity calculation using the main process cache passed as argument
                    local_sensitivity = compute_km_sensitivity(node_df, time_points_logrank, cache=sensitivity_cache_main_process)
                    
                    smoother = DP_Smoother(method=current_method, epsilon=current_epsilon, sensitivity=local_sensitivity)
                    # Set progress info for better error tracking
                    smoother.set_progress_info(f"{distribution_name} - Node {node_name}", f"Rep {rep_idx + 1}/{repetitions}")
                    
                    km_curve_node = kaplan_meier(node_df, time_points_logrank)
                    dp_curve_node = smoother.smooth(km_curve_node)
                    
                    dp_curve_node = np.nan_to_num(dp_curve_node, nan=0.0)
                    dp_curve_node = np.clip(dp_curve_node, 0, 1)
                    if len(time_points_logrank) > 0 and len(dp_curve_node) == len(time_points_logrank):
                        ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
                        dp_curve_node = ir.fit_transform(time_points_logrank, dp_curve_node)
                    
                    surrogate_node_df = generate_dp_surrogate_df(time_points_logrank, dp_curve_node, node_df)
                    if not surrogate_node_df.empty:
                        federated_surrogates_for_this_rep.append(surrogate_node_df)
                
                if all_nodes_empty_for_this_rep or not federated_surrogates_for_this_rep:
                    print(f"  Repetition {rep_idx + 1}/{repetitions} completed (no surrogate data generated). Skipping log-rank.")
                    p_values_for_config_repetitions.append(np.nan)
                    continue

                combined_surrogate_df = pd.concat(federated_surrogates_for_this_rep, ignore_index=True)
                if combined_surrogate_df.empty or combined_surrogate_df['duration'].empty:
                    print(f"  Repetition {rep_idx + 1}/{repetitions} completed (combined surrogate empty). Skipping log-rank.")
                    p_values_for_config_repetitions.append(np.nan)
                    continue
                
                try:
                    test_result = logrank_test(
                        durations_A=central_df['duration'],
                        durations_B=combined_surrogate_df['duration'],
                        event_observed_A=central_df['event'],
                        event_observed_B=combined_surrogate_df['event']
                    )
                    p_values_for_config_repetitions.append(test_result.p_value)
                    analyzed_reps_count += 1
                    print(f"  Repetition {rep_idx + 1}/{repetitions} completed. p-value: {test_result.p_value:.4e}")
                except Exception as e:
                    print(f"  Repetition {rep_idx + 1}/{repetitions} log-rank test error: {e}. Appending NaN.")
                    p_values_for_config_repetitions.append(np.nan)

            # Summarize p-values for the current configuration
            p_values_array = np.array(p_values_for_config_repetitions)
            mean_p = np.nanmean(p_values_array) if analyzed_reps_count > 0 else np.nan
            median_p = np.nanmedian(p_values_array) if analyzed_reps_count > 0 else np.nan
            prop_significant = np.nansum(p_values_array < 0.05) / analyzed_reps_count if analyzed_reps_count > 0 else np.nan
            
            config_end_time = time()
            config_time_taken = config_end_time - config_start_time
            
            results_list.append({
                'Partitioning Strategy': distribution_name,
                'DP Method': current_method,
                'Epsilon': current_epsilon,
                'Mean Log-rank p-value': mean_p,
                'Median Log-rank p-value': median_p,
                'Proportion Significant (p<0.05)': prop_significant,
                'Repetitions Analyzed': analyzed_reps_count,
                'Total Repetitions Run (config)': repetitions,
                'Number of Nodes': len(nodes)
            })
            timing_results_list.append({
                'Partitioning Strategy': distribution_name,
                'DP Method': current_method,
                'Epsilon': current_epsilon,
                'Total Execution Time (s)': config_time_taken, 
                'Avg Time per Repetition (s)': config_time_taken / repetitions if repetitions > 0 else 0,
                'Number of Nodes': len(nodes),
                'Total Repetitions Run (config)': repetitions
            })
            print(f"Completed {current_method} (ε={current_epsilon}) for {distribution_name} in {config_time_taken:.2f}s. \n" + 
                  f"  Mean p: {mean_p:.4e}, Median p: {median_p:.4e}, Prop Sig: {prop_significant:.2%}, Analyzed Reps: {analyzed_reps_count}")

    return pd.DataFrame(results_list), pd.DataFrame(timing_results_list) 