import numpy as np
from sklearn.isotonic import IsotonicRegression
from survival_core import kaplan_meier, compute_km_sensitivity
from dp_core import DP_Smoother
from config import WORKER_VERBOSE_DEBUG

# Attempt to import the new core modules
# These imports assume that dp_core.py, survival_core.py, aggregation_core.py are in python path
# This will be resolved by ensuring the main script correctly manages sys.path or package structure.

# If direct import fails, it implies a path issue for workers. 
# For now, let's assume they can be found.
from aggregation_core import Aggregator

def _internal_federated_dp_km_logic(
    nodes: dict, 
    time_points: np.ndarray, 
    method: str, 
    epsilon: float, 
    worker_sensitivity_cache: dict,
    distribution_name: str = "Unknown"
) -> np.ndarray:
    """Internal logic for computing federated DP KM curve.
    This is the core logic used by both the main process and worker processes.
    """
    if not nodes:
        return np.array([])

    federated_curves = []
    for node_name, node_df in nodes.items():
        if node_df.empty:
            if WORKER_VERBOSE_DEBUG:
                print(f"    Skipping empty node {node_name}")
            continue

        # Compute sensitivity using the worker's cache
        local_sensitivity = compute_km_sensitivity(node_df, time_points, cache=worker_sensitivity_cache)
        
        # Create and configure smoother
        smoother = DP_Smoother(method=method, epsilon=epsilon, sensitivity=local_sensitivity)
        smoother.set_progress_info(f"{distribution_name} - Node {node_name}", "Worker Process")
        
        # Compute local KM and apply DP
        km_curve_node = kaplan_meier(node_df, time_points)
        dp_curve_node = smoother.smooth(km_curve_node)
        
        # Clean up NaN and enforce bounds
        dp_curve_node = np.nan_to_num(dp_curve_node, nan=0.0)
        dp_curve_node = np.clip(dp_curve_node, 0, 1)
        
        # Ensure monotonicity
        if len(time_points) > 0 and len(dp_curve_node) == len(time_points):
            ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
            dp_curve_node = ir.fit_transform(time_points, dp_curve_node)
        
        federated_curves.append(dp_curve_node)

    if not federated_curves:
        return np.array([])

    # Average the curves
    federated_curve = np.mean(federated_curves, axis=0)
    
    # Final monotonicity enforcement on averaged curve
    if len(time_points) > 0 and len(federated_curve) == len(time_points):
        ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
        federated_curve = ir.fit_transform(time_points, federated_curve)

    return federated_curve

def execute_federated_repetition_task(
    nodes: dict, 
    time_points: np.ndarray, 
    method: str, 
    epsilon: float, 
    rep_idx: int, 
    total_reps: int, 
    worker_sensitivity_cache: dict,
    distribution_name: str = "Unknown"
) -> np.ndarray:
    """Worker function for executing a single federated DP KM repetition."""
    if WORKER_VERBOSE_DEBUG:
        print(f"  Repetition {rep_idx + 1}/{total_reps} for {method} (ε={epsilon}) starting in worker..." + 
              f" Cache ID: {id(worker_sensitivity_cache)}")
    
    result = _internal_federated_dp_km_logic(
        nodes, 
        time_points, 
        method, 
        epsilon, 
        worker_sensitivity_cache,
        distribution_name
    )
    
    if WORKER_VERBOSE_DEBUG:
        print(f"  Repetition {rep_idx + 1}/{total_reps} for {method} (ε={epsilon}) completed in worker.")
    return result 