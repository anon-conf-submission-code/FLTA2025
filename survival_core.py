import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# Global cache for sensitivity results within this module (if used by functions here)
# This is distinct from the main process cache or worker caches.
sensitivity_cache_survival_core = {}

def kaplan_meier(df: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
    """Calculates the Kaplan-Meier survival curve.

    Args:
        df: DataFrame with 'duration' and 'event' columns.
        time_points: Numpy array of time points to predict survival probabilities.

    Returns:
        Numpy array of survival probabilities at the given time points.
    """
    if df.empty or len(df['duration']) == 0:
        return np.ones_like(time_points, dtype=np.float32) # All survive if no data

    kmf = KaplanMeierFitter()
    try:
        kmf.fit(df['duration'], df['event'])
        # Predict for each time point; handle cases where predict might return scalar for single t
        survival = np.array([kmf.predict(t) for t in time_points], dtype=np.float32)
    except Exception: # Broad exception for fitting/prediction issues with unusual data
        # print(f"Warning: KaplanMeierFitter error. DF shape: {df.shape}. Returning baseline survival.")
        survival = np.ones_like(time_points, dtype=np.float32)
    return survival

def compute_km_sensitivity(
    df: pd.DataFrame, 
    time_points: np.ndarray, 
    cache: dict = None
) -> float:
    """Computes the sensitivity of Kaplan-Meier estimates by leaving one sample out.

    Uses a provided cache to store and retrieve sensitivity results for identical inputs.
    If no cache is provided, a local default cache (`sensitivity_cache_survival_core`) is used.

    Args:
        df: DataFrame with 'duration' and 'event' columns.
        time_points: Numpy array of time points for KM curve evaluation.
        cache: Optional dictionary to use for caching. 

    Returns:
        Maximum L-infinity norm difference (sensitivity).
    """
    if df.empty:
        return 0.0

    # Determine which cache to use
    active_cache = cache if cache is not None else sensitivity_cache_survival_core

    # Create a robust cache key
    # Sort DataFrame by duration and event to ensure consistency for cache key
    df_sorted_tuple = tuple(map(tuple, df.sort_values(by=['duration', 'event']).reset_index(drop=True).to_numpy().tolist()))
    time_points_tuple = tuple(time_points)
    cache_key = (df_sorted_tuple, time_points_tuple)

    if cache_key in active_cache:
        return active_cache[cache_key]

    # Original KM curve on the full dataset
    base_curve = kaplan_meier(df, time_points)
    
    sensitivities = []
    n_samples = len(df)
    
    if n_samples == 0:
        active_cache[cache_key] = 0.0
        return 0.0

    for i in range(n_samples):
        mod_df = df.drop(df.index[i]).reset_index(drop=True)
        
        if mod_df.empty:
            # If removing one sample makes df empty, perturbed curve is full survival/no events
            modified_curve = np.ones_like(base_curve, dtype=np.float32)
        else:
            modified_curve = kaplan_meier(mod_df, time_points)
        
        sensitivity = np.max(np.abs(base_curve - modified_curve))
        sensitivities.append(sensitivity)
    
    max_sensitivity = np.max(sensitivities) if sensitivities else 0.0
    active_cache[cache_key] = max_sensitivity
    return max_sensitivity

def generate_dp_surrogate_df(
    time_points_grid: np.ndarray, 
    dp_curve_on_grid: np.ndarray, 
    original_node_df: pd.DataFrame
) -> pd.DataFrame:
    """Generates a surrogate dataset from a DP survival curve using inverse transform sampling.

    Args:
        time_points_grid: The time points over which dp_curve_on_grid is defined.
        dp_curve_on_grid: The differentially private survival probabilities on the grid.
        original_node_df: The original DataFrame for a node, used for sample size and event rate.

    Returns:
        A pandas DataFrame with 'duration' and 'event' columns for the surrogate data.
    """
    n_samples = len(original_node_df)
    if n_samples == 0:
        return pd.DataFrame({'duration': [], 'event': []}, dtype=np.float32)

    dp_curve_values = np.asarray(dp_curve_on_grid, dtype=np.float32)
    time_grid = np.asarray(time_points_grid, dtype=np.float32)

    # CDF = 1 - Survival Function. Must be non-decreasing for np.interp.
    # dp_curve_values should be non-increasing (enforced by IsotonicRegression elsewhere).
    cdf_values = 1.0 - dp_curve_values

    # Ensure cdf_values are monotonically non-decreasing for np.interp.
    # This should hold if dp_curve_values is non-increasing.
    # np.interp expects xp (cdf_values here) to be non-decreasing.
    # Add a small epsilon ramp if strictly increasing is needed, but usually not for np.interp.
    for i in range(1, len(cdf_values)):
        if cdf_values[i] < cdf_values[i-1]:
            cdf_values[i] = cdf_values[i-1] # Enforce monotonicity if not already perfect

    random_uniform_draws = np.random.uniform(0, 1, size=n_samples)
    
    # Interpolate surrogate durations from the CDF (inverse transform sampling)
    if len(time_grid) == 0 or len(cdf_values) == 0:
        # Fallback if curve is empty: use original mean duration or 0 if original is empty
        mean_duration = original_node_df['duration'].mean() if not original_node_df.empty else 0
        surrogate_durations = np.full(n_samples, mean_duration, dtype=np.float32)
    elif len(time_grid) == 1:
        # If only one time point in grid, all durations are that time point
        surrogate_durations = np.full(n_samples, time_grid[0], dtype=np.float32)
    else:
        surrogate_durations = np.interp(random_uniform_draws, cdf_values, time_grid)
    
    # Preserve original event rate for generating surrogate events
    if original_node_df.empty or original_node_df['event'].empty:
        original_event_rate = 0.5 # Default if no info or no events in original
    else:
        original_event_rate = original_node_df['event'].mean()
        if not (0 <= original_event_rate <= 1):
            original_event_rate = np.clip(original_event_rate, 0, 1) # Ensure valid probability

    surrogate_events = np.random.binomial(1, original_event_rate, size=n_samples).astype(np.int32)

    return pd.DataFrame({
        'duration': surrogate_durations.astype(np.float32),
        'event': surrogate_events
    }) 