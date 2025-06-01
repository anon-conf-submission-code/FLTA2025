import numpy as np
from typing import List

class Aggregator:
    """Aggregates multiple survival curves into a single federated curve."""
    
    @staticmethod
    def mean_aggregation(curves: List[np.ndarray]) -> np.ndarray:
        """Aggregates curves by taking the element-wise mean.

        Args:
            curves: A list of numpy arrays, where each array is a survival curve.

        Returns:
            A numpy array representing the mean aggregated survival curve.
            Returns an empty array if the input list is empty or curves are invalid.
        """
        if not curves: # Handle empty list of curves
            return np.array([], dtype=np.float32)
        
        # Ensure all curves are numpy arrays and convert if necessary
        processed_curves = []
        for curve in curves:
            if not isinstance(curve, np.ndarray):
                try:
                    processed_curves.append(np.asarray(curve, dtype=np.float32))
                except Exception:
                    # If conversion fails, skip this curve or handle as error
                    # print(f"Warning: Could not convert curve to np.ndarray: {curve}")
                    continue # Skip this problematic curve
            else:
                processed_curves.append(curve.astype(np.float32))
        
        if not processed_curves: # If all curves were invalid
            return np.array([], dtype=np.float32)
            
        # Check for consistent lengths if desired, though np.mean might broadcast or error
        # For safety, let's assume all curves should have the same length. 
        # If not, this mean aggregation might not be meaningful as is.
        # For now, np.mean will raise error if shapes are incompatible for axis=0 mean.
        try:
            mean_curve = np.mean(processed_curves, axis=0)
            return mean_curve.astype(np.float32)
        except ValueError as e:
            # This can happen if curves have different lengths.
            # print(f"Error during mean aggregation: {e}. Curves might have different lengths.")
            # Fallback: return an empty array or the first curve, or re-raise
            # Returning empty array for now.
            return np.array([], dtype=np.float32) 