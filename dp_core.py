from __future__ import annotations
import numpy as np
from typing  import Optional, Dict
from scipy.fftpack       import dct, idct
from scipy.optimize      import curve_fit
import pywt
from config import (
    TV_LAMBDA_BASE,
    TV_REF_SIZE,
    TV_SCALING_POWER
)
################################################################################
# ─────────────────────────── helper algorithms ───────────────────────────────
################################################################################
def _tv1d_denoise(y: np.ndarray, lam: float) -> np.ndarray:
    """
    Condat's fast 1-D total-variation denoiser -- IEEE SPL 2013.
    Pure-numpy, O(n).
    """
    n = y.size
    if n == 0 or lam <= 0:
        return y.copy()

    x      = np.empty_like(y)
    k      = k0 = 0
    vmin   = y[0] - lam
    vmax   = y[0] + lam
    umin   = lam
    umax   = -lam

    for i in range(1, n):
        d = y[i] - vmin
        if d < umin:
            while k0 < i:
                x[k0] = vmin
                k0   += 1
            vmin = y[i] - lam
            vmax = y[i] + lam
            umin = lam
            umax = -lam
            k    = i
        else:
            umin += d
            d     = y[i] - vmax
            if d > umax:
                while k0 < i:
                    x[k0] = vmax
                    k0   += 1
                vmin = y[i] - lam
                vmax = y[i] + lam
                umin = lam
                umax = -lam
                k    = i
            else:
                umin += d
                umax += d
                if umin >= lam:
                    vmin += (umin - lam)/(i - k + 1)
                    umin  = lam
                if umax <= -lam:
                    vmax += (umax + lam)/(i - k + 1)
                    umax  = -lam
    x[k0:] = (vmin + vmax) / 2.0
    return x

def _laplace_noise(shape, sensitivity, epsilon):
    """
    Generate Laplace noise for differential privacy.
    
    Args:
        shape: Shape of the output noise array
        sensitivity: Sensitivity of the function
        epsilon: Privacy parameter
        
    Returns:
        Laplace noise array of specified shape
    """
    # Increased minimum scale for better numerical stability
    b = max(sensitivity/epsilon, 1e-6)
    return np.random.laplace(0.0, b, size=shape)

def _clip_monotone(curve: np.ndarray) -> np.ndarray:
    curve = np.clip(curve, 0, 1)
    return np.minimum.accumulate(curve)

def _adaptive_tv_lambda(n: int) -> float:
    """
    Returns a data-size–aware total-variation regularisation weight.
    
    The formula λ(n) = TV_LAMBDA_BASE * (n/TV_REF_SIZE)^TV_SCALING_POWER * sqrt(log(n+1)) has three components:
    1. TV_LAMBDA_BASE (0.12): Base regularization strength
       - Controls basic trade-off between smoothing and noise
       - Range 0.05-0.2 (lower = less smoothing, higher = more smoothing)
    
    2. (n/TV_REF_SIZE)^TV_SCALING_POWER: Sub-linear scaling with data size
       - n/TV_REF_SIZE normalizes to reference size (default 50)
       - TV_SCALING_POWER (default 0.25) ensures very gentle scaling
       - Examples:
         * n=10:  (10/50)^0.25  ≈ 0.63 (reduces smoothing)
         * n=50:  (50/50)^0.25  = 1.00 (reference point)
         * n=200: (200/50)^0.25 ≈ 1.41 (increases smoothing)
    
    3. sqrt(log(n+1)): Theoretical adjustment
       - Common in non-parametric statistics
       - Provides theoretical convergence guarantees
       - Examples:
         * n=10:  sqrt(log(11))  ≈ 1.20
         * n=50:  sqrt(log(51))  ≈ 1.48
         * n=200: sqrt(log(201)) ≈ 1.66
    
    Combined effect for TV_LAMBDA_BASE = 0.12:
    - Small data (n=10):  λ ≈ 0.12 * 0.63 * 1.20 ≈ 0.091
    - Medium data (n=50): λ ≈ 0.12 * 1.00 * 1.48 ≈ 0.178
    - Large data (n=200): λ ≈ 0.12 * 1.41 * 1.66 ≈ 0.281
    
    Args:
        n: Size of the dataset
    
    Returns:
        Adaptive regularization parameter λ(n)
    """
    return TV_LAMBDA_BASE * (n / TV_REF_SIZE)**TV_SCALING_POWER * np.sqrt(np.log(n + 1))

def analyze_tv_parameters(n: int) -> Dict:
    """
    Analyzes the components of the TV lambda calculation for a given data size.
    
    Args:
        n: Size of the dataset to analyze
    
    Returns:
        Dictionary with detailed breakdown of parameter components:
        {
            'n': Input dataset size,
            'base_value': Base TV lambda value,
            'size_scaling': Scaling factor based on data size,
            'theory_adjustment': Theoretical adjustment factor,
            'final_lambda': Final computed lambda value,
            'components': {
                'base_effect': Effect of base value alone,
                'scaling_effect': Combined effect of base and scaling,
                'total_effect': Total combined effect
            }
        }
    """
    base = TV_LAMBDA_BASE
    scaling = (n / TV_REF_SIZE)**TV_SCALING_POWER
    theory = np.sqrt(np.log(n + 1))
    total = base * scaling * theory
    
    return {
        'n': n,
        'base_value': base,
        'size_scaling': scaling,
        'theory_adjustment': theory,
        'final_lambda': total,
        'components': {
            'base_effect': base,
            'scaling_effect': base * scaling,
            'total_effect': total
        }
    }

################################################################################
# ───────────────────────────── main class ────────────────────────────────────
################################################################################
class DP_Smoother:
    """
    Differential-privacy preserving smoothers for discrete survival curves.

    Available methods:
        • "DCT"     – discrete cosine transform + DP noise  
        • "Wavelet" – Haar wavelet shrinkage + DP noise  
        • "TV"      – total-variation denoising + DP noise  
        • "Weibull" – parametric Weibull fit   + DP noise
    """
    def __init__(self,
                 method      : str   = "DCT",
                 epsilon     : float = 1.0,
                 sensitivity : float = 1.0,
                 tv_base_weight   : Optional[float] = None):
        m = method.title()
        if m not in {"Dct", "Wavelet", "Tv", "Weibull"}:
            raise ValueError(f"Unsupported method '{method}'")
        if epsilon <= 0:
            raise ValueError("ε must be > 0")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        self.method      = m
        self.epsilon     = float(epsilon)
        self.sensitivity = float(sensitivity)
        self.tv_base_weight = float(tv_base_weight) if tv_base_weight is not None else TV_LAMBDA_BASE
        # Initialize tracking attributes
        self._current_partition = "Unknown"
        self._current_repetition = "Unknown"

    def set_progress_info(self, partition_type: str, repetition_info: str) -> None:
        """Sets the current partition type and repetition information for progress tracking."""
        self._current_partition = partition_type
        self._current_repetition = repetition_info

    # ────────────────────── public entry point ────────────────────────────
    def smooth(self, curve: np.ndarray) -> np.ndarray:
        try:
            s = np.asarray(curve, dtype=float).copy()
            if s.ndim == 0:
                s = s[None]
            if s.size == 0:
                raise ValueError("empty curve")

            # ---------- dispatch ----------
            if   self.method == "Dct":
                smoothed = self._smooth_dct(s)
            elif self.method == "Wavelet":
                smoothed = self._smooth_wavelet(s)
            elif self.method == "Tv":
                smoothed = self._smooth_tv(s)
            elif self.method == "Weibull":
                smoothed = self._smooth_weibull(s)
            # No else needed due to __init__ validation

            return _clip_monotone(smoothed).astype(np.float32)
        except Exception as e:
            dist_strategy = "Unknown"
            part_info = getattr(self, '_current_partition', 'Unknown')
            rep_info = getattr(self, '_current_repetition', 'Unknown')

            if "Uniform" in part_info: dist_strategy = "Uniform"
            elif "Non-uniform" in part_info: dist_strategy = "Non-uniform"
            elif "Highly Imbalanced" in part_info: dist_strategy = "Highly Imbalanced"
            
            err_msg = (
                f"{self.method} smoothing failed: {e} | "
                f"Distribution: {dist_strategy} | Nodes/Settings: {part_info} | Repetition: {rep_info} | "
                f"Input curve size: {curve.size if hasattr(curve, 'size') else 'N/A'}"
            )
            raise RuntimeError(err_msg) from e

    # ───────────────────── individual smoothers ───────────────────────────
    def _smooth_dct(self, s):
        coeff  = dct(s, norm="ortho")
        coeff += _laplace_noise(coeff.shape, self.sensitivity, self.epsilon)
        return idct(coeff, norm="ortho")

    def _smooth_wavelet(self, s):
        if s.size < 2:
            raise ValueError("Wavelet needs ≥2 samples")
        coeffs = pywt.wavedec(s, "haar")
        coeffs = [c + _laplace_noise(c.shape, self.sensitivity, self.epsilon)
                  for c in coeffs]
        return pywt.waverec(coeffs, "haar")[: s.size]

    def _smooth_tv(self, s: np.ndarray) -> np.ndarray:
        """
        Total-variation denoising with differential privacy via Laplace noise.
        
        This implementation uses:
        1. Adaptive regularization λ(n) that scales with data size
        2. Optional ε-splitting strategy (disabled by default):
           - 25% of ε for protecting the TV optimization
           - 75% of ε for the final noise addition
        3. Direct noise addition to maintain privacy guarantees
        4. Monotonicity enforcement via _clip_monotone
        
        The implementation ensures ε-differential privacy through:
        - Adaptive TV regularization to control sensitivity
        - Laplace noise calibrated to curve sensitivity
        - Optional composition of privacy budgets if splitting is enabled
        
        Args:
            s: Input survival curve as numpy array
            
        Returns:
            Differentially private smoothed survival curve
        """
        n = s.size
        lam = _adaptive_tv_lambda(n)
        
        # Configuration for epsilon splitting (currently disabled)
        use_epsilon_split = False  # Set to True to enable split strategy
        if use_epsilon_split:
            pre_eps = 0.25 * self.epsilon  # For protecting TV optimization
            post_eps = 0.75 * self.epsilon  # For final noise
        else:
            pre_eps = 0  # No pre-noise
            post_eps = self.epsilon  # All budget for final noise
        
        # Optional: Protect the TV optimization itself
        if use_epsilon_split:
            s = s + _laplace_noise(s.shape, self.sensitivity, pre_eps)
        
        # Apply TV denoising
        smooth = _tv1d_denoise(s, lam)
        
        # Add noise directly to maintain privacy guarantees
        smooth += _laplace_noise(s.shape, self.sensitivity, post_eps)
        
        # Ensure curve starts at 1.0 (survival probability at time 0)
        if smooth.size:
            smooth += 1 - smooth[0]
        
        # Enforce monotonicity constraint for survival curves
        smooth = _clip_monotone(smooth)
        
        return smooth

    def _smooth_weibull(self, s):
        x      = np.arange(1, s.size + 1, dtype=float)
        y      = np.clip(s, 1e-6, 1-1e-6)
        ln_t   = np.log(x)
        ln_lnS = np.log(-np.log(y))
        mask   = np.isfinite(ln_lnS)
        
        # Ensure there are enough finite points for polyfit
        if np.sum(mask) < 2: # polyfit needs at least 2 points for degree 1
            # Fallback if not enough points for polyfit
            smooth = np.minimum.accumulate(y[::-1])[::-1] # Monotone non-parametric
        else:
            try:
                k_hat, ln_inv_scale = np.polyfit(ln_t[mask], ln_lnS[mask], 1)
                k_hat = max(k_hat, 1e-3) # Ensure k_hat is positive
                lam   = np.exp(-ln_inv_scale/k_hat) if k_hat != 0 else np.inf # Avoid division by zero
                
                if np.isinf(lam): # Further fallback if lambda is infinite
                     smooth = np.minimum.accumulate(y[::-1])[::-1]
                else:
                     smooth = np.exp(-(x/lam)**k_hat)

            except Exception: # Catch any other error during polyfit/calculation
                # fallback: monotone non-parametric
                smooth = np.minimum.accumulate(y[::-1])[::-1]
                
        smooth += _laplace_noise(smooth.shape, self.sensitivity, self.epsilon)
        return smooth