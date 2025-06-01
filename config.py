# Configuration settings for the Federated DP Kaplan-Meier Analysis

# Debug and logging settings
VERBOSE_DEBUG = False  # Set to True for detailed logging
WORKER_VERBOSE_DEBUG = False  # Debug logging for worker processes

# Reproducibility settings
SEED_OFFSET = 42  # Base seed for random number generation

# Analysis Configuration
ANALYSIS_METHODS = ['Dct', 'Wavelet', 'Tv', 'Weibull']  # DP smoothing methods
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]  # Privacy budget values
NUM_REPETITIONS = 100  # Number of repetitions for visualizations and log-rank statistical power

# Node Configuration
NUM_NODES_FOR_VIZ = 3  # Standard number of nodes for overview plots
NUM_NODES_LOGRANK = 3  # Number of nodes for log-rank analysis

# Data Distribution Parameters
DIST_NON_UNIFORM_MAJOR_FRAC = 0.6  # Fraction for non-uniform distribution
DIST_HIGHLY_IMBALANCED_MAJOR_FRAC = 0.9  # Fraction for highly imbalanced distribution

# Visualization Settings
TIME_POINTS_COUNT = 100  # Number of time points for KM curves
PLOT_DPI = 300  # DPI for saved plots
PLOT_FIGSIZE = (26, 18)  # Figure size for plots

# Algorithm Parameters
TV_LAMBDA_BASE = 0.12  # Base value for adaptive TV lambda calculation
# TV smoothing formula: λ(n) = TV_LAMBDA_BASE * (n/TV_REF_SIZE)^TV_SCALING_POWER * sqrt(log(n+1))
TV_REF_SIZE = 50      # Reference dataset size for TV smoothing normalization
TV_SCALING_POWER = 0.25  # Power for sub-linear scaling (1/4 for gentle scaling)
# - TV_LAMBDA_BASE = 0.12: empirically determined base strength
# - (n/TV_REF_SIZE)^TV_SCALING_POWER: sub-linear scaling with data size
# - sqrt(log(n+1)): theoretical adjustment from non-parametric statistics
# Smaller values (0.05-0.1) = less smoothing, more noise
# Larger values (0.15-0.2) = more smoothing, less noise

# Performance Settings
MAX_WORKERS = None  # None means use all available CPU cores

# Cache Settings
ENABLE_SENSITIVITY_CACHE = True  # Whether to use sensitivity caching

# You can add more configuration parameters here as needed:
# Example:
# MAX_WORKERS = None  # None means use all available CPU cores
# CACHE_SIZE = 1000   # Maximum number of entries in sensitivity cache 
