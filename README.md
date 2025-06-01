# Federated Differential Privacy for Kaplan-Meier Analysis

This project implements a federated learning approach for privacy-preserving Kaplan-Meier survival analysis. It combines differential privacy techniques with distributed computing to enable collaborative survival analysis while protecting individual patient privacy.

## Features

- **Federated Learning**: Supports distributed computation across multiple nodes
- **Differential Privacy**: Multiple smoothing methods for privacy preservation:
  - Discrete Cosine Transform (DCT)
  - Wavelet Transform
  - Total Variation (TV) Smoothing
  - Parametric Weibull Fitting
- **Adaptive Privacy**: Data-size aware privacy mechanisms
- **Comprehensive Analysis**:
  - KM curve visualization
  - Mean Absolute Error (MAE) analysis
  - Statistical significance testing
  - Privacy parameter analysis
- **Flexible Data Distribution**:
  - Uniform partitioning
  - Non-uniform partitioning
  - Highly imbalanced scenarios

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd FLTA2025_paper_submission
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `dp_core.py`: Core differential privacy implementations
- `fedKMDP.py`: Main federated learning orchestration
- `analysis_orchestration.py`: Analysis workflow management
- `parallel_logic.py`: Parallel processing utilities
- `plotting_utils.py`: Visualization functions
- `reporting_utils.py`: Results reporting and analysis
- `data_handling.py`: Data loading and partitioning
- `config.py`: Centralized configuration management

## Usage

1. Basic Analysis:
```python
python run_analysis.py
```

2. Configuration:
Edit `config.py` to modify:
- Privacy parameters (ε values)
- Node configurations
- Analysis settings
- Visualization preferences

## Privacy Methods

### Total Variation (TV) Smoothing
- Adaptive regularization based on data size
- Three-component lambda calculation:
  1. Base regularization (TV_LAMBDA_BASE)
  2. Data size scaling (n/TV_REF_SIZE)^TV_SCALING_POWER
  3. Theoretical adjustment sqrt(log(n+1))

### Other Methods
- **DCT**: Frequency domain privacy preservation
- **Wavelet**: Multi-resolution analysis with privacy
- **Weibull**: Parametric approach with differential privacy

## Analysis Outputs

The analysis generates:
1. Visualization plots:
   - KM curve comparisons
   - MAE analysis with confidence intervals
   - P-value distributions
   - Data distribution visualizations

2. Statistical reports:
   - MAE tables
   - Log-rank test results
   - Method ranking tables
   - TV parameter analysis

3. Performance metrics:
   - Execution timing
   - Privacy parameter effects
   - Node distribution impact

## Configuration

Key configuration parameters in `config.py`:

```python
# Privacy Parameters
TV_LAMBDA_BASE = 0.12      # Base TV smoothing strength
TV_REF_SIZE = 50          # Reference dataset size
TV_SCALING_POWER = 0.25   # Scaling power for data size adjustment

# Analysis Settings
VERBOSE_DEBUG = False     # Enable/disable detailed logging
SEED_OFFSET = 42         # Random seed offset
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Citation

If you use this code in your research, please cite:
```
[Citation information will be added after publication]
```

## License

[License information]

## Acknowledgments

- NCCTG Lung Cancer dataset
- [Other acknowledgments] 