# Deep Bayesian Neural Network (DBNN) Implementation

The Difference Boosting Neural Network (DBNN), initially published in Intelligent Data Analysis, 4(2000) 463-473, IOS Press, is a simple yet effective Bayesian network that applies imposed conditional independence of joint probability of multiple features for classification. This implementation extends the original work with modern GPU optimization and adaptive learning capabilities.

## Key Features

### Model Types
- **Histogram Model**: Uses non-parametric density estimation with configurable bin sizes
- **Gaussian Model**: Uses multivariate normal distribution for feature pair modelling

### Adaptive Learning
Based on "What is there in a training sample?" (2009 World Congress on Nature & Biologically Inspired Computing), the implementation includes smart sample selection with configurable parameters:

- `active_learning_tolerance`: Controls sample selection based on probability margins
  - Range: 1.0 to 99.0 (higher means more samples selected)
  - Default: 3.0 (samples within 3% of maximum probability)
  - Example: 99.0 selects samples within 99% of the maximum probability

- `cardinality_threshold_percentile`: Controls feature complexity threshold
  - Range: 1 to 100 (lower means more samples selected)
  - Default: 95 (95th percentile)
  - Example: 75 means selecting samples below the 75th percentile of feature cardinality

### Configuration Options for adaptive_dbnn.conf

```json
{
    "training_params": {
        "trials": 100,                                  // Pateince. How many epochs before quitting?
        "debug_enabled": false,             // Set to true to enable debugging
        "debug_components": [],    // or "all" or ["data_loading", "training", etc.]
        "cardinality_threshold": 0.9, // Discards any data that repeats more than this fraction of the data - good to automatically filter out index numbers, etc.
        "cardinality_tolerance": 4,            // Precision to which data will be truncated after decimal points. -1 means no modification. 2 means round off at two decimal places
        "learning_rate": 0.1,
        "random_seed": -1,                      // -1 means random shuffle of data. A positive number means fixed data ordering.
        "epochs": 100,
        "test_fraction": 0.2,                       // Traintest fraction
        "enable_adaptive": true,             // Making it false will allow training to use the entire dataset with train/test fraction split
        "use_interactive_kbd": false,        // "Enable/disable Keyboard interaction for 'q' and 'Q' keys. Requires graphics environment."
        "compute_device": "auto",            // "cpu", "cuda", "auto"
        "modelType":   "Gaussian"           // "Histogram", "Gaussian"
    },
    "execution_flags": {
        "train": true,
        "train_only": false,
        "predict": true,
        "gen_samples": false,
        "fresh_start": false,                          // true means always start a fresh training
        "use_previous_model": true      // use the previous model on new data if fresh_start is true else, fresh training on the provided dataset
    }
}
```
### Configuration Options for sample data (for example, adult.csv from UCI)
```json
{
    "file_path": "adult.csv",  // this could also be a url like  "https://archive.ics.uci.edu/static/public/193/data.csv"
    "column_names": [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income"
    ],
    "target_column": "income",
    "separator": ",",
    "has_header": true,
    "likelihood_config": {
        "feature_group_size": 2,
        "max_combinations": 1000,
        "bin_sizes": [
            21
        ],
        "boosting_enabled": true,
        "boosting_factor": 1.5,
        "active_learning_tolerance": 1.0,  // Percentage tolerance for similar probabilities
        "cardinality_threshold_percentile": 95  // Optional, defaults to 95
        "min_divergence": 0.1  // Minimum required feature divergence between samples

}
}

```
### Feature Selection
- Add `#` before feature names in the config file to exclude them
- Automatic filtering of high cardinality features
- `cardinality_tolerance`: -1 preserves exact precision, positive number rounds to that decimal place
- `random_seed`: -1 enables data shuffling, positive number ensures reproducible splits

### GPU Optimization
- Automatic device selection with `compute_device: "auto"`
- Batch processing for memory efficiency
- Parallel likelihood computation
- Optimized tensor operations

### Model Persistence
- Saves and loads model weights
- Preserves categorical encoders
- Maintains model state between sessions
- Supports continued training with `use_previous_model`

### Visualization and Metrics
- Confusion matrices with colour coding
- Training progress plots
- Probability distribution visualizations
- Detailed classification reports
- Confidence metrics for predictions

### Interactive Training
- Press 'q' or 'Q' to skip to the next training phase (requires X11 on Linux)
- Early stopping based on error rates
- Adaptive sample selection with configurable thresholds

## Usage Examples

### Basic Training
```python
model = GPUDBNN(dataset_name='your_dataset')
results = model.fit_predict(batch_size=32)
```

### Adaptive Learning
```python
model = GPUDBNN(dataset_name='your_dataset', fresh=True)
history = model.adaptive_fit_predict(max_rounds=10)
```

### Data Format Conversion
Use `space2csv.py` to convert space-separated files to CSV:
```python
python space2csv.py input_file.txt output_file.csv
```

## Requirements
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- CUDA (optional for GPU acceleration)

## Performance Considerations
- Use GPU acceleration for large datasets
- Adjust batch size based on available memory
- Configure bin sizes based on data distribution
- Tune active learning parameters for optimal sample selection

## Error Handling
- Automatic fallback to CPU if GPU unavailable
- Robust handling of missing values
- Graceful degradation for large datasets
- Comprehensive error reporting

## Contributions and Issues
Please report any issues or contribute improvements through the project repository.

---
*Note: This implementation extends the original DBNN with modern optimizations and additional features while maintaining its core principles of simplicity and effectiveness.*
