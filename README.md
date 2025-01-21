# Deep Bayesian Neural Network (DBNN) Implementation

The Difference Boosting Neural Network (DBNN), initially published in Intelligent Data Analysis, 4(2000) 463-473, IOS Press, is a simple yet effective Bayesian network that applies imposed conditional independence of joint probability of multiple features for classification. This implementation extends the original work with modern GPU optimization and adaptive learning capabilities.

## Key Features

### Model Types
- **Histogram Model**: Uses non-parametric density estimation with configurable bin sizes
- **Gaussian Model**: Uses multivariate normal distribution for feature pair modelling

### Adaptive Learning
Based on "What is there in a training sample?" (2009 World Congress on Nature & Biologically Inspired Computing), the implementation includes brilliant sample selection with configurable parameters:

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
        /* Basic training parameters */
        "trials": 100,                    // Number of epochs to wait for improvement
        "cardinality_threshold": 0.9,     // Threshold for feature cardinality filtering
        "cardinality_tolerance": 4,       // Decimal places for feature rounding
        "learning_rate": 0.1,             // Initial learning rate
        "random_seed": 42,                // Random seed (-1 for random shuffling)
        "epochs": 1000,                   // Maximum number of epochs
        "test_fraction": 0.2,             // Fraction of data to use for testing
        "enable_adaptive": true,          // Enable adaptive learning
        
        /* Model and computation settings */
        "modelType": "Histogram",         // Model type: "Histogram" or "Gaussian"
        "compute_device": "auto",         // "auto", "cuda", or "cpu"
        "use_interactive_kbd": false,     // Enable keyboard interaction
        "debug_enabled": true,            // Enable detailed debug logging
        
        /* Training data management */
        "Save_training_epochs": true,     // Save data for each epoch
        "training_save_path": "training_data"  // Path for saving training data
    },

    "execution_flags": {
        "train": true,                    // Enable training
        "train_only": false,              // Only perform training
        "predict": true,                  // Enable prediction
        "gen_samples": false,             // Generate sample datasets
        "fresh_start": false,             // Start fresh training
        "use_previous_model": true        // Use previously trained model if available
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
    "separator": ",",                      // CSV separator
    "has_header": true,                    // Whether file has header row
    "target_column": "target",             // Target column name or index

    /* Column configuration */
    "column_names": [                      // List of column names
        "feature1",
        "feature2",
        "#feature3",                       // Prefix with # to exclude feature
        "feature4",
        "target"
    ],

    /* Likelihood computation settings */
    "likelihood_config": {
        "feature_group_size": 2,           // Size of feature groups (usually 2)
        "max_combinations": 1000,          // Maximum feature combinations
        "bin_sizes": [20]                  // Bin sizes for histogram
    },

    /* Active learning parameters */
    "active_learning": {
        "tolerance": 1.0,                  // Learning tolerance
        "cardinality_threshold_percentile": 95,  // Percentile for cardinality threshold
        "strong_margin_threshold": 0.3,    // Threshold for strong failures
        "marginal_margin_threshold": 0.1,  // Threshold for marginal failures
        "min_divergence": 0.1             // Minimum divergence between samples
    },

    /* Training parameters specific to this dataset */
    "training_params": {
        "Save_training_epochs": true,      // Save epoch-specific data
        "training_save_path": "training_data/dataset_name"  // Dataset-specific save path
    },

    /* Model selection */
    "modelType": "Histogram"               // "Histogram" or "Gaussian"
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
