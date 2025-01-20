The Difference Boosting Neural Network (DBNN) published in  Intelligent Data Analysis, 4(2000) 463-473, IOS Press, nl is a simple Bayesian network that applies an imposed conditional independence of joint probability of multiple features
for the classification of the feature space overcoming the limitations of the Naive Bayesian networks while at the same time maintaining its simplicity. The original code was in C++ and was used only by a selected group of people. 
The Python version is a rewrite in just a few lines of code with examples from the UCI repository that can be downloaded and used to create training/test data to evaluate the model quickly. A manual is in preparation.

You can generate config files for UCI data (that can be used as an example) and get the required data for evaluation using the Generate and Get functions. NOTE: You do not need to download the data unless you want to look at it in detail. The config data has the path to the file and can be directly used by the code to get the data.

Add a # to the configure file features before the feature names to exclude any feature from the computation. By default, any repetitions with high cardinality will be filtered out.

By default, the Imposed conditional independence is assumed on pairs of features ("feature_group_size": 2) but can be edited in the conf file to any number equal/less than the total number of features.

Press the "q" or "Q" key during training to interrupt training and go to the next stage. However, in Linux, this works only on machines where you have an X11 window.

The older version of the C++ code used space as the separator, which is not ideal. space2csv.py function can convert the file into CSV.
Let me break down the key components of this GPU-optimized Deep Bayesian Neural Network (DBNN) implementation.
## Sample configuration file for data

{

    "file_path": "cardiotocography.csv", # Also can be the URL like  "file_path": "https://archive.ics.uci.edu/static/public/193/data.csv",
    "target_column": "NSP", # Also can give column_names if desired as "column_names":[...]
    "separator": ",",
    "has_header": true,
    "likelihood_config": {
        "feature_group_size": 2,
        "max_combinations": 1000,
        "bin_sizes": [64],  # Use 64 bins uniformly
        # or use variable bins:
        # "bin_sizes": [3, 7, 13, 21],
        "boosting_enabled": true,
        "boosting_factor": 1.5
        "active_learning_tolerance": 1.0,  // Percentage tolerance for similar probabilities. A higher value means more samples are added during adaptive learning.
        "cardinality_threshold_percentile": 95  // Optional, defaults to 95. A lower value means more samples are added during adaptive learning.
    }
}


## Core Components

**Dataset Configuration**
- Handles dataset loading and configuration through JSON files
- Validates required fields and provides default configurations
- Supports both local and URL-based dataset loading
- Filters features based on configuration settings

**Data Preprocessing**
- Handles categorical feature encoding
- Removes high cardinality columns based on threshold
- Applies standard scaling to numerical features
- Preserves feature order and handles missing columns

**Model Architecture**
The DBNN implementation includes:

**Likelihood Computation**
- Computes multivariate normal PDFs for feature pairs
- Uses GPU-optimized tensor operations for parallel processing
- Maintains numerical stability through epsilon values

**Training Process**
1. Initializes weights for each class and feature pair
2. Computes posterior probabilities in batches
3. Updates weights based on misclassifications
4. Implements early stopping based on error rates

## Supporting Features

**Model Persistence**
- Saves and loads model weights
- Stores categorical encoders
- Preserves model components between sessions

**Visualization**
- Generates confusion matrices
- Plots training progress
- Creates probability distribution visualizations

**Performance Optimization**
- Utilizes GPU acceleration when available
- Implements batch processing for memory efficiency
- Provides parallel computation of likelihoods

## Utility Functions

**Benchmarking**
- Runs performance tests on datasets
- Generates test datasets (XOR, 3D XOR)
- Reports classification metrics

**Error Handling**
- Validates input data
- Provides fallback mechanisms
- Implements graceful degradation when GPU is unavailable

The code follows a modular design pattern with a clear separation of concerns between data handling, model training, and evaluation components.

# Adaptive Learning 
### (What is there in a training sample?, 2009 World Congress on Nature & Biologically Inspired Computing (NaBIC), Coimbatore, India, 2009, pp. 1507-1511, doi: 10.1109/NABIC.2009.5393682.)

The adaptive learning process in the code follows a specific strategy for including new examples based on misclassifications. Here's how it works:

## Initial Training Set Creation
The process starts by selecting one example per class from the dataset to form the initial training set. This ensures the representation of all classes from the beginning.
### cardinality_tolerance": -1 
means the exact resolution of the input data will be used for analysis. If set to a positive number, the decimals will be rounded to that precision. 
###   "random_seed": -1,
means the input data will be randomised before analysis. A positive number can be used to get the exact data split for testing and training.
## Adaptive Learning Loop
The key components of the adaptive learning process are:

**Training Phase**
1. The model trains on the current training set
2. The remaining data becomes the test set (using a boolean mask: `test_mask = ~train_mask`)
3. Predictions are made on this test set

**Selection Process**
For each misclassified example in the test set, the code:
1. Group misclassifications by predicted class
2. For each class where misclassifications occurred, it selects the example with the highest prediction probability
3. These selected examples are then added to the training set for the next iteration

**Stopping Criteria**
The process continues until either:
- All examples are correctly classified
- No improvement is seen for 3 consecutive rounds
- The maximum number of rounds is reached

## Code Implementation
Here's the key section that handles the selection:

```python
# Find misclassified examples
misclassified_mask = (predictions != true_labels)
misclassified_indices = np.where(misclassified_mask)[0]

# Select the most confident misclassifications
new_train_indices = []
for class_idx, class_label in enumerate(unique_classes):
    # Find misclassified examples predicted as this class
    class_mask = (predictions == class_label) & misclassified_mask
    if np.any(class_mask):
        # Get probabilities for predicted class
        class_probs = predictions_df[prob_col].values[class_mask]
        max_prob_idx = np.argmax(class_probs)
        # Map back to original dataset index
        original_idx = test_indices[np.where(class_mask)[0][max_prob_idx]]
        new_train_indices.append(original_idx)
```

This confirms that adaptive learning is indeed happening on the test set (data not used for training), as evidenced by:
1. The explicit creation of test indices using the complement of training mask
2. The selection of new examples only from the test set indices
3. The mapping back to original dataset indices when adding new examples to the training set
