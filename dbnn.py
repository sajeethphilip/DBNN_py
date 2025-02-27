import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Union
from collections import defaultdict
import requests
from io import StringIO
import os
import json
from itertools import combinations
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import normaltest
import numpy as np
from itertools import combinations
import torch
import os
import pickle

# Assume no keyboard control on Linux servers
nokbd = True
if os.name == 'nt' or 'darwin' in os.uname()[0].lower():  # Windows or MacOS
    try:
        from pynput import keyboard
        nokbd = False
    except:
        print("Could not initialize keyboard control")
else:
    # Check if X server is available on Linux
    def is_x11_available():
        if os.name != 'posix':  # Not Linux/Unix
            return True

        # Check if DISPLAY environment variable is set
        if 'DISPLAY' not in os.environ:
            return False

        # Optional: You could also try a lightweight X11 connection test
        try:
            import subprocess
            subprocess.run(['xset', 'q'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except:
            return False

    # Only try to import pynput if X11 is available
    nokbd = True
    if is_x11_available():
        try:
            from pynput import keyboard
            nokbd = False
        except:
            print("Could not initialize keyboard control despite X11 being available")
    else:
        print("Keyboard control using q key for skipping training is not supported without X11!")

#------------------------------------------------------------------------Declarations---------------------
Trials = 100  # Number of epochs to wait for improvement in training
cardinality_threshold =0.9
cardinality_tolerance=4 #Use when the features are likely to be extremly diverse and deciimal values;4 means, precison restricted to 4 decimal places
LearningRate =0.1
TrainingRandomSeed=42  #None # 42
Epochs=1000
TestFraction=0.2
Train=True #True #False #
Train_only=False #True #
Predict=True
Gen_Samples=False
#----------------------------------------------------------------------------------------------------------------
class DatasetConfig:
    """Handle dataset configuration loading and validation"""

    DEFAULT_CONFIG = {
        "likelihood_config": {
            "feature_group_size": 2,
            "max_combinations": 1000
        }
    }

    @staticmethod
    def _get_user_input(prompt: str, default_value: Any = None, validation_fn: callable = None) -> Any:
        """Helper method to get validated user input"""
        while True:
            if default_value is not None:
                user_input = input(f"{prompt} (default: {default_value}): ").strip()
                if not user_input:
                    return default_value
            else:
                user_input = input(f"{prompt}: ").strip()

            if validation_fn:
                try:
                    validated_value = validation_fn(user_input)
                    return validated_value
                except ValueError as e:
                    print(f"Invalid input: {e}")
            else:
                return user_input

    @staticmethod
    def _validate_boolean(value: str) -> bool:
        """Validate and convert string to boolean"""
        value = value.lower()
        if value in ('true', 't', 'yes', 'y', '1'):
            return True
        elif value in ('false', 'f', 'no', 'n', '0'):
            return False
        raise ValueError("Please enter 'yes' or 'no'")

    @staticmethod
    def _validate_int(value: str) -> int:
        """Validate and convert string to integer"""
        try:
            return int(value)
        except ValueError:
            raise ValueError("Please enter a valid integer")

    @staticmethod
    def _prompt_for_config(dataset_name: str) -> Dict:
        """Prompt user for configuration parameters"""
        print(f"\nConfiguration file for {dataset_name} not found or invalid.")
        print("Please provide the following configuration parameters:\n")

        config = {}

        # Get file path
        config['file_path'] = DatasetConfig._get_user_input(
            "Enter the path to your CSV file",
            f"{dataset_name}.csv"
        )

        # Get target column
        target_column = DatasetConfig._get_user_input(
            "Enter the name or index of the target column",
            "target"
        )
        # Convert to integer if possible
        try:
            config['target_column'] = int(target_column)
        except ValueError:
            config['target_column'] = target_column

        # Get separator
        config['separator'] = DatasetConfig._get_user_input(
            "Enter the CSV separator character",
            ","
        )

        # Get header information
        config['has_header'] = DatasetConfig._get_user_input(
            "Does the file have a header row? (yes/no)",
            "yes",
            DatasetConfig._validate_boolean
        )

        # Get likelihood configuration
        print("\nLikelihood Configuration:")
        config['likelihood_config'] = {
            'feature_group_size': DatasetConfig._get_user_input(
                "Enter the feature group size",
                2,
                DatasetConfig._validate_int
            ),
            'max_combinations': DatasetConfig._get_user_input(
                "Enter the maximum number of feature combinations",
                1000,
                DatasetConfig._validate_int
            )
        }

        return config

    @staticmethod
    def load_config(dataset_name: str) -> Dict:
        """Load configuration from file, ignoring comments starting with #"""
        config_path = f"{dataset_name}.conf"

        try:
            if os.path.exists(config_path):
                # Read existing configuration
                with open(config_path, 'r') as f:
                    # Skip lines starting with # and join remaining lines
                    config_str = ''.join(line for line in f if not line.strip().startswith('#'))

                    try:
                        config = json.load(StringIO(config_str))
                    except json.JSONDecodeError:
                        print(f"Error reading configuration file: {config_path}")
                        config = DatasetConfig._prompt_for_config(dataset_name)
            else:
                config = DatasetConfig._prompt_for_config(dataset_name)

            # Validate and ensure all required fields exist
            required_fields = ['file_path', 'target_column', 'separator', 'has_header']
            missing_fields = [field for field in required_fields if field not in config]

            if missing_fields:
                print(f"Missing required fields: {missing_fields}")
                config = DatasetConfig._prompt_for_config(dataset_name)

            # Ensure likelihood_config exists with defaults
            if 'likelihood_config' not in config:
                config['likelihood_config'] = DatasetConfig.DEFAULT_CONFIG['likelihood_config']
            else:
                # Ensure all likelihood_config fields exist
                default_likelihood = DatasetConfig.DEFAULT_CONFIG['likelihood_config']
                for key, default_value in default_likelihood.items():
                    if key not in config['likelihood_config']:
                        config['likelihood_config'][key] = default_value

            # Save the configuration
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                print(f"\nConfiguration saved to: {config_path}")

            return config

        except Exception as e:
            print(f"Error handling configuration: {str(e)}")
            return DatasetConfig._prompt_for_config(dataset_name)

    @staticmethod
    def get_available_datasets() -> List[str]:
        """Get list of available dataset configurations"""
        # Look for .conf files in the current directory
        return [f.split('.')[0] for f in os.listdir()
                if f.endswith('.conf') and os.path.isfile(f)]
#---------------------------------------Feature Filter with a #------------------------------------
def _filter_features_from_config(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter DataFrame columns based on commented features in config

    Args:
        df: Input DataFrame
        config: Configuration dictionary containing column names

    Returns:
        DataFrame with filtered columns
    """
    # If no column names in config, return original DataFrame
    if 'column_names' not in config:
        return df

    # Get column names from config
    column_names = config['column_names']

    # Create mapping of position to column name
    col_mapping = {i: name.strip() for i, name in enumerate(column_names)}

    # Identify commented features (starting with #)
    commented_features = {
        i: name.lstrip('#').strip()
        for i, name in col_mapping.items()
        if name.startswith('#')
    }

    # Get current DataFrame columns
    current_cols = df.columns.tolist()

    # Columns to drop (either by name or position)
    cols_to_drop = []

    for pos, name in commented_features.items():
        # Try to drop by name first
        if name in current_cols:
            cols_to_drop.append(name)
        # If name not found, try position
        elif pos < len(current_cols):
            cols_to_drop.append(current_cols[pos])

    # Drop identified columns
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped commented features: {cols_to_drop}")

    return df
#-----------------------------------------------------------------------------------------------------------

class GPUDBNN:
    """GPU-Optimized Deep Bayesian Neural Network with Parallel Feature Pair Processing"""

    def __init__(
        self,
        dataset_name: str,
        learning_rate: float = LearningRate,
        max_epochs: int = Epochs,
        test_size: float = TestFraction,
        random_state: int = TrainingRandomSeed,
        device: str = None
    ):
        # Automatically select device if none specified
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.dataset_name = dataset_name.lower()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size
        self.random_state = random_state

        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None

        # Categorical feature handling
        self.categorical_encoders = {}

        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load dataset configuration and data
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.data = self._load_dataset()
        self.target_column = self.config['target_column']

        # Load saved weights and encoders
        self._load_best_weights()
        self._load_categorical_encoders()

    def _calculate_cardinality_threshold(self):
        """Calculate the cardinality threshold based on the number of distinct classes"""
        #n_classes = len(self.label_encoder.classes_)
        #return 1.5 / n_classes
        return cardinality_threshold


    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        df=df.round(cardinality_tolerance)
        """
        Remove columns with unique values exceeding the threshold percentage

        Args:
            df: Input DataFrame
            threshold: Maximum allowed percentage of unique values (default: 0.8 or 80%)

        Returns:
            DataFrame with high cardinality columns removed
        """
        df_filtered = df.copy()
        columns_to_drop = []

        for column in df.columns:
            # Skip target column
            if column == self.target_column:
                continue

            # Calculate percentage of unique values
            unique_ratio = len(df[column].unique()) / len(df)

            if unique_ratio > threshold:
                columns_to_drop.append(column)

        if columns_to_drop:
            df_filtered = df_filtered.drop(columns=columns_to_drop)
            print(f"Dropped {len(columns_to_drop)} high cardinality columns: {columns_to_drop}")

        return df_filtered






    def _generate_feature_combinations(self, n_features: int, group_size: int, max_combinations: int = None) -> torch.Tensor:
        """
        Generate feature combinations of specified size

        Args:
            n_features: Total number of features
            group_size: Number of features in each group
            max_combinations: Optional maximum number of combinations to use

        Returns:
            Tensor containing feature combinations
        """
        # Generate all possible combinations
        all_combinations = list(combinations(range(n_features), group_size))

        # If max_combinations specified and less than total combinations,
        # randomly sample combinations
        if max_combinations and len(all_combinations) > max_combinations:
            import random
            random.seed(self.random_state)
            all_combinations = random.sample(all_combinations, max_combinations)

        return torch.tensor(all_combinations).to(self.device)

    def _compute_pairwise_likelihood_parallel(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Compute likelihood parameters in parallel using feature groups"""
        # Move data to GPU if available
        dataset = dataset.to(self.device)
        labels = labels.to(self.device)

        # Get likelihood configuration
        group_size = self.config.get('likelihood_config', {}).get('feature_group_size', 2)
        max_combinations = self.config.get('likelihood_config', {}).get('max_combinations', None)

        # Generate feature combinations
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            group_size,
            max_combinations
        )

        unique_classes = torch.unique(labels)

        # Initialize storage for likelihood parameters
        n_combinations = len(self.feature_pairs)
        n_classes = len(unique_classes)

        # Preallocate tensors on GPU
        means = torch.zeros((n_classes, n_combinations, group_size), device=self.device)
        covs = torch.zeros((n_classes, n_combinations, group_size, group_size), device=self.device)

        for class_idx, class_id in enumerate(unique_classes):
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]

            # Extract all feature groups in parallel
            # Reshape to handle arbitrary group sizes
            group_data = torch.stack([
                class_data[:, self.feature_pairs[i]] for i in range(n_combinations)
            ], dim=1)

            # Compute means for all groups simultaneously
            means[class_idx] = torch.mean(group_data, dim=0)

            # Compute covariances for all groups in parallel
            centered_data = group_data - means[class_idx].unsqueeze(0)

            # Update batch covariance computation for arbitrary group sizes
            for i in range(n_combinations):
                batch_cov = torch.mm(
                    centered_data[:, i].T,
                    centered_data[:, i]
                ) / (len(class_data) - 1)
                covs[class_idx, i] = batch_cov

            # Add small diagonal term for numerical stability
            covs[class_idx] += torch.eye(group_size, device=self.device) * 1e-6

        return {
            'means': means,
            'covs': covs,
            'classes': unique_classes
        }

    def _compute_batch_posterior(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Compute posterior probabilities for a batch of samples using feature groups"""
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])

        # Get group size from feature pairs shape
        group_size = self.feature_pairs.shape[1]

        # Extract groups for the batch
        batch_groups = torch.stack([
            features[:, self.feature_pairs[i]] for i in range(len(self.feature_pairs))
        ], dim=1)  # Shape: [batch_size, n_combinations, group_size]

        # Initialize log likelihoods
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for class_idx in range(n_classes):
            # Get parameters for current class
            class_means = self.likelihood_params['means'][class_idx]
            class_covs = self.likelihood_params['covs'][class_idx]
            class_priors = self.current_W[class_idx]

            # Compute mahalanobis distance for all groups in parallel
            centered = batch_groups - class_means.unsqueeze(0)

            # Compute inverse of covariance matrices
            inv_covs = torch.inverse(class_covs)

            # Compute quadratic form for all samples and groups
            quad_form = torch.zeros((batch_size, len(self.feature_pairs)), device=self.device)

            for i in range(len(self.feature_pairs)):
                quad_form[:, i] = torch.sum(
                    torch.matmul(centered[:, i, :], inv_covs[i]) * centered[:, i, :],
                    dim=1
                )

            # Compute log determinant
            log_det = torch.logdet(class_covs)

            # Compute log likelihood for all groups
            pair_log_likelihood = -0.5 * (
                group_size * np.log(2 * np.pi) +
                log_det.unsqueeze(0) +
                quad_form
            )

            # Add prior weights
            weighted_likelihood = pair_log_likelihood + torch.log(class_priors + epsilon)

            # Sum over groups for each sample
            log_likelihoods[:, class_idx] = weighted_likelihood.sum(dim=1)

        # Compute posteriors using log-sum-exp trick
        max_log_likelihood = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
        likelihoods = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors = likelihoods / (likelihoods.sum(dim=1, keepdim=True) + epsilon)

        return posteriors




    def _update_priors_parallel(self, failed_cases: List[Tuple], batch_size: int = 32):
            """Update priors in parallel for failed cases"""
            n_failed = len(failed_cases)
            n_batches = (n_failed + batch_size - 1) // batch_size

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_failed)
                batch_cases = failed_cases[start_idx:end_idx]

                # Prepare batch data
                features = torch.stack([case[0] for case in batch_cases]).to(self.device)
                true_classes = torch.tensor([case[1] for case in batch_cases]).to(self.device)

                # Compute posteriors for batch
                posteriors = self._compute_batch_posterior(features)

                # Create identity matrix on the same device
                n_classes = len(self.likelihood_params['classes'])
                eye_matrix = torch.eye(n_classes, device=self.device)

                # Compute adjustments in parallel
                true_probs = posteriors[torch.arange(len(batch_cases), device=self.device), true_classes]
                max_other_probs = torch.max(
                    posteriors * (1 - eye_matrix)[true_classes],
                    dim=1
                )[0]

                # Compute adjustments with bounds
                raw_adjustments = self.learning_rate * (1 - true_probs/max_other_probs)
                adjustments = torch.clamp(raw_adjustments, -0.5, 0.5)

                # Update weights for each class
                for class_idx, class_id in enumerate(self.likelihood_params['classes']):
                    class_mask = (true_classes == class_id)
                    if not class_mask.any():
                        continue

                    class_adjustments = adjustments[class_mask].mean()
                    self.current_W[class_idx] *= (1 + class_adjustments)
                    self.current_W[class_idx].clamp_(1e-10, 10.0)


    def predict(self, X: torch.Tensor, batch_size: int = 32):
        """Make predictions in batches using the best model weights"""
        # Store current weights temporarily
        temp_W = self.current_W

        # Use best weights for prediction
        self.current_W = self.best_W.clone() if self.best_W is not None else self.current_W

        X = X.to(self.device)
        predictions = []

        try:
            for i in range(0, len(X), batch_size):
                batch_X = X[i:min(i + batch_size, len(X))]
                posteriors = self._compute_batch_posterior(batch_X)
                batch_predictions = torch.argmax(posteriors, dim=1)
                predictions.append(batch_predictions)
        finally:
            # Restore current weights
            self.current_W = temp_W

        return torch.cat(predictions).cpu()


    def _save_best_weights(self):
        """Save the best weights to file"""
        if self.best_W is not None:
            # Convert tensor to numpy for saving
            weights_array = self.best_W.cpu().numpy()

            weights_dict = {
                'version': 2,  # Add version to track format
                'weights': weights_array.tolist(),
                'shape': list(weights_array.shape)
            }

            with open(self._get_weights_filename(), 'w') as f:
                json.dump(weights_dict, f)

    def _load_best_weights(self):
        """Load the best weights from file if they exist"""
        weights_file = self._get_weights_filename()
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights_dict = json.load(f)

            try:
                if 'version' in weights_dict and weights_dict['version'] == 2:
                    # New format (tensor-based)
                    weights_array = np.array(weights_dict['weights'])
                    self.best_W = torch.tensor(
                        weights_array,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    # Old format (dictionary-based)
                    # Convert old format to tensor format
                    class_ids = sorted([int(k) for k in weights_dict.keys()])
                    max_class_id = max(class_ids)

                    # Get number of feature pairs from first class
                    first_class = weights_dict[str(class_ids[0])]
                    n_pairs = len(first_class)

                    # Initialize tensor
                    weights_array = np.zeros((max_class_id + 1, n_pairs))

                    # Fill in weights from old format
                    for class_id in class_ids:
                        class_weights = weights_dict[str(class_id)]
                        for pair_idx, (pair, weight) in enumerate(class_weights.items()):
                            weights_array[class_id, pair_idx] = float(weight)

                    self.best_W = torch.tensor(
                        weights_array,
                        dtype=torch.float32,
                        device=self.device
                    )

                print(f"Loaded best weights from {weights_file}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_file}: {str(e)}")
                self.best_W = None

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor,batch_size: int = 32):
        """Train the model using batch processing"""
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        train_losses=[]
        test_losses=[]
        train_accuracies=[]
        test_accuracies=[]
        stop_training=False
        patience_counter = 0  # Initialize patience counter here
        # Start keyboard listener
        def on_press(key):
            nonlocal stop_training
            try:
                if key.char in ('q', 'Q'):
                    stop_training = True
            except AttributeError:
                pass
        if not nokbd:
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
        # Compute likelihood parameters
        self.likelihood_params = self._compute_pairwise_likelihood_parallel(
            X_train, y_train, X_train.shape[1]
        )

        # Initialize weights if not loaded
        if self.current_W is None:
            n_pairs = len(self.feature_pairs)
            n_classes = len(self.likelihood_params['classes'])
            self.current_W = torch.full(
                (n_classes, n_pairs),
                0.1,
                device=self.device,
                dtype=torch.float32
            )
            # If best_W not loaded, initialize it too
            if self.best_W is None:
                self.best_W = self.current_W.clone()

        n_samples = len(X_train)
        error_rates = []
        patience = Trials  # Number of epochs to wait for improvement
        for epoch in range(self.max_epochs):
            failed_cases = []
            n_errors = 0

            # Process in batches
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_X = X_train[i:batch_end]
                batch_y = y_train[i:batch_end]

                # Compute posteriors for batch
                posteriors = self._compute_batch_posterior(batch_X)
                predictions = torch.argmax(posteriors, dim=1)

                # Track failures
                failures = (predictions != batch_y)
                n_errors += failures.sum().item()

                if failures.any():
                    failed_indices = torch.where(failures)[0]
                    for idx in failed_indices:
                        failed_cases.append((
                            batch_X[idx],
                            batch_y[idx].item(),
                            posteriors[idx].cpu().numpy()
                        ))

            error_rate = n_errors / n_samples
            error_rates.append(error_rate)

            print(f"Epoch {epoch + 1}: Error rate = {error_rate:.4f}")
            # Update best weights if improved
            if error_rate <= self.best_error:
                improvement = self.best_error - error_rate
                self.best_error = error_rate
                self.best_W = self.current_W.clone()
                self._save_best_weights()
                if improvement <=0:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping with patience
            if patience_counter >= patience:
                print(f"No significant improvement for {patience} epochs. Early stopping.")
                break
            if stop_training:
                if not nokbd:
                    listener.stop()
                print("\nTraining interrupted by user")
                break

            if n_errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

            if failed_cases:
                self._update_priors_parallel(failed_cases, batch_size)
            # Calculate training metrics
            train_loss = n_errors / n_samples
            train_pred = self.predict(X_train, batch_size)
            train_acc = (train_pred == y_train.cpu()).float().mean()

            # Calculate test metrics if test data provided
            if X_test is not None and y_test is not None:
                test_pred = self.predict(X_test, batch_size)
                test_loss = (test_pred != y_test.cpu()).float().mean()
                test_acc = (test_pred == y_test.cpu()).float().mean()
            else:
                test_loss = train_loss
                test_acc = train_acc

            # Store metrics
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            # Plot metrics every epoch
            self.plot_training_metrics(
                train_losses, test_losses,
                train_accuracies, test_accuracies,
                save_path=f'{self.dataset_name}_training_metrics.png'
            )
        self._save_model_components()
        return self.current_W.cpu(), error_rates

    def plot_training_metrics(self, train_loss, test_loss, train_acc, test_acc, save_path=None):
        """Plot training and testing metrics over epochs"""
        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Train Loss', marker='o')
        plt.plot(test_loss, label='Test Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(train_acc, label='Train Accuracy', marker='o')
        plt.plot(test_acc, label='Test Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            # Also save metrics to CSV
            metrics_df = pd.DataFrame({
                'epoch': range(1, len(train_loss) + 1),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            })
            metrics_df.to_csv(save_path.replace('.png', '_metrics.csv'), index=False)

        plt.close()

    def save_predictions(self, X: pd.DataFrame, predictions: torch.Tensor, output_file: str, true_labels: pd.Series = None):
        """
        Save predictions along with input data and probabilities to a CSV file and generate visualization plots
        Args:
            X: Input DataFrame
            predictions: Model predictions
            output_file: Path to save the CSV file
            true_labels: Ground truth labels if available
        """
        # Move tensors to CPU for numpy operations
        predictions = predictions.cpu()
        # Create result DataFrame with input data
        result_df = X.copy()
        # Convert predictions to original class labels
        pred_labels = self.label_encoder.inverse_transform(predictions.numpy())
        result_df['predicted_class'] = pred_labels
        # Add ground truth if available
        if true_labels is not None:
            result_df['true_class'] = true_labels
        # Get the preprocessed features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        # Compute probabilities in batches
        batch_size = 32
        all_probabilities = []
        for i in range(0, len(X_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_tensor))
            batch_X = X_tensor[i:batch_end]
            # Get probabilities for this batch
            batch_probs = self._compute_batch_posterior(batch_X)
            all_probabilities.append(batch_probs.cpu().numpy())
        # Combine all batch probabilities
        all_probabilities = np.vstack(all_probabilities)

        # Add probability columns for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            result_df[f'prob_{class_name}'] = all_probabilities[:, i]

        # Add maximum probability column
        result_df['max_probability'] = all_probabilities.max(axis=1)

        # Get runner-up information
        top2_indices = np.argsort(all_probabilities, axis=1)[:, -2:]
        runner_up_indices = top2_indices[:, 0]  # Second highest probability
        runner_up_classes = self.label_encoder.inverse_transform(runner_up_indices)
        runner_up_probs = np.array([all_probabilities[i, idx] for i, idx in enumerate(runner_up_indices)])

        # Add runner-up information to DataFrame
        result_df['runner_up_class'] = runner_up_classes
        result_df['runner_up_probability'] = runner_up_probs

        if true_labels is not None:
            # Convert true labels to indices
            true_indices = self.label_encoder.transform(true_labels)
            # Get the probabilities for true classes
            true_probs = all_probabilities[np.arange(len(true_indices)), true_indices]
            # Check if true class has the highest probability
            max_prob_indices = np.argmax(all_probabilities, axis=1)
            correct_prediction = (true_indices == max_prob_indices)

            # Add confidence-based pass/fail label
            n_classes = len(self.label_encoder.classes_)
            confidence_threshold = 1.0 / n_classes
            result_df['confidence_check'] = np.where(
                (true_probs >= confidence_threshold) & correct_prediction,
                'Passed',
                'Failed'
            )

            # Print summary statistics
            n_failed = (result_df['confidence_check'] == 'Failed').sum()
            print(f"\nConfidence Check Summary:")
            print(f"Total predictions: {len(result_df)}")
            print(f"Failed (true class prob <= {confidence_threshold:.3f} or not max prob): {n_failed}")
            print(f"Passed (true class prob > {confidence_threshold:.3f} and is max prob): {len(result_df) - n_failed}")

            # Create confusion matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            # Create confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)

            # Create confusion matrix plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            # Save confusion matrix plot
            confusion_matrix_file = output_file.rsplit('.', 1)[0] + '_confusion_matrix.png'
            plt.savefig(confusion_matrix_file, bbox_inches='tight')
            plt.close()

            print(f"Saved confusion matrix plot to {confusion_matrix_file}")

        # Create probability distribution plots
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot 1: Overall maximum probability distribution
        ax1.hist(result_df['max_probability'], bins=50, color='lightblue', edgecolor='black')
        ax1.axvline(x=confidence_threshold, color='red', linestyle='--', label='Confidence Threshold')
        ax1.set_title('Distribution of Maximum Prediction Probabilities')
        ax1.set_xlabel('Maximum Probability')
        ax1.set_ylabel('Count')
        ax1.legend()

        # Plot 2: Distribution for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_col = f'prob_{class_name}'
            ax2.hist(result_df[prob_col], bins=50, alpha=0.5, label=class_name)

        ax2.axvline(x=confidence_threshold, color='red', linestyle='--', label='Confidence Threshold')
        ax2.set_title('Distribution of Prediction Probabilities by Class')
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Count')
        ax2.legend()

        plt.tight_layout()

        # Save the plots
        plot_file = output_file.rsplit('.', 1)[0] + '_probability_distributions.png'
        plt.savefig(plot_file)
        plt.close()

        # Save to CSV
        result_df.to_csv(output_file, index=False)
        print(f"Saved predictions with probabilities to {output_file}")
        print(f"Saved probability distribution plots to {plot_file}")

#------------------------------------------------------------End of PP code ---------------------------------------------------
    def _compute_pairwise_likelihood(self, dataset, labels, feature_dims):
        """Compute pairwise likelihood PDFs"""
        unique_classes = torch.unique(labels)
        feature_pairs = list(combinations(range(feature_dims), 2))
        likelihood_pdfs = {}

        for class_id in unique_classes:
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]
            likelihood_pdfs[class_id.item()] = {}

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.stack([
                    class_data[:, feat_i],
                    class_data[:, feat_j]
                ], dim=1)

                mean = torch.mean(pair_data, dim=0)
                centered_data = pair_data - mean
                cov = torch.mm(centered_data.T, centered_data) / (len(pair_data) - 1)
                cov = cov + torch.eye(2) * 1e-6

                likelihood_pdfs[class_id.item()][(feat_i, feat_j)] = {
                    'mean': mean,
                    'cov': cov
                }

        return likelihood_pdfs


    def _get_weights_filename(self):
        """Get the filename for saving/loading weights"""
        return os.path.join('Model', f'Best_{self.dataset_name}_weights.json')

    def _get_encoders_filename(self):
        """Get the filename for saving/loading categorical encoders"""
        return os.path.join('Model', f'Best_{self.dataset_name}_encoders.json')




    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns in the dataset"""
        categorical_columns = []
        for column in df.columns:
            if column != self.target_column:
                if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                    categorical_columns.append(column)
                elif df[column].dtype in ['int64', 'float64']:
                    # Check if the number of unique values is small relative to the dataset size
                    if len(df[column].unique()) < min(50, len(df) * 0.05):
                        categorical_columns.append(column)
        return categorical_columns



    def _preprocess_data(self, X: pd.DataFrame, is_training: bool = True) -> torch.Tensor:
        # Make a copy to avoid modifying original data
        X = X.copy()

        # Calculate cardinality threshold
        cardinality_threshold = self._calculate_cardinality_threshold()

        if is_training:
            # Store original column names before any transformations
            self.original_columns = X.columns.tolist()

            # Remove high cardinality columns during training
            X = self._remove_high_cardinality_columns(X, cardinality_threshold)
            self.feature_columns = X.columns.tolist()

            # Store high cardinality columns for future reference
            self.high_cardinality_columns = list(set(self.original_columns) - set(self.feature_columns))
        else:
            # For prediction, ensure we have the same columns as training
            if not hasattr(self, 'feature_columns'):
                raise ValueError("Model has not been trained yet - feature columns not found")

            # Remove known high cardinality columns
            if hasattr(self, 'high_cardinality_columns'):
                X = X.drop(columns=self.high_cardinality_columns, errors='ignore')

            # Ensure all required columns are present
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Reorder columns to match training order
            X = X[self.feature_columns]

        # Handle categorical features
        X_encoded = self._encode_categorical_features(X, is_training)

        # Scale the features
        try:
            if is_training:
                X_scaled = self.scaler.fit_transform(X_encoded)
            else:
                X_scaled = self.scaler.transform(X_encoded)
        except Exception as e:
            print(f"Warning: Scaling failed: {str(e)}. Using normalized data instead.")
            # Fallback to basic normalization
            X_scaled = (X_encoded - X_encoded.mean()) / (X_encoded.std() + 1e-8)

        return torch.FloatTensor(X_scaled)




    def _load_dataset(self) -> pd.DataFrame:
        """Load and preprocess dataset"""
        file_path = self.config['file_path']

        try:
            # Handle URL or local file
            if file_path.startswith(('http://', 'https://')):
                response = requests.get(file_path)
                response.raise_for_status()
                data = StringIO(response.text)
            else:
                data = file_path

            # Read CSV with appropriate parameters
            read_params = {
                'sep': self.config['separator'],
                'header': 0 if self.config['has_header'] else None,
                'names': self.config.get('column_names'),
            }

            df = pd.read_csv(data, **read_params)

            # Filter features based on config
            df = _filter_features_from_config(df, self.config)

            # Handle target column
            if isinstance(self.config['target_column'], int):
                # If target column is specified by index
                cols = df.columns.tolist()
                self.config['target_column'] = cols[self.config['target_column']]

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")


    def _multivariate_normal_pdf(self, x, mean, cov):
        """Compute multivariate normal PDF"""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        dim = 2
        centered_x = x - mean.unsqueeze(0)
        inv_cov = torch.inverse(cov)
        det = torch.det(cov)
        quad_form = torch.sum(torch.mm(centered_x, inv_cov) * centered_x, dim=1)
        norm_const = 1.0 / (torch.sqrt((2 * torch.tensor(np.pi)) ** dim * det))
        return norm_const * torch.exp(-0.5 * quad_form)

    def _initialize_priors(self):
        """Initialize weights"""
        if self.best_W is not None:
            return self.best_W

        W = {}
        for class_id in self.likelihood_pdfs.keys():
            W[class_id] = {}
            for feature_pair in self.likelihood_pdfs[class_id].keys():
                W[class_id][feature_pair] = torch.tensor(0.1, dtype=torch.float32)
        return W

    def compute_posterior(self, feature_data, class_id=None, epsilon=1e-10):
        """Compute posterior probabilities"""
        classes = list(self.likelihood_pdfs.keys())
        n_classes = len(classes)
        feature_pairs = list(self.likelihood_pdfs[classes[0]].keys())
        log_likelihoods = torch.zeros(n_classes, dtype=torch.float32)

        for idx, c_id in enumerate(classes):
            class_log_likelihood = 0.0

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.tensor([
                    feature_data[feat_i].item(),
                    feature_data[feat_j].item()
                ], dtype=torch.float32).reshape(1, 2)

                pdf_params = self.likelihood_pdfs[c_id][(feat_i, feat_j)]
                pair_likelihood = self._multivariate_normal_pdf(
                    pair_data,
                    pdf_params['mean'],
                    pdf_params['cov']
                ).squeeze()

                prior = self.current_W[c_id][(feat_i, feat_j)].item()
                likelihood_term = (pair_likelihood * prior + epsilon).item()
                class_log_likelihood += torch.log(torch.tensor(likelihood_term))

            log_likelihoods[idx] = class_log_likelihood

        max_log_likelihood = torch.max(log_likelihoods)
        likelihoods = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors = likelihoods / (likelihoods.sum() + epsilon)

        return {c_id: posteriors[idx].item() for idx, c_id in enumerate(classes)}


    def _update_priors(self, failed_cases, epsilon=1e-10):
        """Update priors with numerical stability measures"""
        W_updates = defaultdict(lambda: defaultdict(list))

        for features, true_class, posteriors in failed_cases:
            P1 = posteriors[true_class] + epsilon
            P2 = max((p for c, p in posteriors.items() if c != true_class), default=epsilon)
            P2 = P2 + epsilon

            raw_adjustment = self.learning_rate * (1 - P1/P2)
            adjustment = max(min(raw_adjustment, 0.5), -0.5)

            for feature_pair in self.likelihood_pdfs[true_class].keys():
                W_updates[true_class][feature_pair].append(adjustment)

        for class_id in W_updates:
            for feature_pair in W_updates[class_id]:
                updates = W_updates[class_id][feature_pair]
                avg_update = sum(updates) / len(updates)

                current_w = self.current_W[class_id][feature_pair].item()
                new_w = current_w * (1 + avg_update)
                new_w = max(min(new_w, 10.0), epsilon)

                self.current_W[class_id][feature_pair] = torch.tensor(new_w, dtype=torch.float32)


    def fit_predict(self, batch_size: int = 32, save_path: str = None):
        """
        Full training and prediction pipeline with GPU optimization and optional prediction saving

        Args:
            batch_size: Batch size for training and prediction
            save_path: Path to save predictions CSV file. If None, predictions won't be saved
        """
        # Prepare data
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        y_encoded = self.label_encoder.fit_transform(y)

        # Preprocess features including categorical encoding
        X_processed = self._preprocess_data(X, is_training=True)

        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor.cpu().numpy(),
            y_tensor.cpu().numpy(),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_tensor.cpu().numpy()
        )

        # Convert split data back to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)

        # Train model
        final_W, error_rates = self.train(X_train, y_train,X_test, y_test, batch_size=batch_size)

        # Save categorical encoders
        self._save_categorical_encoders()

        # Make predictions
        y_pred = self.predict(X_test, batch_size=batch_size)

        # Save predictions if path is provided
        if save_path:
            # Get corresponding rows from original DataFrame for test set
            X_test_indices = range(len(X_test))
            X_test_df = X.iloc[X_test_indices]
            y_test_series = y.iloc[X_test_indices]

            self.save_predictions(X_test_df, y_pred, save_path, y_test_series)

        # Calculate metrics
        y_test_cpu = y_test.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()

        # Convert numerical labels back to original classes
        y_test_labels = self.label_encoder.inverse_transform(y_test_cpu)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_cpu)

        # Prepare results
        results = {
            'classification_report': classification_report(y_test_labels, y_pred_labels),
            'confusion_matrix': confusion_matrix(y_test_labels, y_pred_labels),
            'error_rates': error_rates,
            'test_accuracy': (y_pred_cpu == y_test_cpu).mean()
        }

        print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
        self._save_model_components()
        return results

    def _get_model_components_filename(self):
        """Get filename for model components"""
        return os.path.join('Model', f'Best_{self.dataset_name}_components.pkl')
#----------------Handling categorical variables across sessions -------------------------
    def _save_categorical_encoders(self):
        """Save categorical feature encoders"""
        if self.categorical_encoders:
            # Create a serializable dictionary structure
            encoders_dict = {
                'encoders': {
                    column: {
                        str(k): v for k, v in mapping.items()
                    } for column, mapping in self.categorical_encoders.items()
                }
            }

            # Add metadata
            if hasattr(self, 'original_columns'):
                if isinstance(self.original_columns, list):
                    column_types = {col: str(self.data[col].dtype) for col in self.original_columns if col in self.data.columns}
                else:
                    column_types = {col: str(dtype) for col, dtype in self.original_columns.items()}

                encoders_dict['metadata'] = {
                    'column_types': column_types,
                    'timestamp': pd.Timestamp.now().isoformat()
                }

            with open(self._get_encoders_filename(), 'w') as f:
                json.dump(encoders_dict, f, indent=2)

    def _load_categorical_encoders(self):
        """Load categorical feature encoders from file"""
        encoders_file = self._get_encoders_filename()
        if os.path.exists(encoders_file):
            try:
                with open(encoders_file, 'r') as f:
                    data = json.load(f)

                # Extract encoders from the loaded data
                if 'encoders' in data:
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data['encoders'].items()
                    }
                else:
                    # Handle legacy format where encoders were at top level
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data.items()
                    }

                print(f"Loaded categorical encoders from {encoders_file}")
            except Exception as e:
                print(f"Warning: Failed to load categorical encoders: {str(e)}")
                self.categorical_encoders = {}

    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True):
        df_encoded = df.copy()
        categorical_columns = self._detect_categorical_columns(df)

        for column in categorical_columns:
            if is_training:
                if column not in self.categorical_encoders:
                    unique_values = df[column].unique()
                    self.categorical_encoders[column] = {
                        value: idx for idx, value in enumerate(unique_values)
                    }
            mapping = self.categorical_encoders[column]
            df_encoded[column] = df[column].map(lambda x: mapping.get(x, -1))

        return df_encoded

#--------------------------------------------------------------------------------------------------------------

    def _save_model_components(self):
            """Save all necessary model components"""
            components = {
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'likelihood_params': self.likelihood_params,
                'feature_pairs': self.feature_pairs,
                'categorical_encoders': self.categorical_encoders,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'target_classes': self.label_encoder.classes_,
                'target_mapping': dict(zip(self.label_encoder.classes_,
                                         range(len(self.label_encoder.classes_)))),
                'config': self.config,
                'high_cardinality_columns': getattr(self, 'high_cardinality_columns', []),
                'original_columns': getattr(self, 'original_columns', None),
                'best_error': getattr(self, 'best_error', float('inf'))
            }

            with open(self._get_model_components_filename(), 'wb') as f:
                pickle.dump(components, f)



    def _load_model_components(self):
        """Load all model components"""
        components_file = self._get_model_components_filename()
        if os.path.exists(components_file):
            with open(components_file, 'rb') as f:
                components = pickle.load(f)
                self.label_encoder.classes_ = components['target_classes']
                self.scaler = components['scaler']
                self.label_encoder = components['label_encoder']
                self.likelihood_params = components['likelihood_params']
                self.feature_pairs = components['feature_pairs']
                self.feature_columns = components.get('feature_columns')
                self.categorical_encoders = components['categorical_encoders']
                self.high_cardinality_columns = components.get('high_cardinality_columns', [])
                print(f"Loaded model components from {components_file}")
                return True



    def predict_and_save(self, save_path=None, batch_size: int = 32):
        """Make predictions on data and save them using best model weights"""
        try:
            # First try to load existing model and components
            weights_loaded = os.path.exists(self._get_weights_filename())
            components_loaded = self._load_model_components()

            if not (weights_loaded and components_loaded):
                print("Complete model not found. Training required.")
                results = self.fit_predict(batch_size=batch_size)
                return results

            # Load the model weights and encoders
            self._load_best_weights()
            self._load_categorical_encoders()

            # Explicitly use best weights for prediction
            if self.best_W is None:
                print("No best weights found. Training required.")
                results = self.fit_predict(batch_size=batch_size)
                return results

            # Store current weights temporarily
            temp_W = self.current_W

            # Use best weights for prediction
            self.current_W = self.best_W.clone()

            try:
                # Load and preprocess input data
                X = self.data.drop(columns=[self.target_column])
                true_labels = self.data[self.target_column]

                # Preprocess the data using the existing method
                X_tensor = self._preprocess_data(X, is_training=False)

                # Make predictions
                predictions = self.predict(X_tensor, batch_size=batch_size)

                # Save predictions and metrics
                if save_path:
                    self.save_predictions(X, predictions, save_path, true_labels)

                return predictions
            finally:
                # Restore current weights
                self.current_W = temp_W

        except Exception as e:
            print(f"Error during prediction process: {str(e)}")
            print("Falling back to training pipeline...")
            results = self.fit_predict(batch_size=batch_size)
            return results

def run_gpu_benchmark(dataset_name: str, batch_size: int = 32):
    """Run benchmark using GPU-optimized implementation"""
    print(f"\nRunning GPU benchmark on {dataset_name} dataset...")

    model = GPUDBNN(
        dataset_name=dataset_name,
        learning_rate=0.01,
        max_epochs=100
    )

    results = model.fit_predict(batch_size=batch_size)

    plot_training_progress(results['error_rates'], dataset_name)
    plot_confusion_matrix(
        results['confusion_matrix'],
        model.label_encoder.classes_,
        dataset_name
    )

    print(f"\nClassification Report for {dataset_name}:")
    print(results['classification_report'])

    return model, results
def run_benchmark(dataset_name: str):
    """Run benchmark on specified dataset"""
    print(f"\nRunning benchmark on {dataset_name} dataset...")

    model = DBNN(
        dataset_name=dataset_name,
        learning_rate=0.01,
        max_epochs=100
    )

    results = model.fit_predict()

    plot_training_progress(results['error_rates'], dataset_name)
    plot_confusion_matrix(
        results['confusion_matrix'],
        model.label_encoder.classes_,
        dataset_name
    )

    print(f"\nClassification Report for {dataset_name}:")
    print(results['classification_report'])

    return model, results

def plot_training_progress(error_rates: List[float], dataset_name: str):
    """Plot training error rates over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title(f'Training Progress - {dataset_name.capitalize()} Dataset')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(confusion_mat: np.ndarray, class_names: np.ndarray, dataset_name: str):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {dataset_name.capitalize()} Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()




def generate_test_datasets():
    """Generate XOR and 3D XOR test datasets"""
    # Generate 2D XOR
    with open('xor.csv', 'w') as f:
        f.write('x1,x2,target\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')

    # Generate 3D XOR
    with open('xor3d.csv', 'w') as f:
        f.write('x1,x2,x3,target\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')


if __name__ == "__main__":
    if Gen_Samples:

            # Generate test datasets
            generate_test_datasets()

            # Test XOR and 3D XOR
            test_datasets = ['xor', 'xor3d']
            for dataset in test_datasets:
                try:
                    print(f"\nTesting {dataset} dataset...")
                    model = GPUDBNN(dataset_name=dataset)
                    if Train:
                        model, results = run_gpu_benchmark(dataset)
                        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
                    if Predict:
                        predictions = model.predict_and_save(save_path=f"{dataset}_predictions.csv")
                except Exception as e:
                    print(f"Error testing {dataset}: {str(e)}")

            # Print available datasets
            print("Available datasets:", ", ".join(DatasetConfig.get_available_datasets()))

    # Example: Run benchmark on available datasets
    datasets_to_test = DatasetConfig.get_available_datasets()

    for dataset in datasets_to_test:
        #try:
            model = GPUDBNN(dataset_name=dataset)
            if Train:
                model, results = run_gpu_benchmark(dataset)
            if Train_only:
                results = model.fit_predict(save_path=f"{dataset}_train_test_predictions.csv")
            if Predict:
                predictions = model.predict_and_save(save_path=f"{dataset}_predictions.csv")
            print(f"\nCompleted benchmark for {dataset}")
            print("-" * 50)
        #except:
         #   print(f"Skipping data {dataset} as it ran into error while training/testing..")
