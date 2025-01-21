import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import json
from typing import List, Optional, Dict
import pickle

class EpochVisualizer:
    def __init__(self, config_file: str):
        """
        Initialize visualizer using configuration files.

        Args:
            config_file: Path to the dataset's .conf file
        """
        self.config_file = config_file
        self.dataset_name = os.path.splitext(os.path.basename(config_file))[0]

        # Load configurations
        self.dataset_config = self._load_dataset_config()
        self.global_config = self._load_global_config()

        # Get paths from config
        training_params = self.dataset_config.get('training_params', {})
        self.base_training_path = os.path.join(
            training_params.get('training_save_path', 'training_data'),
            self.dataset_name
        )
        self.base_viz_path = os.path.join('visualizations', self.dataset_name)

        # Load and preprocess data
        self.full_data = self._load_and_preprocess_data()

        # Create visualization directory
        os.makedirs(self.base_viz_path, exist_ok=True)

        print(f"Training data path: {self.base_training_path}")
        print(f"Visualization path: {self.base_viz_path}")

    def _load_dataset_config(self) -> dict:
        """Load dataset-specific configuration."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_text = f.read()

            # Remove comments (but preserve URLs)
            lines = []
            for line in config_text.split('\n'):
                if '//' in line and not ('http://' in line or 'https://' in line):
                    line = line.split('//')[0]
                if line.strip():
                    lines.append(line)

            clean_config = '\n'.join(lines)
            return json.loads(clean_config)

        except Exception as e:
            print(f"Error loading dataset config: {str(e)}")
            return None

    def _load_global_config(self) -> dict:
        """Load global configuration from adaptive_dbnn.conf."""
        try:
            with open('adaptive_dbnn.conf', 'r', encoding='utf-8') as f:
                config_text = f.read()

            # Remove comments
            lines = []
            for line in config_text.split('\n'):
                if '//' in line:
                    line = line.split('//')[0]
                if line.strip():
                    lines.append(line)

            clean_config = '\n'.join(lines)
            return json.loads(clean_config)

        except Exception as e:
            print(f"Error loading global config: {str(e)}")
            return None

    def _remove_high_cardinality_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove high cardinality columns based on threshold from config."""
        if not self.global_config:
            return df

        threshold = self.global_config.get('training_params', {}).get('cardinality_threshold', 0.9)
        df_filtered = df.copy()
        columns_to_drop = []
        target_column = self.dataset_config['target_column']

        # Keep track of non-target feature columns
        feature_columns = [col for col in df.columns if col != target_column]

        for column in feature_columns:
            unique_count = len(df[column].unique())
            unique_ratio = unique_count / len(df)

            if unique_ratio > threshold:
                columns_to_drop.append(column)
                print(f"Dropping high cardinality column: {column} (ratio: {unique_ratio:.3f})")

        # Check if we would drop all feature columns
        remaining_features = [col for col in feature_columns if col not in columns_to_drop]

        if not remaining_features:
            # Keep the top 3 features with lowest cardinality if we would drop everything
            feature_cardinality = [(col, len(df[col].unique()) / len(df))
                                 for col in feature_columns]
            sorted_features = sorted(feature_cardinality, key=lambda x: x[1])
            columns_to_keep = [col for col, _ in sorted_features[:3]]

            print("\nWARNING: All features would be dropped due to high cardinality.")
            print("Keeping top 3 features with lowest cardinality ratios:")
            for col, ratio in sorted_features[:3]:
                print(f"- {col} (ratio: {ratio:.3f})")

            columns_to_drop = [col for col in feature_columns if col not in columns_to_keep]

        if columns_to_drop:
            df_filtered = df_filtered.drop(columns=columns_to_drop)

        # Verify we still have features to work with
        remaining_features = [col for col in df_filtered.columns if col != target_column]
        print(f"\nRemaining features after cardinality filtering: {len(remaining_features)}")
        print("Features:", remaining_features)

        return df_filtered

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset according to configurations."""
        if not self.dataset_config:
            raise ValueError("Dataset configuration not loaded")

        # Load data
        file_path = self.dataset_config['file_path']
        try:
            if file_path.startswith(('http://', 'https://')):
                print(f"Loading data from URL: {file_path}")
                df = pd.read_csv(file_path)
            else:
                print(f"Loading data from file: {file_path}")
                df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

        # Apply column names if specified
        if 'column_names' in self.dataset_config:
            df.columns = self.dataset_config['column_names']

        # Remove high cardinality columns
        df = self._remove_high_cardinality_columns(df)

        return df

    def _load_epoch_indices(self, epoch: int) -> tuple:
        """Load training and testing indices for given epoch."""
        epoch_dir = os.path.join(self.base_training_path, f'epoch_{epoch}')
        model_type = self.global_config.get('training_params', {}).get('modelType', 'Histogram')

        try:
            with open(os.path.join(epoch_dir, f'{model_type}_train_indices.pkl'), 'rb') as f:
                train_indices = pickle.load(f)
            with open(os.path.join(epoch_dir, f'{model_type}_test_indices.pkl'), 'rb') as f:
                test_indices = pickle.load(f)
            return train_indices, test_indices
        except FileNotFoundError:
            print(f"No data found for epoch {epoch} in {epoch_dir}")
            return None, None

    def  create_visualizations(self, epoch: int):
        """Create visualizations for a specific epoch."""
        # Load indices for this epoch
        train_indices, test_indices = self._load_epoch_indices(epoch)
        if train_indices is None:
            return

        # Create epoch visualization directory
        epoch_viz_dir = os.path.join(self.base_viz_path, f'epoch_{epoch}')
        os.makedirs(epoch_viz_dir, exist_ok=True)

        # Split data into train and test
        train_data = self.full_data.iloc[train_indices]
        test_data = self.full_data.iloc[test_indices]

        target_column = self.dataset_config['target_column']

        print(f"\nCreating visualizations for epoch {epoch}:")
        print(f"Training set size: {len(train_indices)}")
        print(f"Test set size: {len(test_indices)}")

        # Create visualizations for both sets
        self._create_epoch_visualizations(train_data, 'train', target_column, epoch_viz_dir)
        self._create_epoch_visualizations(test_data, 'test', target_column, epoch_viz_dir)

        print(f"Created visualizations for epoch {epoch} in {epoch_viz_dir}")

    def _create_epoch_visualizations(self, data: pd.DataFrame, set_type: str,
                                   target_column: str, save_dir: str):
        """Create all visualizations for one dataset."""
        feature_cols = [col for col in data.columns if col != target_column]

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[feature_cols])

        # 1. t-SNE 2D
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(scaled_features)

        tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df[target_column] = data[target_column]

        fig_2d = px.scatter(tsne_df, x='TSNE1', y='TSNE2', color=target_column,
                           title=f't-SNE 2D Projection - {set_type} set')
        fig_2d.write_html(os.path.join(save_dir, f'tsne_2d_{set_type}.html'))

        # 2. t-SNE 3D
        tsne = TSNE(n_components=3, random_state=42)
        tsne_result = tsne.fit_transform(scaled_features)

        tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2', 'TSNE3'])
        tsne_df[target_column] = data[target_column]

        fig_3d = px.scatter_3d(tsne_df, x='TSNE1', y='TSNE2', z='TSNE3',
                              color=target_column,
                              title=f't-SNE 3D Projection - {set_type} set')
        fig_3d.write_html(os.path.join(save_dir, f'tsne_3d_{set_type}.html'))

        # 3. Feature pairs 3D scatter (first 3 features)
        if len(feature_cols) >= 3:
            fig_3d_feat = px.scatter_3d(data, x=feature_cols[0], y=feature_cols[1],
                                      z=feature_cols[2], color=target_column,
                                      title=f'First 3 Features - {set_type} set')
            fig_3d_feat.write_html(os.path.join(save_dir, f'features_3d_{set_type}.html'))

        # 4. Parallel coordinates
        fig_parallel = px.parallel_coordinates(data, dimensions=feature_cols,
                                            color=target_column,
                                            title=f'Parallel Coordinates - {set_type} set')
        fig_parallel.write_html(os.path.join(save_dir, f'parallel_coords_{set_type}.html'))

def main():
    # Get user input
    config_file = input("Enter the name of your dataset configuration file (e.g., dataset.conf): ")

    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found!")
        return

    visualizer = EpochVisualizer(config_file)

    # Check if training data exists and epochs are saved
    dataset_name = os.path.splitext(os.path.basename(config_file))[0]
    training_path = os.path.join(visualizer.base_training_path)

    if not os.path.exists(training_path):
        print(f"No training data found at {training_path}")
        return

    # Get available epochs
    epoch_dirs = [d for d in os.listdir(training_path) if d.startswith('epoch_')]
    if not epoch_dirs:
        print(f"No epoch data found in {training_path}")
        return

    print(f"\nFound {len(epoch_dirs)} epochs of training data")
    epoch_input = input("Enter epoch number (or press Enter for all epochs): ")

    try:
        if epoch_input.strip():
            # Visualize specific epoch
            epoch = int(epoch_input)
            epoch_dir = f'epoch_{epoch}'
            if epoch_dir not in epoch_dirs:
                print(f"No data found for epoch {epoch}")
                return
            visualizer.create_visualizations(epoch)
        else:
            # Visualize all epochs
            for epoch_dir in sorted(epoch_dirs, key=lambda x: int(x.split('_')[1])):
                epoch = int(epoch_dir.split('_')[1])
                print(f"\nProcessing epoch {epoch}...")
                visualizer.create_visualizations(epoch)

    except ValueError as e:
        print(f"Error processing epoch: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
