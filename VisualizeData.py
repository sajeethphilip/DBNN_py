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
        """Optionally remove high cardinality columns based on threshold from config."""
        if not self.global_config:
            return df

        # Get threshold from config, but allow it to be disabled
        threshold = self.global_config.get('training_params', {}).get('cardinality_threshold', None)
        if threshold is None:  # If no threshold specified, keep all features
            return df

        df_filtered = df.copy()
        columns_to_drop = []
        target_column = self.dataset_config['target_column']
        feature_columns = [col for col in df.columns if col != target_column]

        # Calculate cardinality for all features
        feature_cardinality = []
        for column in feature_columns:
            unique_count = len(df[column].unique())
            unique_ratio = unique_count / len(df)
            feature_cardinality.append((column, unique_ratio))
            print(f"Feature {column} cardinality ratio: {unique_ratio:.3f}")

        print("\nKeeping all features regardless of cardinality.")
        remaining_features = [col for col in df_filtered.columns if col != target_column]
        print(f"Total features: {len(remaining_features)}")
        print("Features:", remaining_features)

        return df_filtered

    def _get_size_mapping(self, data, target_column):
        """
        Create a size mapping based on class frequencies.
        Classes with fewer samples get larger points for visibility.
        """
        class_counts = data[target_column].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()

        # Create inverse size mapping (smaller classes get bigger points)
        # Using linear scale instead of log to avoid NaN issues
        size_mapping = {}
        for class_label, count in class_counts.items():
            # Linear scaling from 8 to 15
            if max_count == min_count:
                size = 10  # If all classes have same count, use middle size
            else:
                size = 8 + (7 * (max_count - count) / (max_count - min_count))
            size_mapping[class_label] = float(size)  # Ensure float type

        # Create array of sizes matching the data
        sizes = data[target_column].map(size_mapping)

        # Handle any NaN values (shouldn't occur, but just in case)
        sizes = sizes.fillna(8.0)  # Default to minimum size for any NaN

        print("\nPoint size mapping (class: size):")
        for class_label, size in size_mapping.items():
            count = class_counts[class_label]
            print(f"Class {class_label}: {size:.1f} px ({count} samples)")

        return sizes

    def _create_epoch_visualizations(self, data: pd.DataFrame, set_type: str,
                                   target_column: str, save_dir: str):
        """Create all visualizations for one dataset."""
        # Reset index to ensure alignment
        data = data.reset_index(drop=True)

        feature_cols = [col for col in data.columns if col != target_column]
        print(f"\nCreating visualizations using {len(feature_cols)} features")

        # Get size mapping for points based on class frequencies
        point_sizes = self._get_size_mapping(data, target_column)

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[feature_cols])

        # 1. t-SNE 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data) - 1))
        tsne_result = tsne.fit_transform(scaled_features)

        tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df[target_column] = data[target_column]
        tsne_df = tsne_df.reset_index(drop=True)  # Reset index to ensure alignment

        # Instead of using plotly express, use graph objects for more control
        fig_2d = go.Figure()

        # Add traces for each class separately
        for class_label in sorted(data[target_column].unique()):
            mask = tsne_df[target_column] == class_label
            size_value = float(point_sizes[mask.index[mask]].iloc[0])

            fig_2d.add_trace(go.Scatter(
                x=tsne_df.loc[mask, 'TSNE1'],
                y=tsne_df.loc[mask, 'TSNE2'],
                mode='markers',
                name=f'Class {class_label}',
                marker=dict(
                    size=size_value,
                    line=dict(width=0.5, color='DarkSlateGrey'),
                    opacity=0.7
                )
            ))

        fig_2d.update_layout(title=f't-SNE 2D Projection - {set_type} set')
        fig_2d.write_html(os.path.join(save_dir, f'tsne_2d_{set_type}.html'))

        # 2. t-SNE 3D
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(data) - 1))
        tsne_result = tsne.fit_transform(scaled_features)

        tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2', 'TSNE3'])
        tsne_df[target_column] = data[target_column]
        tsne_df = tsne_df.reset_index(drop=True)  # Reset index to ensure alignment

        fig_3d = go.Figure()

        # Add traces for each class separately
        for class_label in sorted(data[target_column].unique()):
            mask = tsne_df[target_column] == class_label
            size_value = float(point_sizes[mask.index[mask]].iloc[0])

            fig_3d.add_trace(go.Scatter3d(
                x=tsne_df.loc[mask, 'TSNE1'],
                y=tsne_df.loc[mask, 'TSNE2'],
                z=tsne_df.loc[mask, 'TSNE3'],
                mode='markers',
                name=f'Class {class_label}',
                marker=dict(
                    size=size_value,
                    line=dict(width=0.5, color='DarkSlateGrey'),
                    opacity=0.7
                )
            ))

        fig_3d.update_layout(title=f't-SNE 3D Projection - {set_type} set')
        fig_3d.write_html(os.path.join(save_dir, f'tsne_3d_{set_type}.html'))

        # 3. Feature combinations 3D scatter plots
        if len(feature_cols) >= 3:
            from itertools import combinations
            feature_combinations = list(combinations(feature_cols, 3))
            max_combinations = 10  # Limit number of combinations
            if len(feature_combinations) > max_combinations:
                print(f"\nLimiting to {max_combinations} feature combinations for 3D plots")
                feature_combinations = feature_combinations[:max_combinations]

            for i, (f1, f2, f3) in enumerate(feature_combinations):
                fig_3d_feat = go.Figure()

                # Add traces for each class separately
                for class_label in sorted(data[target_column].unique()):
                    mask = data[target_column] == class_label
                    size_value = float(point_sizes[mask.index[mask]].iloc[0])

                    fig_3d_feat.add_trace(go.Scatter3d(
                        x=data.loc[mask, f1],
                        y=data.loc[mask, f2],
                        z=data.loc[mask, f3],
                        mode='markers',
                        name=f'Class {class_label}',
                        marker=dict(
                            size=size_value,
                            line=dict(width=0.5, color='DarkSlateGrey'),
                            opacity=0.7
                        )
                    ))

                fig_3d_feat.update_layout(
                    title=f'Features: {f1}, {f2}, {f3} - {set_type} set',
                    scene=dict(
                        xaxis_title=f1,
                        yaxis_title=f2,
                        zaxis_title=f3
                    )
                )
                fig_3d_feat.write_html(os.path.join(save_dir, f'features_3d_{i+1}_{set_type}.html'))

        # 4. Parallel coordinates (shows all features)
        print("\nCreating parallel coordinates plot with all features")
        fig_parallel = px.parallel_coordinates(data, dimensions=feature_cols,
                                            color=target_column,
                                            title=f'Parallel Coordinates - {set_type} set')
        fig_parallel.write_html(os.path.join(save_dir, f'parallel_coords_{set_type}.html'))

        # 5. Correlation Matrix
        print("Creating correlation matrix visualization")
        corr_matrix = data[feature_cols + [target_column]].corr()
        fig_corr = px.imshow(corr_matrix,
                            title=f'Correlation Matrix - {set_type} set',
                            aspect='auto')  # Adjust aspect ratio automatically
        fig_corr.write_html(os.path.join(save_dir, f'correlation_matrix_{set_type}.html'))
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
