import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple, Dict
import os

class FeatureSpaceVisualizer:
    """A class for visualizing high-dimensional feature spaces with interactive controls."""

    def __init__(self, data: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]] = None):
        """
        Initialize the visualizer with dataset and configuration.

        Args:
            data: DataFrame containing the features and target
            target_column: Name of the target/class column
            feature_columns: List of feature column names to visualize. If None, uses all numeric columns.
        """
        self.data = data.copy()
        self.target_column = target_column

        # Setup feature columns
        if feature_columns is None:
            self.feature_columns = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if self.target_column in self.feature_columns:
                self.feature_columns.remove(self.target_column)
        else:
            self.feature_columns = feature_columns

        # Scale features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.data[self.feature_columns])

        # Get unique classes for coloring
        self.classes = sorted(self.data[target_column].unique())

    def create_3d_scatter(self, features: List[str], title: str = "3D Feature Space Visualization",
                         save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive 3D scatter plot of three selected features.
        """
        if len(features) != 3:
            raise ValueError("Exactly three features must be specified for 3D scatter plot")

        fig = px.scatter_3d(
            self.data,
            x=features[0],
            y=features[1],
            z=features[2],
            color=self.target_column,
            title=title,
            labels={
                features[0]: features[0].replace('_', ' ').title(),
                features[1]: features[1].replace('_', ' ').title(),
                features[2]: features[2].replace('_', ' ').title(),
                self.target_column: 'Class'
            }
        )

        # Enhance the appearance
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            legend_title_font_size=14,
            legend_font_size=12,
            scene=dict(
                xaxis_title_font=dict(size=12),
                yaxis_title_font=dict(size=12),
                zaxis_title_font=dict(size=12)
            )
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_tsne_projection(self, n_components: int = 2,
                             save_path: Optional[str] = None) -> go.Figure:
        """
        Create t-SNE projection of the feature space.

        Args:
            n_components: Number of components for projection (2 or 3)
            save_path: Optional path to save the plot as HTML

        Returns:
            Plotly figure object
        """
        # Create t-SNE projection
        tsne = TSNE(n_components=n_components, random_state=42)
        embedding = tsne.fit_transform(self.scaled_features)

        # Create DataFrame with projection
        proj_df = pd.DataFrame(embedding,
                             columns=[f'TSNE{i+1}' for i in range(n_components)])
        proj_df[self.target_column] = self.data[self.target_column]

        # Create visualization
        if n_components == 2:
            fig = px.scatter(
                proj_df,
                x='TSNE1',
                y='TSNE2',
                color=self.target_column,
                title='t-SNE Projection of Feature Space',
                labels={'TSNE1': 't-SNE Component 1',
                       'TSNE2': 't-SNE Component 2',
                       self.target_column: 'Class'}
            )
        else:  # 3D projection
            fig = px.scatter_3d(
                proj_df,
                x='TSNE1',
                y='TSNE2',
                z='TSNE3',
                color=self.target_column,
                title='3D t-SNE Projection of Feature Space',
                labels={'TSNE1': 't-SNE Component 1',
                       'TSNE2': 't-SNE Component 2',
                       'TSNE3': 't-SNE Component 3',
                       self.target_column: 'Class'}
            )

        # Enhance the appearance
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            legend_title_font_size=14,
            legend_font_size=12
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_parallel_coordinates(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create parallel coordinates plot of all features.
        """
        fig = px.parallel_coordinates(
            self.data,
            dimensions=self.feature_columns,
            color=self.target_column,
            title='Parallel Coordinates Plot of Feature Space',
        )

        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            showlegend=True,
            legend_title_font_size=14,
            legend_font_size=12
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_feature_matrix(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create a matrix of scatter plots for all feature pairs.
        """
        fig = px.scatter_matrix(
            self.data,
            dimensions=self.feature_columns,
            color=self.target_column,
            title='Feature Space Matrix'
        )

        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            showlegend=True,
            legend_title_font_size=14,
            legend_font_size=12
        )

        if save_path:
            fig.write_html(save_path)

        return fig

def visualize_feature_space(data: pd.DataFrame,
                          target_column: str,
                          feature_columns: Optional[List[str]] = None,
                          output_dir: str = "visualizations"):
    """
    Create and save a comprehensive set of feature space visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    viz = FeatureSpaceVisualizer(data, target_column, feature_columns)

    # 1. t-SNE 2D projection
    fig_tsne_2d = viz.create_tsne_projection(n_components=2)
    fig_tsne_2d.write_html(os.path.join(output_dir, 'tsne_2d.html'))

    # 2. t-SNE 3D projection
    fig_tsne_3d = viz.create_tsne_projection(n_components=3)
    fig_tsne_3d.write_html(os.path.join(output_dir, 'tsne_3d.html'))

    # 3. Parallel coordinates
    fig_parallel = viz.create_parallel_coordinates()
    fig_parallel.write_html(os.path.join(output_dir, 'parallel_coordinates.html'))

    # 4. Feature matrix
    fig_matrix = viz.create_feature_matrix()
    fig_matrix.write_html(os.path.join(output_dir, 'feature_matrix.html'))

    # 5. 3D scatter plots for first few feature combinations
    features = viz.feature_columns
    for i in range(min(len(features) - 2, 3)):
        feature_set = features[i:i+3]
        fig_3d = viz.create_3d_scatter(feature_set)
        fig_3d.write_html(os.path.join(output_dir, f'3d_scatter_{i+1}.html'))

    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    # Example with iris dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    visualize_feature_space(df, 'target')
