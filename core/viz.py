"""Automatic visualization for query results."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class AutoViz:
    """Automatic chart generation based on data characteristics."""

    def __init__(self, output_dir: Path):
        """
        Initialize AutoViz.

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def infer_chart_type(self, df: pd.DataFrame) -> str:
        """
        Infer appropriate chart type from DataFrame.

        Args:
            df: Query result DataFrame

        Returns:
            Chart type: 'line', 'bar', 'scatter', or 'table'
        """
        if df.empty or len(df.columns) < 2:
            return 'table'

        # Get first two columns
        col1, col2 = df.columns[0], df.columns[1]
        dtype1, dtype2 = df[col1].dtype, df[col2].dtype

        # Time series: date + numeric
        if pd.api.types.is_datetime64_any_dtype(dtype1) or 'date' in col1.lower():
            if pd.api.types.is_numeric_dtype(dtype2):
                return 'line'

        # Two numeric columns: scatter
        if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
            return 'scatter'

        # Categorical + numeric: bar
        if pd.api.types.is_object_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
            return 'bar'

        return 'table'

    def plot_line(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Create line chart.

        Args:
            df: DataFrame with date/time in first column
            filename: Output filename

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get first column (x-axis, usually time)
        x_col = df.columns[0]
        x_data = pd.to_datetime(df[x_col]) if not pd.api.types.is_datetime64_any_dtype(df[x_col]) else df[x_col]

        # Plot numeric columns
        for col in df.columns[1:]:
            if pd.api.types.is_numeric_dtype(df[col]):
                ax.plot(x_data, df[col], marker='o', label=col, linewidth=2)

        ax.set_xlabel(x_col)
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis for dates
        if pd.api.types.is_datetime64_any_dtype(x_data):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_bar(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Create bar chart.

        Args:
            df: DataFrame with category in first column
            filename: Output filename

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Limit to top N categories if too many
        plot_df = df.head(15)

        x_col = plot_df.columns[0]
        y_col = plot_df.columns[1]

        ax.bar(range(len(plot_df)), plot_df[y_col], color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df[x_col], rotation=45, ha='right')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_scatter(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Create scatter plot.

        Args:
            df: DataFrame with two numeric columns
            filename: Output filename

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x_col, y_col = df.columns[0], df.columns[1]
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=80, color='steelblue')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def visualize(
        self,
        df: pd.DataFrame,
        filename: str = "chart.png",
        chart_type: Optional[str] = None
    ) -> Tuple[Optional[Path], str]:
        """
        Automatically visualize DataFrame.

        Args:
            df: Query result DataFrame
            filename: Output filename
            chart_type: Chart type (auto-inferred if None)

        Returns:
            Tuple of (chart_path, chart_type)
        """
        if df.empty:
            return None, 'empty'

        # Infer chart type if not provided
        if chart_type is None:
            chart_type = self.infer_chart_type(df)

        # Generate chart
        try:
            if chart_type == 'line':
                path = self.plot_line(df, filename)
                return path, 'line'
            elif chart_type == 'bar':
                path = self.plot_bar(df, filename)
                return path, 'bar'
            elif chart_type == 'scatter':
                path = self.plot_scatter(df, filename)
                return path, 'scatter'
            else:
                return None, 'table'
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
            return None, 'error'

    def summarize_result(self, df: pd.DataFrame, chart_path: Optional[Path] = None) -> str:
        """
        Generate textual summary of query result.

        Args:
            df: Query result DataFrame
            chart_path: Path to chart if generated

        Returns:
            Summary text
        """
        if df.empty:
            return "Query returned no results."

        summary_parts = [
            f"Query returned {len(df)} rows and {len(df.columns)} columns."
        ]

        # Add value ranges for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            min_val, max_val = df[col].min(), df[col].max()
            summary_parts.append(
                f"{col} ranges from {min_val:.2f} to {max_val:.2f}."
            )

        if chart_path:
            summary_parts.append(f"Chart saved to {chart_path.name}.")

        return " ".join(summary_parts)
