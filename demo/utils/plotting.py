"""
Plotting utilities - simple matplotlib helpers for channel metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List


def plot_channel_metric(
    df: pd.DataFrame,
    channel_col: str = 'channel',
    metric_col: str = 'value',
    title: str = 'Metric by Channel',
    ylabel: Optional[str] = None,
    highlight_channels: Optional[List[str]] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create a bar chart for a metric by channel.

    Args:
        df: DataFrame with channel and metric columns
        channel_col: Name of channel column
        metric_col: Name of metric column to plot
        title: Chart title
        ylabel: Y-axis label (uses metric_col if None)
        highlight_channels: List of channels to highlight with different color
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object
    """
    # Sort by metric value
    df_sorted = df.sort_values(metric_col, ascending=False).copy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    colors = []
    if highlight_channels:
        for channel in df_sorted[channel_col]:
            if channel in highlight_channels:
                colors.append('#ff7f0e')  # Orange for highlighted
            else:
                colors.append('#1f77b4')  # Blue for others
    else:
        colors = '#1f77b4'

    # Create bar chart
    bars = ax.bar(df_sorted[channel_col], df_sorted[metric_col], color=colors)

    # Labels and title
    ax.set_xlabel('Channel', fontsize=11)
    ax.set_ylabel(ylabel or metric_col, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if pd.notna(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    # Grid for readability
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_hypothesis_comparison(
    current_cac: float,
    projected_cac: float,
    ci_lower: float,
    ci_upper: float,
    title: str = 'CAC Projection with 95% CI',
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Create a comparison chart for hypothesis testing.

    Args:
        current_cac: Current blended CAC
        projected_cac: Projected CAC after budget shift
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        title: Chart title
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    scenarios = ['Current', 'Projected']
    values = [current_cac, projected_cac]
    colors = ['#1f77b4', '#2ca02c']

    bars = ax.bar(scenarios, values, color=colors, alpha=0.7, width=0.5)

    # Add error bar for projected
    ax.errorbar(
        [1],  # Position of 'Projected' bar
        [projected_cac],
        yerr=[[projected_cac - ci_lower], [ci_upper - projected_cac]],
        fmt='none',
        color='black',
        capsize=10,
        capthick=2,
        label='95% CI'
    )

    # Labels
    ax.set_ylabel('CAC ($)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            value,
            f'${value:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Add CI annotation
    ax.text(
        1,
        ci_upper + (ci_upper - ci_lower) * 0.1,
        f'CI: [${ci_lower:.2f}, ${ci_upper:.2f}]',
        ha='center',
        fontsize=9
    )

    # Grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend()
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, output_path: str, dpi: int = 100):
    """
    Save a matplotlib figure to file.

    Args:
        fig: Matplotlib figure
        output_path: Path to save file
        dpi: Resolution in dots per inch
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
