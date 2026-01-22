"""Visualization tools with fixed styles."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autoviz_agent.registry.tools import tool
from autoviz_agent.runtime.determinism import configure_matplotlib_backend, get_deterministic_style
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)

# Configure matplotlib backend once on import
configure_matplotlib_backend()


@tool(
    description="Create line plot",
    param_overrides={
        "x": {"role": "temporal"},
        "y": {"role": "numeric"},
    },
)
def plot_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Path:
    """
    Create a line plot.

    Args:
        df: Input DataFrame
        x: X-axis column
        y: Y-axis column
        output_path: Path to save plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Path to saved plot
    """
    plt.rcParams.update(get_deterministic_style())
    fig, ax = plt.subplots()

    ax.plot(df[x], df[y], marker="o")
    ax.set_title(title or f"{y} over {x}")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    logger.info(f"Saved line plot to {output_path}")
    return output_path


@tool(
    description="Create bar plot",
    param_overrides={
        "x": {"role": "categorical"},
        "y": {"role": "numeric"},
        "hue": {"role": "categorical"},
    },
)
def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: Optional[str] = None,
    horizontal: bool = False,
    hue: Optional[str] = None,
) -> Path:
    """
    Create a bar plot.

    Args:
        df: Input DataFrame
        x: X-axis column (categories)
        y: Y-axis column (values)
        output_path: Path to save plot
        title: Plot title
        horizontal: Whether to create horizontal bars
        hue: Column for color grouping (creates grouped bars)

    Returns:
        Path to saved plot
    """
    plt.rcParams.update(get_deterministic_style())
    fig, ax = plt.subplots()

    if hue:
        # Use seaborn for grouped bar chart
        import seaborn as sns
        if horizontal:
            sns.barplot(data=df, y=x, x=y, hue=hue, ax=ax)
        else:
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(title or f"{y} by {x} and {hue}")
    else:
        if horizontal:
            ax.barh(df[x], df[y])
            ax.set_xlabel(y)
            ax.set_ylabel(x)
        else:
            ax.bar(df[x], df[y])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        ax.set_title(title or f"{y} by {x}")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    logger.info(f"Saved bar plot to {output_path}")
    return output_path


@tool(
    description="Create scatter plot",
    param_overrides={
        "x": {"role": "numeric"},
        "y": {"role": "numeric"},
        "hue": {"role": "categorical"},
    },
)
def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: Optional[str] = None,
    hue: Optional[str] = None,
) -> Path:
    """
    Create a scatter plot.

    Args:
        df: Input DataFrame
        x: X-axis column
        y: Y-axis column
        output_path: Path to save plot
        title: Plot title
        hue: Column for color encoding

    Returns:
        Path to saved plot
    """
    plt.rcParams.update(get_deterministic_style())
    fig, ax = plt.subplots()

    if hue:
        for category in sorted(df[hue].unique()):
            mask = df[hue] == category
            ax.scatter(df[mask][x], df[mask][y], label=str(category), alpha=0.6)
        ax.legend()
    else:
        ax.scatter(df[x], df[y], alpha=0.6)

    ax.set_title(title or f"{y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    logger.info(f"Saved scatter plot to {output_path}")
    return output_path


@tool(
    description="Create histogram",
    param_overrides={"column": {"role": "numeric"}},
)
def plot_histogram(
    df: pd.DataFrame,
    column: str,
    output_path: Path,
    bins: int = 20,
    title: Optional[str] = None,
) -> Path:
    """
    Create a histogram.

    Args:
        df: Input DataFrame
        column: Column to plot
        output_path: Path to save plot
        bins: Number of bins
        title: Plot title

    Returns:
        Path to saved plot
    """
    plt.rcParams.update(get_deterministic_style())
    fig, ax = plt.subplots()

    ax.hist(df[column].dropna(), bins=bins, edgecolor="black", alpha=0.7)
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    logger.info(f"Saved histogram to {output_path}")
    return output_path


@tool(description="Create heatmap")
def plot_heatmap(
    data: pd.DataFrame,
    output_path: Path,
    title: Optional[str] = None,
    cmap: str = "coolwarm",
    annot: bool = True,
    select_numeric: bool = False,
) -> Path:
    """
    Create a heatmap (e.g., correlation matrix).

    Args:
        data: Input data (typically correlation matrix)
        output_path: Path to save plot
        title: Plot title
        cmap: Color map
        annot: Whether to annotate cells
        select_numeric: If True, select only numeric columns and compute correlation

    Returns:
        Path to saved plot
    """
    plt.rcParams.update(get_deterministic_style())
    fig, ax = plt.subplots(figsize=(10, 8))

    # If select_numeric is True, filter to numeric columns and compute correlation
    if select_numeric:
        numeric_df = data.select_dtypes(include=['number'])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for heatmap")
        # Compute correlation matrix for numeric columns
        data = numeric_df.corr()

    sns.heatmap(data, annot=annot, fmt=".2f", cmap=cmap, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(title or "Heatmap")

    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    logger.info(f"Saved heatmap to {output_path}")
    return output_path


@tool(
    description="Create box plot",
    param_overrides={
        "column": {"role": "numeric"},
        "by": {"role": "categorical"},
    },
)
def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    output_path: Path,
    by: Optional[str] = None,
    title: Optional[str] = None,
) -> Path:
    """
    Create a box plot.

    Args:
        df: Input DataFrame
        column: Column to plot
        output_path: Path to save plot
        by: Optional grouping column
        title: Plot title

    Returns:
        Path to saved plot
    """
    plt.rcParams.update(get_deterministic_style())
    fig, ax = plt.subplots()

    if by:
        groups = sorted(df[by].unique())
        data = [df[df[by] == g][column].dropna() for g in groups]
        ax.boxplot(data, labels=groups)
        ax.set_xlabel(by)
    else:
        ax.boxplot(df[column].dropna())

    ax.set_title(title or f"Box Plot of {column}")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    logger.info(f"Saved box plot to {output_path}")
    return output_path
