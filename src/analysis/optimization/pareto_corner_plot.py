import matplotlib.pyplot as plt
import numpy as np


def _style_ax(ax: plt.Axes, number_of_ticks: int = 3) -> None:
    """Apply a clean, consistent style to an axis."""
    for spine in ax.spines.values():
        spine.set_edgecolor("#C8CDD6")
        spine.set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=11, colors="#555566", length=3, width=0.7)
    ax.grid(True, color="white", linewidth=0.6, alpha=0.8)
    # Limit ticks per axis to avoid crowding
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=number_of_ticks, prune="both"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=number_of_ticks, prune="both"))
    # Scientific notation for large/small values (outside 0.01–999)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", scilimits=(-2, 3), useMathText=True)


def pareto_corner_plot(
    data: np.ndarray,
    random_points: np.ndarray | None,
    labels: list[str] | None = None,
    figsize: tuple | None = None,
    point_color: str = "#2D6A9F",
    point_alpha: float = 0.4,
    point_size: float = 8,
    hist_color: str = "#2D6A9F",
    hist_alpha: float = 0.7,
    bins: int = 30,
    show_diagonal: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """
    Create an upper-triangular corner plot from an (n, v) numpy array.

    The diagonal cells show a histogram of each variable.
    The upper-triangular cells show scatter plots of variable pairs (row, col).
    The lower-triangular cells are left blank (they mirror the upper triangle).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n, v), where n is the number of points and v is the
        number of variables.
    random_points : np.ndarray | None
        Show points corresponding to random design vectors. Omit if None.
    labels : list of str, optional
        Variable names for axis labels. Defaults to ["x0", "x1", ...].
    figsize : tuple, optional
        Figure size. Defaults to (2.5*v, 2.5*v).
    point_color : str
        Color for scatter plot points.
    point_alpha : float
        Alpha (opacity) for scatter plot points.
    point_size : float
        Marker size for scatter plot points.
    hist_color : str
        Color for histogram bars.
    hist_alpha : float
        Alpha for histogram bars.
    bins : int
        Number of histogram bins on the diagonal.
    show_diagonal : bool
        If True, show histograms on the diagonal. If False, leave diagonal blank.
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n, v = data.shape

    if labels is None:
        labels = [f"x{i}" for i in range(v)]
    if len(labels) != v:
        raise ValueError(f"Expected {v} labels, got {len(labels)}.")
    if figsize is None:
        figsize = (2.5 * v, 2.5 * v)

    fig, axes = plt.subplots(v, v, figsize=figsize)

    for row in range(v):
        for col in range(v):
            ax = axes[row, col]
            ax.set_facecolor("#F0F3F8")

            if row == col and show_diagonal:
                # Diagonal: histogram of variable `row`
                ax.hist(
                    data[:, row],
                    bins=bins,
                    color=hist_color,
                    alpha=hist_alpha,
                    edgecolor="white",
                    linewidth=0.4,
                )
                _style_ax(ax)

            elif col > row:
                # Upper triangle: scatter of (row variable, col variable)
                ax.scatter(
                    data[:, col],
                    data[:, row],
                    color=point_color,
                    alpha=point_alpha,
                    s=point_size,
                    linewidths=0,
                )
                if random_points is not None:
                    ax.scatter(
                        random_points[:, col],
                        random_points[:, row],
                        color="orange",
                        alpha=point_alpha,
                        s=point_size,
                        linewidths=0,
                    )
                _style_ax(ax)

            else:
                # Lower triangle: hide
                ax.set_visible(False)

            # Axis labels on the edges only
            if col > row or (row == col and show_diagonal):
                if row == 0:
                    ax.set_xlabel(labels[col], fontsize=13, labelpad=4)
                    ax.xaxis.set_label_position("top")
                    ax.xaxis.tick_top()
                if col == row and show_diagonal:
                    # Last diagonal cell: show x-ticks on the right side
                    if row == v - 1:
                        ax.set_ylabel(labels[col], fontsize=13, labelpad=4)
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.tick_right()
                elif col == v - 1:
                    ax.set_ylabel(labels[row], fontsize=13, labelpad=4)
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()

                # Only show tick labels on outer edges
                if row != 0:
                    ax.set_xticklabels([])
                if col != v - 1:
                    ax.set_yticklabels([])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01, color="#1A1A2E")

    plt.tight_layout(pad=0.5)
    return fig
