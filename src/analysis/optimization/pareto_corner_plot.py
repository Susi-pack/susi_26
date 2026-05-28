import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def pareto_corner_plot(
    data: np.ndarray,
    random_points: np.ndarray | None,
    labels: list[str] | None = None,
    figsize: tuple | None = None,
    point_color: str = "#2D6A9F",
    point_alpha: float = 0.4,
    point_size: float = 8,
    label_fontsize: int = 11,
    hist_color: str = "#2D6A9F",
    hist_alpha: float = 0.4,
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
        Figure size. Defaults to (1.8*v, 1.8*v).
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
    _n, v = data.shape

    if labels is None:
        labels = [f"x{i}" for i in range(v)]
    if len(labels) != v:
        raise ValueError(f"Expected {v} labels, got {len(labels)}.")
    if figsize is None:
        figsize = (1.8 * v, 1.8 * v)

    df = pd.DataFrame(data, columns=pd.Index(labels))

    if random_points is not None:
        df["source"] = "Pareto"
        df_rand = pd.DataFrame(random_points, columns=pd.Index(labels))
        df_rand["source"] = "Random"
        df = pd.concat([df, df_rand], ignore_index=True)
        hue = "source"
        palette = {"Pareto": point_color, "Random": "orange"}
    else:
        hue = None
        palette = None

    g = sns.pairplot(
        df,
        hue=hue,
        palette=palette,
        corner=True,
        diag_kind="hist" if show_diagonal else None,
        plot_kws={
            "alpha": point_alpha,
            "s": point_size,
            "linewidths": 0,
        },
        diag_kws={
            "color": hist_color,
            "alpha": hist_alpha,
            "bins": bins,
            "edgecolor": "white",
            "linewidth": 0.4,
        },
        height=figsize[0] / v,
    )

    g.fig.set_size_inches(figsize)

    if title:
        g.fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01, color="#1A1A2E")

    if not show_diagonal:
        for i in range(v):
            if g.axes[i, i] is not None:
                g.axes[i, i].remove()

    for ax in g.axes.flatten():
        if ax is None or ax.get_visible() is False:
            continue
        ax.tick_params(labelsize=label_fontsize)
        if ax.get_xlabel():
            ax.xaxis.label.set_size(label_fontsize)
        if ax.get_ylabel():
            ax.yaxis.label.set_size(label_fontsize)

    if hue:
        g._legend.set_title("")
        for text in g._legend.get_texts():
            text.set_fontsize(label_fontsize)

    return g.fig
