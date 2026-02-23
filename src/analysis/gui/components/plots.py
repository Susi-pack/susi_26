import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np


def spatial_bars(data: np.ndarray) -> matplotlib.figure.Figure:
    """
    data is a 2D np.ndarray with spatial and temporal values.
    No scenarios
    """
    n_time, n_space = data.shape

    fig, ax = plt.subplots(figsize=(8, 5))

    # Grouped bars - each time step has bars for each space point
    x = np.arange(n_time)
    width = 0.8 / n_space

    for space_idx in range(n_space):
        offset = (space_idx - n_space / 2 + 0.5) * width
        ax.bar(
            x + offset,
            data[:, space_idx],
            width,
            label=f"Space {space_idx}" if n_space <= 10 else None,
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Temporal Evolution")
    if n_space <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    return fig


def temporal_stats(data: np.ndarray) -> matplotlib.figure.Figure:
    n_time, n_space = data.shape
    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate over space
    mean_over_space = np.mean(data, axis=1)
    std_over_space = np.std(data, axis=1)

    x_time = np.arange(n_time)
    ax.plot(x_time, mean_over_space, "b-", label="Mean", linewidth=2)
    ax.fill_between(
        x_time,
        mean_over_space - std_over_space,
        mean_over_space + std_over_space,
        alpha=0.3,
        color="blue",
        label="±1 Std",
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Spatial Mean and Variability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
