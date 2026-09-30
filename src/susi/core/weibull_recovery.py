"""
Python-script for parameter recovery for the unweighted 2-parameter Weibull function assumed for diameter distribution.

Originally authored by Lauri Mehtätalo and Jouni Siipilehto for R language, see:
    Siipilehto & Mehtätalo 2013. "Parameter recovery vs. parameter prediction for the Weibull distribution
    validated for Scots pine stands in Finland". Silva Fennica 47(4), article id 1057, open access.

Python conversion done by Mikko Niemi, with the help of Microsoft Copilot
"""

import math

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gamma


def scale_dg_mean(D, shape):
    """Compute scale parameter for basal area-weighted mean diameter."""
    if shape <= 0.1:
        return np.nan
    try:
        return D * gamma(2 / shape + 1) / gamma(3 / shape + 1)
    except ArithmeticError:
        return np.nan


def residual(shape, G, N, D):
    """Residual to be minimized: difference between observed and Weibull-derived second moment."""
    scale = scale_dg_mean(D, shape)
    if np.isnan(scale):
        return np.nan
    return 40000 * G / (np.pi * N) - scale**2 * gamma(2 / shape + 1)


def recweib_b(G, N, D, trace=False):
    """
    Recover Weibull shape and scale parameters using basal area-weighted mean diameter.

    Parameters:
        G (float): Basal area (m²/ha)
        N (float): Number of stems (1/ha)
        D (float): Basal area-weighted mean diameter (cm)
        trace (bool): If True, prints intermediate results

    Returns:
        dict: Contains shape, scale, and residual value
    """
    if G <= 0 or N <= 0 or D <= 0:
        raise ValueError("G, N, and D must be positive numbers.")

    def safe_residual(s):
        r = residual(s, G, N, D)
        return r if not np.isnan(r) else 1e6

    # Try Brent's method first
    try:
        result = root_scalar(safe_residual, bracket=[0.2, 30.0], method="brentq")
        if result.converged:
            shape = result.root
            scale = scale_dg_mean(D, shape)
            val = residual(shape, G, N, D)
            if trace:
                print(
                    f"Recovered shape: {shape:.4f}, scale: {scale:.4f}, residual: {val:.4e}"
                )
            return {"shape": shape, "scale": scale, "val": val}
    except ValueError:  # brentq's "f(a) and f(b) must have different signs"
        if trace:
            print("Brentq failed, trying grid search…")

    # Fallback: grid search
    best_shape = None
    min_resid = float("inf")
    for s in np.linspace(0.2, 30.0, 300):
        r = safe_residual(s)
        if not np.isnan(r) and abs(r) < min_resid:
            min_resid = abs(r)
            best_shape = s

    if best_shape is None:
        if trace:
            print("Grid search failed.")
        return {"shape": np.nan, "scale": np.nan, "val": np.nan}

    shape = best_shape
    scale = scale_dg_mean(D, shape)
    val = residual(shape, G, N, D)

    if trace:
        print(
            f"Recovered shape (grid): {shape:.4f}, scale: {scale:.4f}, residual: {val:.4e}"
        )

    return {"shape": shape, "scale": scale, "val": val}


def generate_weibull_tree_list(
    n_classes: int, G: float, Dg: float, stems_ha: float
) -> list[tuple[float, float]]:
    """
    Generate synthetic tree list using 2-parameter Weibull distribution.
    Ensures minimum diameter is 1 cm.

    Parameters:
        n_classes (int): Number of diameter classes
        G (float): Basal area (m²/ha)
        Dg (float): Basal area-weighted mean diameter (cm)
        stems_ha (float): Total stems per hectare

    Returns:
        List of tuples: (stems per class, diameter midpoint in cm)
    """

    weibull_parameters = recweib_b(G, stems_ha, Dg)
    shape = weibull_parameters["shape"]
    scale = weibull_parameters["scale"]
    a = 0.0  # Assume location parameter is zero

    x_max = scale * (-math.log(0.01)) ** (1.0 / shape)  # ~99th percentile

    d_min = 1.0  # Enforce minimum diameter

    interval = (x_max - d_min) / n_classes

    result = []
    f_prev = 0.0
    x = d_min

    for _ in range(n_classes):
        x += interval
        d_mid = x - interval / 2.0

        f = 1 - math.exp(-(((x - a) / scale) ** shape))
        f = min(f, 1.0)

        p = f - f_prev
        f_prev = f

        stems_i = p * stems_ha
        result.append((stems_i, d_mid))

    return result
