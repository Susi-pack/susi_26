# Prunes out Pareto dominated scenarios for each stand.
# This is possible because all stands are decoupled from each other in the objective function

import numpy as np


def preprune_pareto_dominated_scenarios(A: np.ndarray, eps: float = 1e-9) -> list[int]:
    """
    Pareto prune a set of points (which works for minimization).

    Parameters
    ----------
    A : ndarray of shape (n_scenarios, n_optimization_variables)
        The set of values to minimize for each stand.

    eps : float
        Added to handle numerical tolerance issues in inequality checks.

    Returns
    -------
    list[int]
        Indices of non-dominated scenarios
    """

    n_scenarios, _n_optimization_variables = A.shape

    # Create array holding all possible pairwise comparisons
    # Initialize with no domination at all
    dominates = np.zeros((n_scenarios, n_scenarios), dtype=bool)

    for i in range(n_scenarios):
        for j in range(n_scenarios):
            if i == j:
                continue

            # Pareto pairwise comparisons:
            # A[i] dominates A[j] if:
            # A[i] <= A[j] in all dims AND strictly < in at least one
            less_or_equal = np.all(A[i] <= A[j] + eps)
            strictly_less = np.any(A[i] < A[j] - eps)

            dominates[i, j] = less_or_equal and strictly_less

    # Prune any scenario that is Pareto-dominated by at least one other scenario.
    dominated = np.any(dominates, axis=0)

    # return A[~dominated]
    return [
        scenario_ix for scenario_ix in range(n_scenarios) if not dominated[scenario_ix]
    ]
