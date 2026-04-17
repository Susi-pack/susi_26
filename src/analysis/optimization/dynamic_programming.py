from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PartialParetoPoint:
    """
    Info about any point considered in the algorithm.
    """

    # v-dimensional vector in the objective space
    objective_vector: tuple[float, ...]

    # Pointer to the parent vector (the i-1 step in the algorithm).
    # Takes None value for the first stand, which has no parents.
    parent_point: PartialParetoPoint | None

    # scenario choice for the ith step in the algorithm.
    # The design space vector of the current point can be recovered
    # by attaching this choice number to the parent point.
    current_scenario_choice: int

    # step in the algorithm, a.k.a. i. Storing just in case.
    stand_index: int


def assign_bucket_to_point(
    point: PartialParetoPoint, epsilon: float
) -> tuple[int, ...]:
    """
    A bucket is the v-dimensional pixel where the Pareto point falls
    when the space is coarse-grained
    """
    # Delta used to avoid log(0) errors
    DELTA = 1e-9
    base = np.log(1 + epsilon)

    return tuple(
        int(np.floor(np.log(point_coordinate + DELTA) / base))
        for point_coordinate in point.objective_vector
    )


def get_minimum_values_per_variable(data: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(data).min(axis=0)


def shift_points_to_positive_values(
    data: list[np.ndarray], minimum_values_per_variable: np.ndarray
) -> list[np.ndarray]:
    """
    Do a translation of the coordinate system so that all values of the variables are positive.
    """
    DELTA = 1e-6  # to avoid zeroes
    return [arr - minimum_values_per_variable + DELTA for arr in data]


def undo_shift_to_positive_values(
    data: list[np.ndarray], minimum_values_per_variable: np.ndarray
) -> list[np.ndarray]:
    DELTA = 1e-6  # to avoid zeroes
    return [arr + minimum_values_per_variable - DELTA for arr in data]
