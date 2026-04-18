from __future__ import annotations
from typing import Any, Sequence
from dataclasses import dataclass
import numpy as np


@dataclass
class PartialParetoPoint(frozen=True):
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


def undo_shift_single_point_to_positive_value(
    point: Sequence[float], min_values_per_variable: Sequence[float]
) -> tuple[float, ...]:
    DELTA = 1e-6  # to avoid zeroes
    return tuple(
        coord + min_value - DELTA
        for coord, min_value in zip(point, min_values_per_variable)
    )


def undo_shift_to_positive_values(
    data: Sequence[Sequence[float]], minimum_values_per_variable: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        undo_shift_single_point_to_positive_value(
            point=point, min_values_per_variable=minimum_values_per_variable
        )
        for point in data
    )


def from_numpy_arrays_to_nested_tuples(
    arrays: Sequence[np.ndarray],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(map(tuple, arr.tolist())) for arr in arrays)


def from_nested_tuples_to_numpy_arrays(
    nested_tuples: tuple[tuple[float, ...], ...],
) -> tuple[np.ndarray, ...]:
    return tuple(np.array(t) for t in nested_tuples)


def does_p_dominate_q(p: PartialParetoPoint, q: PartialParetoPoint) -> bool:
    """
    Check Pareto dominance
    """

    return all(
        p_k <= q_k for p_k, q_k in zip(p.objective_vector, q.objective_vector)
    ) and any(p_k < q_k for p_k, q_k in zip(p.objective_vector, q.objective_vector))


def pareto_epsilon_prune(
    points: list[PartialParetoPoint], epsilon: float
) -> tuple[PartialParetoPoint, ...]:
    buckets: dict[tuple[int, ...], PartialParetoPoint] = {}
    for point in points:
        bucket_key = assign_bucket_to_point(point, epsilon=epsilon)

        if bucket_key not in buckets.keys():
            buckets[bucket_key] = point
        else:
            # If bucket already full, check if current
            # point is a better choice
            if does_p_dominate_q(p=point, q=buckets[bucket_key]):
                buckets[bucket_key] = point

    return tuple(buckets.values())
