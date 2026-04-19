from __future__ import annotations
from typing import Any, Sequence
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
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


def p_dominates_q(p: PartialParetoPoint, q: PartialParetoPoint) -> bool:
    """
    Check Pareto dominance
    """

    return all(
        p_k <= q_k for p_k, q_k in zip(p.objective_vector, q.objective_vector)
    ) and any(p_k < q_k for p_k, q_k in zip(p.objective_vector, q.objective_vector))


def compress_into_buckets(
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
            if p_dominates_q(p=point, q=buckets[bucket_key]):
                buckets[bucket_key] = point

    return tuple(buckets.values())


def pareto_prune(points: tuple[PartialParetoPoint, ...]):

    pareto = []

    for point in points:
        new_pareto = pareto.copy()
        keep_point = True
        for q in pareto:
            if p_dominates_q(p=q, q=point):
                keep_point = False
                break
            elif p_dominates_q(p=point, q=q):
                new_pareto.remove(q)

        if keep_point:
            new_pareto.append(point)

        pareto = new_pareto

    return pareto


def pareto_epsilon_prune(
    points: list[PartialParetoPoint], epsilon: float
) -> tuple[PartialParetoPoint, ...]:
    compressed_points = compress_into_buckets(points=points, epsilon=epsilon)
    return pareto_prune(compressed_points)


def find_pareto_front(
    data_table: tuple[tuple[Any, ...], ...],
) -> tuple[PartialParetoPoint, ...]:

    n_stands = len(data_table)

    # Initialize the partial Pareto  fronts with
    # the scenarios in the first stand
    first_stand_vectors = data_table[0]
    pareto_front: tuple[PartialParetoPoint, ...] = tuple(
        PartialParetoPoint(
            objective_vector=vector,
            parent_point=None,
            current_scenario_choice=scenario_number,
            stand_index=0,
        )
        for scenario_number, vector in enumerate(first_stand_vectors)
    )

    # Remove stand 0 from loop: already considered in the initialization
    for stand_ix in tqdm(range(1, n_stands)):
        pareto_front_new: list[PartialParetoPoint] = []

        for pareto_point in pareto_front:
            for scenario_ix in range(len(data_table[stand_ix])):
                new_point = tuple(
                    previous + new
                    for previous, new in zip(
                        pareto_point.objective_vector, data_table[stand_ix][scenario_ix]
                    )
                )
                pareto_front_new.append(
                    PartialParetoPoint(
                        objective_vector=new_point,
                        parent_point=pareto_point,
                        current_scenario_choice=scenario_ix,
                        stand_index=stand_ix,
                    )
                )

        pareto_front = pareto_epsilon_prune(points=pareto_front_new, epsilon=1e-7)

    return pareto_front
