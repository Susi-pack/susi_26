from __future__ import annotations
from typing import Any, Sequence
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# Delta used to avoid log(0) errors
_DELTA = 1e-9


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


@dataclass(frozen=True)
class SolutionParetoPoint:
    """
    Solution of the optimization algorithm
    """

    design_vector: tuple[int, ...]
    target_vector: tuple[float, ...]


def assign_bucket_to_point(
    point: PartialParetoPoint, epsilon: float
) -> tuple[int, ...]:
    """
    Assign a point to a bucket in the coarse-grained space.

    A bucket is the v-dimensional pixel where the Pareto point falls
    when the space is coarse-grained using logarithmic binning.

    Args:
        point: The Pareto point to assign to a bucket
        epsilon: The binning parameter controlling bucket size

    Returns:
        A tuple of integers representing the bucket coordinates
    """
    base = np.log(1 + epsilon)

    return tuple(
        int(np.floor(np.log(point_coordinate + _DELTA) / base))
        for point_coordinate in point.objective_vector
    )


def get_minimum_values_per_variable(data: list[np.ndarray]) -> tuple[float, ...]:
    return tuple(np.concatenate(data).min(axis=0).tolist())


def shift_points_to_positive_values(
    data: list[np.ndarray], minimum_values_per_variable: tuple[float, ...]
) -> list[np.ndarray]:
    """
    Do a translation of the coordinate system so that all values of the variables are positive.
    """
    return [arr - minimum_values_per_variable + _DELTA for arr in data]


def from_numpy_arrays_to_nested_tuples(
    arrays: Sequence[np.ndarray],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(map(tuple, arr.tolist())) for arr in arrays)


def from_numpy_arrays_to_nested_lists(
    arrays: Sequence[np.ndarray],
) -> list[list[Any]]:
    return list(list(arr.tolist()) for arr in arrays)


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


def pareto_prune(points: tuple[PartialParetoPoint, ...]) -> tuple[PartialParetoPoint]:

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

    return tuple(pareto)


def pareto_epsilon_prune(
    points: list[PartialParetoPoint], epsilon: float
) -> tuple[PartialParetoPoint, ...]:
    compressed_points = compress_into_buckets(points=points, epsilon=epsilon)
    return pareto_prune(compressed_points)


def find_pareto_front(
    data_table: tuple[tuple[Any, ...], ...], epsilon: float
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

        pareto_front = pareto_epsilon_prune(points=pareto_front_new, epsilon=epsilon)

    return pareto_front


def recover_scenario_choices(point: PartialParetoPoint) -> tuple[int, ...]:
    choices = []
    current = point
    while current is not None:
        choices.append(current.current_scenario_choice)
        current = current.parent_point
    choices.reverse()
    return tuple(choices)


def compute_objective(
    design_vector: tuple[int, ...],
    data_table: tuple[tuple[Any, ...], ...],
) -> tuple[float, ...]:
    assert len(design_vector) == len(data_table)

    n_variables = len(data_table[0][0])
    objective = [0] * n_variables

    for stand_ix, scenario_ix in enumerate(design_vector):
        for var in range(n_variables):
            objective[var] += data_table[stand_ix][scenario_ix][var]

    return tuple(objective)


def reconstruct_solution_pareto_front(
    pareto_front: tuple[PartialParetoPoint, ...],
    data_table: tuple[tuple[Any, ...], ...],
) -> tuple[SolutionParetoPoint, ...]:
    """
    - Recovers scenario choices from nested structure
    - Shifts the target vectors back from positive space
    """
    solution = []
    for point in pareto_front:
        design_vector = recover_scenario_choices(point)
        solution.append(
            SolutionParetoPoint(
                design_vector=design_vector,
                target_vector=compute_objective(
                    design_vector=design_vector, data_table=data_table
                ),
            )
        )

    return tuple(solution)
