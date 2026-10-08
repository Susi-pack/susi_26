# %%
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, assert_never

import numpy as np
import pareto_dp
from numpy.typing import NDArray

import susi.io.load_output_data as load_output
from analysis.optimization.dynamic_programming import from_numpy_arrays_to_nested_lists
from analysis.optimization.prepruning import preprune_pareto_dominated_scenarios
from susi.io.load_output_data import (
    NetcdfVariablePath,
    OutputDataStore,
    ScenarioID,
    StandID,
)


# INPUT API
class Direction(Enum):
    """
    Whether the optimization seeks a target's smallest or largest value.

    Always about the signed value: maximizing a variable that is always
    negative brings it closer to zero. It never means "minimize the absolute
    value", which would disagree with maximizing as soon as a variable's
    values cross zero.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class TargetSpec:
    """
    How one netcdf variable becomes one target of the optimization.

    aggregation collapses the variable's time/space array to the single
    float the optimization needs, e.g. `NetcdfVariableArray.mean_of_all_values`.

    A dict of these, keyed by the variable's path, is the optimization's
    whole configuration. It can be written by hand or read off the
    notebook's widgets, and its insertion order sets the order of the
    target vectors' columns.
    """

    aggregation: load_output.NetcdfAggregationFn
    direction: Direction


# The reductions a user can pick from to collapse a target variable's
# time/space array down to the single float the optimization needs, keyed by
# the label shown in the UI. Lives here rather than in a frontend so that the
# Streamlit page and the notebook cannot drift apart on what a label means.
AGGREGATION_METHODS_BY_LABEL: dict[str, load_output.NetcdfAggregationFn] = {
    "Mean of all values": load_output.NetcdfVariableArray.mean_of_all_values,
    "Spatial mean at last timestep": load_output.NetcdfVariableArray.spatial_mean_at_last_timestep,
    "Mean over space, sum over time": load_output.NetcdfVariableArray.mean_over_space_sum_over_time,
    "Spatial mean at initial timestep": load_output.NetcdfVariableArray.spatial_mean_at_initial_timestep,
}

# The range of epsilon (the Pareto front's precision) that solve_optimization
# accepts. Smaller is more precise but slower. Here, not in a frontend, so that
# the Streamlit page's input and a value typed into the notebook are held to
# the same bounds.
EPSILON_MIN = 1e-9
EPSILON_MAX = 1e4

# The help text shown next to a target's direction picker. Lives here for the
# same reason as AGGREGATION_METHODS_BY_LABEL: one wording for both frontends.
DIRECTION_HELP = (
    "Whether to seek the target's smallest or largest value. This is about "
    "the signed value: maximizing a variable that is always negative brings "
    "it closer to zero."
)


# OUTPUT API
@dataclass(frozen=True)
class DesignAndTargetVectors:
    design_vectors: NDArray[np.float64]  # shape: (n_pareto_solutions, n_vars)
    target_vectors: NDArray[np.float64]  # shape: (n_pareto_solutions, n_vars)


@dataclass(frozen=True)
class PreparedOptimizationData:
    """
    A project's netcdf data, reduced to what the Pareto search consumes.

    Produced by prepare_optimization_data() and passed to
    solve_optimization(). target_specs is carried along because the solve
    step needs it to undo the sign flip applied to maximized targets.
    """

    target_specs: dict[NetcdfVariablePath, TargetSpec]

    # One entry per stand; within a stand, one row per surviving scenario and
    # one column per target variable, already weighted by the stand's area.
    data_table: list[list[Any]]


@dataclass(frozen=True)
class OptimizationResults:
    """
    Results of the optimization run.

    random_points is included here (rather than computed separately) because
    both it and pareto_front require a database read — co-locating them avoids
    reading the data twice.

    Each field is a DesignAndTarget where the array fields are 2D matrices,
    with one row per point.
    """

    pareto_front: DesignAndTargetVectors
    random_points: DesignAndTargetVectors


# INTERNAL DATA STRUCTURES
@dataclass
class _TargetVariableArrays:
    """
    Target variable values in proper
    array shape for the optimization.

    Holds stands, scenarios and variables data
    to enable conversion between representations.
    """

    stands: list[StandID]  # ["stand_A", "stand_B", ...]
    scenarios: dict[
        StandID, Sequence[ScenarioID]
    ]  # {"stand_A": ["scen_1", "scen_2"], ...}
    variables: Sequence[NetcdfVariablePath]
    variables_target_specs: Sequence[TargetSpec]

    # Each stand has a different area. This is weighted in the optimization.
    stand_areas: dict[StandID, float]

    data_weighted_by_area: list[np.ndarray]

    def validate(self):
        # 1st level corresponds to stands
        if len(self.data_weighted_by_area) != len(self.stands):
            raise ValueError(
                "TargetVariableArray not initialized properly. First dimension must be stands."
            )
        # rows correspond to scenarios; columns to target variables
        for n_stand, scenarios in enumerate(self.scenarios.values()):
            if self.data_weighted_by_area[n_stand].shape[0] != len(scenarios):
                raise ValueError(
                    f"TargetVariableArray not initialized properly. Rows must correspond to scenarios. Found mismatch in stand number {n_stand}"
                )
            if self.data_weighted_by_area[n_stand].shape[1] != len(self.variables):
                raise ValueError(
                    f"TargetVariableArray not initialized properly. Columns must correspond to variables. Found mismatch in stand number {n_stand}"
                )
            if self.data_weighted_by_area[n_stand].ndim != 2:
                raise ValueError("Arrays must be 2-D.")


def read_data(
    run_dirpath: Path,
    target_specs: dict[NetcdfVariablePath, TargetSpec],
) -> OutputDataStore:
    """
    Read the target variables of every stand and scenario of one run.

    run_dirpath is one run's folder (`<project>/outputs/<run_id>/`), not the
    project's.

    The stands come from `load_output.list_stand_folders`, the same listing
    `stand_areas.stand_areas_for_run` uses. That is what lets
    `build_optimization_array` look each stand of the returned store up in the
    areas dict, and it fixes the stand order (natural sort of the stand IDs)
    so that it does not depend on the filesystem.
    """
    metadata_by_stand = load_output.load_all_metadatas_from_stands(
        folders=load_output.list_stand_folders(run_dirpath=run_dirpath)
    )
    return load_output.read_netcdf_files_for_selected_variables_from_metadatas(
        selected_variables=list(target_specs.keys()),
        metadata_by_stand=metadata_by_stand,
    )


def _sign_for_minimization(direction: Direction) -> int:
    """
    The factor that turns a target into one the optimizer can minimize.

    The Pareto search only ever minimizes, so a maximized target goes in with
    its sign flipped. Multiplying by the same factor again undoes the flip.
    """
    match direction:
        case Direction.MINIMIZE:
            return 1
        case Direction.MAXIMIZE:
            return -1
        case _:
            # A hand-written spec could carry e.g. the string "maximize";
            # fail here, naming it, rather than multiply by None later.
            assert_never(direction)


def _aggregate_time_series_to_float(
    target_spec: TargetSpec,
    array: load_output.NetcdfVariableArray,
) -> float:
    return _sign_for_minimization(target_spec.direction) * target_spec.aggregation(
        array
    )


def _product_of_elements_in_list(
    list_of_numbers: Sequence[int | float],
) -> int | float:
    product = 1
    for element in list_of_numbers:
        product *= element
    return product


def _compute_scenarios_cardinality(
    scenarios_per_stand: dict[StandID, Sequence[ScenarioID]],
) -> tuple[int, ...]:
    return tuple([len(scenarios) for _, scenarios in scenarios_per_stand.items()])


def build_optimization_array(
    data_store: OutputDataStore,
    stand_areas: dict[StandID, float],
    target_specs: dict[NetcdfVariablePath, TargetSpec],
) -> _TargetVariableArrays:
    number_of_target_variables = len(target_specs)

    variables = []
    variables_target_specs = []
    for var_path, target_spec in target_specs.items():
        variables.append(var_path)
        variables_target_specs.append(target_spec)

    # Initialize output variable
    data = []

    for _, stand_name in enumerate(data_store.stands):
        stand_area = stand_areas[stand_name]
        number_of_scenarios = len(data_store.scenarios[stand_name])

        # Initialize the array that holds the variables' values
        target_var_array = (
            np.ones((number_of_scenarios, number_of_target_variables)) * np.nan
        )

        for scenario_i, scenario_name in enumerate(data_store.scenarios[stand_name]):
            for var_i, (var_path, target_spec) in enumerate(target_specs.items()):
                target_var_array[scenario_i, var_i] = (
                    stand_area
                    * _aggregate_time_series_to_float(
                        array=data_store.get_variable_value_for_scenario_and_stand(
                            variable_path=var_path,
                            stand_id=stand_name,
                            scenario_id=scenario_name,
                        ),
                        target_spec=target_spec,
                    )
                )
        data.append(target_var_array)
    target_var_arrays = _TargetVariableArrays(
        stands=data_store.stands,
        scenarios=data_store.scenarios,
        variables=variables,
        variables_target_specs=variables_target_specs,
        stand_areas=stand_areas,
        data_weighted_by_area=data,
    )
    target_var_arrays.validate()
    return target_var_arrays


def _preprune_dominated_scenarios(
    var_arrays: _TargetVariableArrays,
) -> _TargetVariableArrays:
    prepruned_scenarios: dict[StandID, Sequence[ScenarioID]] = {}
    prepruned_data: list[np.ndarray] = []

    for (stand_name, scenario_names), stand_array in zip(
        var_arrays.scenarios.items(), var_arrays.data_weighted_by_area
    ):
        non_dominated_scenarios_index = preprune_pareto_dominated_scenarios(stand_array)
        prepruned_scenarios[StandID(stand_name)] = [
            ScenarioID(scenario_names[i]) for i in non_dominated_scenarios_index
        ]

        prepruned_data.append(stand_array[non_dominated_scenarios_index])

    return _TargetVariableArrays(
        stands=var_arrays.stands,
        scenarios=prepruned_scenarios,
        variables=var_arrays.variables,
        variables_target_specs=var_arrays.variables_target_specs,
        stand_areas=var_arrays.stand_areas,
        data_weighted_by_area=prepruned_data,
    )


def _print_cardinality_info(
    scenarios: dict[StandID, Sequence[ScenarioID]],
    prepruned_scenarios: dict[StandID, Sequence[ScenarioID]],
) -> None:
    scenarios_cardinality = _compute_scenarios_cardinality(scenarios)
    prepruned_scenarios_cardinality = _compute_scenarios_cardinality(
        prepruned_scenarios
    )

    n_total_combinations = _product_of_elements_in_list(scenarios_cardinality)

    print(
        f"\nTotal number of scenarios before pre-pruning:{sum(scenarios_cardinality)}"
    )
    print(
        f"Total number of scenarios after  pre-pruning:{sum(prepruned_scenarios_cardinality)}"
    )
    print(
        f"Number of pre-pruned scenarios: {sum(scenarios_cardinality) - sum(prepruned_scenarios_cardinality)}"
    )
    print(
        f"\nNumber of possible combinations before pre-pruning: {n_total_combinations:.2e}"
    )
    print(
        f"Number of possible combinations after  pre-pruning: {_product_of_elements_in_list(prepruned_scenarios_cardinality):.2e}"
    )
    print(
        f"Number of combinations pruned: {n_total_combinations - _product_of_elements_in_list(prepruned_scenarios_cardinality):.2e}"
    )


def find_pareto_front(
    target_specs: dict[NetcdfVariablePath, TargetSpec],
    run_dirpath: Path,
    stand_areas: dict[StandID, float],
) -> list[pareto_dp.ParetoFrontSolution]:

    print("optimization - Reading data...")

    data_store = read_data(target_specs=target_specs, run_dirpath=run_dirpath)

    print("optimization - Transforming data...")
    target_var_arrays = build_optimization_array(
        data_store=data_store,
        target_specs=target_specs,
        stand_areas=stand_areas,
    )
    prepruned = _preprune_dominated_scenarios(target_var_arrays)

    _print_cardinality_info(
        scenarios=data_store.scenarios, prepruned_scenarios=prepruned.scenarios
    )

    data_table_list = from_numpy_arrays_to_nested_lists(prepruned.data_weighted_by_area)

    print("optimization - Finding Pareto front...")
    return pareto_dp.find_pareto_front(data=data_table_list, epsilon=1e-7)


def _convert_list_of_points_to_array(
    list_of_points: list[pareto_dp.ParetoFrontSolution],
) -> DesignAndTargetVectors:

    return DesignAndTargetVectors(
        target_vectors=np.array([point.target_vector for point in list_of_points]),
        design_vectors=np.array([point.design_vector for point in list_of_points]),
    )


def _restore_maximized_targets_sign(
    target_specs: dict[NetcdfVariablePath, TargetSpec],
    vectors: DesignAndTargetVectors,
) -> DesignAndTargetVectors:
    """Reverses the sign flip applied to maximized targets.

    During optimization, targets with ``Direction.MAXIMIZE`` are multiplied
    by -1, because the Pareto search only ever minimizes. This function
    restores their original sign in the output.

    Only ``target_vectors`` are modified; ``design_vectors`` are passed
    through unchanged.

    Example:
        >>> import numpy as np
        >>> target_specs = {
        ...     NetcdfVariablePath("var_a"): TargetSpec(aggregation=None, direction=Direction.MINIMIZE),
        ...     NetcdfVariablePath("var_b"): TargetSpec(aggregation=None, direction=Direction.MAXIMIZE),
        ... }
        >>> vectors = DesignAndTargetVectors(
        ...     design_vectors=np.array([[0.5, 0.5]]),
        ...     target_vectors=np.array([[10.0, -20.0]]),
        ... )
        >>> result = _restore_maximized_targets_sign(target_specs, vectors)
        >>> result.target_vectors
        array([[10., 20.]])
        >>> result.design_vectors
        array([[0.5, 0.5]])
    """
    # No points at all (e.g. n_random_points=0) come back as a 1-D empty
    # array, which has no columns to index.
    if vectors.target_vectors.size == 0:
        return vectors
    target_vectors = vectors.target_vectors.copy()
    for var_i, target_spec in enumerate(target_specs.values()):
        target_vectors[:, var_i] *= _sign_for_minimization(target_spec.direction)
    return DesignAndTargetVectors(
        target_vectors=target_vectors,
        design_vectors=vectors.design_vectors,
    )


def prepare_optimization_data(
    target_specs: dict[NetcdfVariablePath, TargetSpec],
    run_dirpath: Path,
    stand_areas: dict[StandID, float],
) -> PreparedOptimizationData:
    """
    Read a run's netcdfs and reduce them to the Pareto search's input.

    This is the slow, I/O-bound half of run_optimization: it reads every
    stand/scenario netcdf, aggregates each target variable to one
    area-weighted float, and drops the scenarios that are dominated within
    their own stand. It prints how much that pre-pruning saved, so a caller
    can see the size of the search before committing to it.
    """
    print("optimization - Reading data...")

    data_store = read_data(target_specs=target_specs, run_dirpath=run_dirpath)

    print("optimization - Transforming data...")
    target_var_arrays = build_optimization_array(
        data_store=data_store,
        target_specs=target_specs,
        stand_areas=stand_areas,
    )
    prepruned = _preprune_dominated_scenarios(target_var_arrays)

    _print_cardinality_info(
        scenarios=data_store.scenarios, prepruned_scenarios=prepruned.scenarios
    )

    return PreparedOptimizationData(
        target_specs=target_specs,
        data_table=from_numpy_arrays_to_nested_lists(prepruned.data_weighted_by_area),
    )


def solve_optimization(
    prepared: PreparedOptimizationData,
    epsilon: float,
    n_random_points: int = 10000,
) -> OptimizationResults:
    """
    Run the Pareto search over already-prepared data.

    This is the compute-bound half of run_optimization, and the expensive
    one: cost grows with the number of scenario combinations reported by
    prepare_optimization_data() and with how fine epsilon is.
    """
    if not EPSILON_MIN <= epsilon <= EPSILON_MAX:
        raise ValueError(
            f"epsilon must be between {EPSILON_MIN} and {EPSILON_MAX}, got {epsilon}."
        )

    print("optimization - Finding Pareto front...")
    pareto_front = pareto_dp.find_pareto_front(
        data=prepared.data_table, epsilon=epsilon
    )

    print("optimization - Generating random solutions...")
    random_points = pareto_dp.create_random_points(
        data=prepared.data_table, n_points=n_random_points
    )

    return OptimizationResults(
        pareto_front=_restore_maximized_targets_sign(
            target_specs=prepared.target_specs,
            vectors=_convert_list_of_points_to_array(pareto_front),
        ),
        random_points=_restore_maximized_targets_sign(
            target_specs=prepared.target_specs,
            vectors=_convert_list_of_points_to_array(random_points),
        ),
    )


def run_optimization(
    target_specs: dict[NetcdfVariablePath, TargetSpec],
    run_dirpath: Path,
    stand_areas: dict[StandID, float],
    epsilon: float,
    n_random_points: int = 10000,
) -> OptimizationResults:
    """
    Read a run's data and find its Pareto front, in one call.

    Callers that want to inspect the problem's size before paying for the
    search (as the notebook does, one step per cell) can call
    prepare_optimization_data() and solve_optimization() separately instead.
    """
    return solve_optimization(
        prepared=prepare_optimization_data(
            target_specs=target_specs,
            run_dirpath=run_dirpath,
            stand_areas=stand_areas,
        ),
        epsilon=epsilon,
        n_random_points=n_random_points,
    )
