# %%
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
@dataclass(frozen=True)
class TargetVariableProperties:
    aggregation_function: load_output.NetcdfAggregationFn
    invert_optimization: bool  # If True, this puts a minus sign in the value: turn maximization into minimization


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
    solve_optimization(). variable_info is carried along because the solve
    step needs it to undo the sign flip applied to inverted variables.
    """

    variable_info: dict[NetcdfVariablePath, TargetVariableProperties]

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
    variables_properties: Sequence[TargetVariableProperties]

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
    project_dirpath: Path,
    variable_info: dict[NetcdfVariablePath, TargetVariableProperties],
) -> OutputDataStore:

    metadata_by_stand = load_output.load_all_metadatas_from_stands(
        folders=load_output.list_subdirectories(project_dirpath)
    )
    return load_output.read_netcdf_files_for_selected_variables_from_metadatas(
        selected_variables=list(variable_info.keys()),
        metadata_by_stand=metadata_by_stand,
    )


def _aggregate_time_series_to_float(
    target_variable_properties: TargetVariableProperties,
    array: load_output.NetcdfVariableArray,
) -> float:
    inverter: int = -1 if target_variable_properties.invert_optimization else 1

    return inverter * target_variable_properties.aggregation_function(array)


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
    optimization_variables: dict[NetcdfVariablePath, TargetVariableProperties],
) -> _TargetVariableArrays:
    number_of_target_variables = len(optimization_variables)

    variables = []
    variables_properties = []
    for var_path, var_properties in optimization_variables.items():
        variables.append(var_path)
        variables_properties.append(var_properties)

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
            for var_i, (var_path, var_properties) in enumerate(
                optimization_variables.items()
            ):
                target_var_array[scenario_i, var_i] = (
                    stand_area
                    * _aggregate_time_series_to_float(
                        array=data_store.get_variable_value_for_scenario_and_stand(
                            variable_path=var_path,
                            stand_id=stand_name,
                            scenario_id=scenario_name,
                        ),
                        target_variable_properties=var_properties,
                    )
                )
        data.append(target_var_array)
    target_var_arrays = _TargetVariableArrays(
        stands=data_store.stands,
        scenarios=data_store.scenarios,
        variables=variables,
        variables_properties=variables_properties,
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
        variables_properties=var_arrays.variables_properties,
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
    variable_info: dict[NetcdfVariablePath, TargetVariableProperties],
    project_dirpath: Path,
    stand_areas: dict[StandID, float],
) -> list[pareto_dp.ParetoFrontSolution]:

    print("optimization - Reading data...")

    data_store = read_data(variable_info=variable_info, project_dirpath=project_dirpath)

    print("optimization - Transforming data...")
    target_var_arrays = build_optimization_array(
        data_store=data_store,
        optimization_variables=variable_info,
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


def _flip_inverted_variables_sign(
    variable_info: dict[NetcdfVariablePath, TargetVariableProperties],
    vectors: DesignAndTargetVectors,
) -> DesignAndTargetVectors:
    """Reverses the sign flip applied to inverted target variables.

    During optimization, variables with ``invert_optimization=True`` are
    multiplied by -1 to convert maximization problems into minimization.
    This function restores their original sign in the output.

    Only ``target_vectors`` are modified; ``design_vectors`` are passed
    through unchanged.

    Example:
        >>> import numpy as np
        >>> var_info = {
        ...     NetcdfVariablePath("var_a"): TargetVariableProperties(aggregation_function=None, invert_optimization=False),
        ...     NetcdfVariablePath("var_b"): TargetVariableProperties(aggregation_function=None, invert_optimization=True),
        ... }
        >>> vectors = DesignAndTargetVectors(
        ...     design_vectors=np.array([[0.5, 0.5]]),
        ...     target_vectors=np.array([[10.0, -20.0]]),
        ... )
        >>> result = _flip_inverted_variables_sign(var_info, vectors)
        >>> result.target_vectors
        array([[10., 20.]])
        >>> result.design_vectors
        array([[0.5, 0.5]])
    """
    target_vectors = vectors.target_vectors.copy()
    for var_i, var_properties in enumerate(variable_info.values()):
        if var_properties.invert_optimization:
            target_vectors[:, var_i] *= -1
    return DesignAndTargetVectors(
        target_vectors=target_vectors,
        design_vectors=vectors.design_vectors,
    )


def prepare_optimization_data(
    variable_info: dict[NetcdfVariablePath, TargetVariableProperties],
    project_dirpath: Path,
    stand_areas: dict[StandID, float],
) -> PreparedOptimizationData:
    """
    Read the project's netcdfs and reduce them to the Pareto search's input.

    This is the slow, I/O-bound half of run_optimization: it reads every
    stand/scenario netcdf, aggregates each target variable to one
    area-weighted float, and drops the scenarios that are dominated within
    their own stand. It prints how much that pre-pruning saved, so a caller
    can see the size of the search before committing to it.
    """
    print("optimization - Reading data...")

    data_store = read_data(variable_info=variable_info, project_dirpath=project_dirpath)

    print("optimization - Transforming data...")
    target_var_arrays = build_optimization_array(
        data_store=data_store,
        optimization_variables=variable_info,
        stand_areas=stand_areas,
    )
    prepruned = _preprune_dominated_scenarios(target_var_arrays)

    _print_cardinality_info(
        scenarios=data_store.scenarios, prepruned_scenarios=prepruned.scenarios
    )

    return PreparedOptimizationData(
        variable_info=variable_info,
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
    print("optimization - Finding Pareto front...")
    pareto_front = pareto_dp.find_pareto_front(
        data=prepared.data_table, epsilon=epsilon
    )

    print("optimization - Generating random solutions...")
    random_points = pareto_dp.create_random_points(
        data=prepared.data_table, n_points=n_random_points
    )

    return OptimizationResults(
        pareto_front=_flip_inverted_variables_sign(
            variable_info=prepared.variable_info,
            vectors=_convert_list_of_points_to_array(pareto_front),
        ),
        random_points=_flip_inverted_variables_sign(
            variable_info=prepared.variable_info,
            vectors=_convert_list_of_points_to_array(random_points),
        ),
    )


def run_optimization(
    variable_info: dict[NetcdfVariablePath, TargetVariableProperties],
    project_dirpath: Path,
    stand_areas: dict[StandID, float],
    epsilon: float,
    n_random_points: int = 10000,
) -> OptimizationResults:
    """
    Read a project's data and find its Pareto front, in one call.

    Callers that want to inspect the problem's size before paying for the
    search (as the notebook does, one step per cell) can call
    prepare_optimization_data() and solve_optimization() separately instead.
    """
    return solve_optimization(
        prepared=prepare_optimization_data(
            variable_info=variable_info,
            project_dirpath=project_dirpath,
            stand_areas=stand_areas,
        ),
        epsilon=epsilon,
        n_random_points=n_random_points,
    )
