# %%
from susi.io.utils import read_json_file
from typing import Callable, Sequence, Literal, assert_never, NewType
from dataclasses import dataclass
import numpy as np
from numba import jit
import time
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.visualization.scatter import Scatter

from susi.io.app_settings import AppSettings
import susi.io.load_output_data as load_output
from susi.io.load_output_data import (
    NetcdfVariablePath,
    StandID,
    ScenarioID,
    OutputDataStore,
)

# %%
RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)

# %% Load paroninkorpi metadata

PROJECT_ID = "paroninkorpi"
stand_folderpaths = load_output.list_subdirectories(
    AppSettings().output_folder / PROJECT_ID
)
# stand_ids: list[StandID] = [StandID(path.name) for path in stand_folderpaths]

metadata_by_stand = load_output.load_all_metadatas_from_stands(
    folders=stand_folderpaths
)


# %% Read netcdf file structure and choose (filter) variables
# All Susi netcdf files have the same structure.
# Pick one to get the group/variables hierarchical structure.
# This will be useful in choosing what variables to read later.
sample_netcdf_filepath = metadata_by_stand[
    load_output.StandID(stand_folderpaths[0].name)
].iloc[0]["netcdf_output_filepath"]

all_variables: dict[load_output.NetcdfVariablePath, load_output.NetcdfVariableInfo] = (
    load_output.list_all_netcdf_variables(sample_netcdf_filepath)
)

# %% Read variables

TargetVariableDict = NewType("TargetVariableDict", dict[NetcdfVariablePath, float])


@dataclass
class TargetVariableProperties:
    aggregation_method: Literal["mean_all", "mean_last_timestep"]
    invert_optimization: bool  # If True, this puts a minus sign in the value: turn maximization into minimization


def aggregate_time_series_to_float(
    array: load_output.NetcdfVariableArray,
    target_variable_properties: TargetVariableProperties,
) -> float:
    inverter: int = -1 if target_variable_properties.invert_optimization else 1

    match target_variable_properties.aggregation_method:
        case "mean_all":
            return inverter * array.mean_of_all_values()
        case "mean_last_timestep":
            return inverter * array.spatial_mean_at_last_timestep()
        case _:
            assert_never()


TARGET_VARIABLES = {
    NetcdfVariablePath("/stand/volume"): TargetVariableProperties(
        aggregation_method="mean_last_timestep",
        invert_optimization=True,
    ),
    NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"): TargetVariableProperties(
        aggregation_method="mean_all",
        invert_optimization=True,
    ),
    NetcdfVariablePath("/balance/N/to_water"): TargetVariableProperties(
        aggregation_method="mean_all", invert_optimization=False
    ),
}

data_store: load_output.OutputDataStore = (
    load_output.read_netcdf_files_for_selected_variables(
        selected_variables=list(TARGET_VARIABLES.keys()),
        metadata_by_stand=metadata_by_stand,
    )
)


# %% compute total number of combinations
def product_of_elements_in_list(list_of_numbers: Sequence[int | float]) -> int | float:
    product = 1
    for element in list_of_numbers:
        product *= element
    return product


def compute_scenarios_cardinality(
    scenarios_per_stand: dict[StandID, list[ScenarioID]],
) -> tuple[int, ...]:
    return tuple([len(scenarios) for stand, scenarios in scenarios_per_stand.items()])


scenarios_cardinality = compute_scenarios_cardinality(data_store.scenarios)
n_total_combinations = product_of_elements_in_list(scenarios_cardinality)

print("\nNumber of scenarios for each stand:")
print(scenarios_cardinality)
print("\nTotal number of scenarios (sum):")
print(sum(scenarios_cardinality))
print(f"\nNumber of total combinations: {n_total_combinations:.2e}")


# %% Transform data representation from data_store to optimization array
@dataclass
class TargetVariableArrays:
    """
    Target variable values in proper
    array shape for the optimization.

    Holds stands, scenarios and variables data
    to enable conversion between representations.
    """

    stands: list[StandID]  # ["stand_A", "stand_B", ...]
    scenarios: dict[StandID, list[ScenarioID]]  # {"stand_A": ["scen_1", "scen_2"], ...}
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


def build_optimization_array(
    data_store: OutputDataStore,
    stand_areas: dict[StandID, float],
    optimization_variables: dict[NetcdfVariablePath, TargetVariableProperties],
) -> TargetVariableArrays:
    number_of_target_variables = len(optimization_variables)

    variables = []
    variables_properties = []
    for var_path, var_properties in optimization_variables.items():
        variables.append(var_path)
        variables_properties.append(var_properties)

    # Initialize output variable
    data = []

    for stand_i, stand_name in enumerate(data_store.stands):
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
                    * aggregate_time_series_to_float(
                        array=data_store.get_variable_value_for_scenario_and_stand(
                            variable_path=var_path,
                            stand_id=stand_name,
                            scenario_id=scenario_name,
                        ),
                        target_variable_properties=var_properties,
                    )
                )
        data.append(target_var_array)
    target_var_arrays = TargetVariableArrays(
        stands=data_store.stands,
        scenarios=data_store.scenarios,
        variables=variables,
        variables_properties=variables_properties,
        stand_areas=stand_areas,
        data_weighted_by_area=data,
    )
    target_var_arrays.validate()
    return target_var_arrays


# %% Get stand areas from json file derived from xml
JSON_FROM_XML_PATH = (
    AppSettings().project_root_path / "xmltoallometry_with_areas/extra_XML_info.json"
)
j = read_json_file(path=JSON_FROM_XML_PATH)


def _get_stand_area_from_json_file(json: dict, stand_number: int) -> float:
    return json["stand_datas"][str(stand_number)]["area"]


def _stand_id_to_stand_number(stand_id: StandID) -> int:
    return int(str(stand_id).split("_")[-1])


stand_areas_ha: dict[StandID, float] = {
    stand_id: _get_stand_area_from_json_file(
        json=j, stand_number=_stand_id_to_stand_number(stand_id)
    )
    for stand_id in data_store.stands
}

# %% Build target arrays
target_var_arrays = build_optimization_array(
    data_store=data_store,
    optimization_variables=TARGET_VARIABLES,
    stand_areas=stand_areas_ha,
)

# %% Pre-prune Pareto-dominated scenarios
from analysis.optimization.prepruning import preprune_pareto_dominated_scenarios

pruned_scenarios: dict[StandID, list[ScenarioID]] = {}
pruned_data: list[np.ndarray] = []

for (stand_name, scenario_names), stand_array in zip(
    target_var_arrays.scenarios.items(), target_var_arrays.data_weighted_by_area
):
    non_dominated_scenarios_index = preprune_pareto_dominated_scenarios(stand_array)
    pruned_scenarios[StandID(stand_name)] = [
        ScenarioID(scenario_names[i]) for i in non_dominated_scenarios_index
    ]

    pruned_data.append(stand_array[non_dominated_scenarios_index])

pruned_target_var_arrays = TargetVariableArrays(
    stands=target_var_arrays.stands,
    scenarios=pruned_scenarios,
    variables=target_var_arrays.variables,
    variables_properties=target_var_arrays.variables_properties,
    stand_areas=target_var_arrays.stand_areas,
    data_weighted_by_area=pruned_data,
)

pruned_scenarios_cardinality = compute_scenarios_cardinality(
    pruned_target_var_arrays.scenarios
)

n_total_combinations = product_of_elements_in_list(scenarios_cardinality)

print(f"\nTotal number of scenarios before pre-pruning:{sum(scenarios_cardinality)}")
print(
    "Total number of scenarios after  pre-pruning:{sum(pruned_scenarios_cardinality)}"
)
print(
    f"Number of pre-pruned scenarios: {sum(scenarios_cardinality) - sum(pruned_scenarios_cardinality)}"
)
print(
    f"\nNumber of possible combinations before pre-pruning: {n_total_combinations:.2e}"
)
print(
    f"Number of possible combinations after  pre-pruning: {product_of_elements_in_list(pruned_scenarios_cardinality):.2e}"
)
print(
    f"Number of combinations pruned: {n_total_combinations - product_of_elements_in_list(pruned_scenarios_cardinality):.2e}"
)


# %% target function


def create_random_organism(scenarios_cardinality: tuple[int, ...]) -> np.ndarray:
    return np.array([rng.integers(low=0, high=m) for m in scenarios_cardinality])


def create_random_population(
    scenarios_cardinality: tuple[int, ...], n_organisms: int
) -> np.ndarray:
    return np.stack(
        [create_random_organism(scenarios_cardinality) for _ in range(n_organisms)]
    )


def target_function_for_single_organism(
    configuration: np.ndarray,
    arrays_of_vars_of_interest: list[np.ndarray],
    number_of_target_variables: int,
) -> np.ndarray:
    chosen_vars = np.stack(
        [arr[idx] for arr, idx in zip(arrays_of_vars_of_interest, configuration)]
    )
    assert chosen_vars.shape == (
        len(arrays_of_vars_of_interest),
        number_of_target_variables,
    )

    return chosen_vars.sum(axis=0)


def make_target_function_numba_version(
    arrays_of_vars_of_interest: list[np.ndarray],
    number_of_target_variables: int,
    n_stands: int,
) -> Callable:
    """
    This function is used to create a Numba function that only takes the current configuration.
    The idea is to leverage the fact that array_of_vars_of_interest, number_of_target_variables and n_stands do not need to be recomputed each time: they can be known and fixed at compile time.
    """
    arrays_tuple = tuple(arrays_of_vars_of_interest)

    @jit(nopython=True)
    def target_function_for_single_organism(configuration: np.ndarray) -> np.ndarray:
        # Pre-allocate the result array
        chosen_vars = np.empty((n_stands, number_of_target_variables), dtype=np.float64)

        # Fill the array manually instead of using list comprehension + stack
        for i in range(n_stands):
            chosen_vars[i] = arrays_tuple[i][configuration[i]]

        return chosen_vars.sum(axis=0)

    return target_function_for_single_organism


def target_function_for_population(
    configurations: np.ndarray,
    arrays_of_vars_of_interest: list[np.ndarray],
    number_of_target_variables: int,
) -> np.ndarray:
    return np.array(
        [
            target_function_for_single_organism(
                config, arrays_of_vars_of_interest, number_of_target_variables
            )
            for config in configurations
        ]
    )


number_of_target_variables = len(target_var_arrays.variables)

random_organism = create_random_organism(pruned_scenarios_cardinality)
random_population = create_random_population(
    pruned_scenarios_cardinality, n_organisms=10
)

random_organism_fitness = target_function_for_single_organism(
    configuration=random_organism,
    arrays_of_vars_of_interest=pruned_target_var_arrays.data_weighted_by_area,
    number_of_target_variables=number_of_target_variables,
)
random_population_fitness = target_function_for_population(
    configurations=random_population,
    arrays_of_vars_of_interest=pruned_target_var_arrays.data_weighted_by_area,
    number_of_target_variables=number_of_target_variables,
)

# %% number of total candidates that can be evaluated per second
# This helps to choose the right optimization algorithm
N_STANDS = len(pruned_target_var_arrays.stands)

start = time.time()

N_ITER_TO_EVALUATE = int(1e4)

for i in range(N_ITER_TO_EVALUATE):
    random_organism = create_random_organism(pruned_scenarios_cardinality)

    random_organism_fitness = target_function_for_single_organism(
        configuration=random_organism,
        arrays_of_vars_of_interest=pruned_target_var_arrays.data_weighted_by_area,
        number_of_target_variables=number_of_target_variables,
    )
end = time.time()

print(
    f"Total runtime to evaluate {N_ITER_TO_EVALUATE} iterations without numba: {end - start} seconds."
)


# Numba time.
# Run once to JIT function
random_organism = create_random_organism(pruned_scenarios_cardinality)

target_numba_function = make_target_function_numba_version(
    arrays_of_vars_of_interest=pruned_target_var_arrays.data_weighted_by_area,
    number_of_target_variables=number_of_target_variables,
    n_stands=N_STANDS,
)

_ = target_numba_function(configuration=random_organism)

# Now start timer
start = time.time()

for i in range(N_ITER_TO_EVALUATE):
    random_organism = create_random_organism(pruned_scenarios_cardinality)

    random_organism_fitness = target_numba_function(configuration=random_organism)
end = time.time()

print(
    f"Total runtime to evaluate {N_ITER_TO_EVALUATE} iterations without numba: {end - start} seconds."
)

# %% pymoo
# NOTE: pymoo was made for floats, not ints. Putting int values is cumbersome, although there are some tutorials in the docs.
# Perhaps use DEAP instead?
# Or write my own algo?


class MyProblem(Problem):
    def __init__(
        self,
        arrays_of_vars_of_interest: list[np.ndarray],
        n_target_variables: int,
        scenarios_cardinality: Sequence[int],
    ):
        super().__init__(
            n_var=N_STANDS,
            n_obj=n_target_variables,
            xl=0,
            xu=scenarios_cardinality,
            vtype=int,
        )

        self.arrays_of_vars_of_interest = arrays_of_vars_of_interest
        self.n_target_variables = n_target_variables

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = target_function_for_population(
            configurations=x,
            arrays_of_vars_of_interest=self.arrays_of_vars_of_interest,
            number_of_target_variables=self.n_target_variables,
        )


problem = MyProblem(
    arrays_of_vars_of_interest=pruned_target_var_arrays.data_weighted_by_area,
    n_target_variables=number_of_target_variables,
    scenarios_cardinality=[
        c - 1 for c in pruned_scenarios_cardinality
    ],  # In pymoo, upper limits are inclusive. We need [0,4), not [0,4] for array indices
)

# pymoo was originally made for floats, but ints can still be used by
# rounding floats into ints. I don't like this!
# See the docs: https://pymoo.org/customization/discrete.html
algorithm = NSGA2(
    pop_size=100,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True,
)

res = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=("n_gen", 1000),
    seed=RANDOM_SEED,
    verbose=True,
)

# %% Visualize Pymoo solutions

variable_labels = [var_path.split("/")[-1] for var_path in target_var_arrays.variables]


def compute_single_variable_minima(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.argmin(array, axis=0) for array in arrays]).transpose()


def compute_single_variable_maxima(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.argmax(array, axis=0) for array in arrays]).transpose()


# Minima and maxima for each dimension
single_var_minima = np.stack(
    [
        target_numba_function(m)
        for m in compute_single_variable_minima(
            pruned_target_var_arrays.data_weighted_by_area
        )
    ]
)
single_var_maxima = np.stack(
    [
        target_numba_function(m)
        for m in compute_single_variable_maxima(
            pruned_target_var_arrays.data_weighted_by_area
        )
    ]
)


random_points = np.stack(
    [
        target_numba_function(config)
        for config in create_random_population(
            scenarios_cardinality=pruned_scenarios_cardinality, n_organisms=1000
        )
    ]
)


# Multi-dimension scatter
plot = Scatter(tight_layout=True, labels=variable_labels, plot_3d=False, legend=True)
plot.add(res.F, label="Pareto", color="blue")
plot.add(single_var_minima, label="single_var_minima", color="red")
# plot.add(single_var_maxima, label="single_var_maxima", color="green")
plot.add(random_points, label="random", color="orange")
plot.show()

# %% Dynamic Programming
from analysis.optimization.dynamic_programming import (
    shift_points_to_positive_values,
    get_minimum_values_per_variable,
    from_numpy_arrays_to_nested_tuples,
    from_numpy_arrays_to_nested_lists,
    find_pareto_front,
    reconstruct_solution_pareto_front,
    compute_objective,
)


def shift_table_of_objectives_to_positive_values(
    table_of_objectives: list[np.ndarray],
) -> list[np.ndarray]:
    minimum_values_per_variable = get_minimum_values_per_variable(table_of_objectives)
    assert len(minimum_values_per_variable) == table_of_objectives[0].shape[1]
    return shift_points_to_positive_values(
        data=table_of_objectives,
        minimum_values_per_variable=minimum_values_per_variable,
    )


data_table_arrays = pruned_target_var_arrays.data_weighted_by_area

# Shift all values to positive so that there are no problems with negative logs() later.
# At the end, will have to undo everything to report results in the original scale.

# or linear transformations. That's why we can do it.
shifted = shift_table_of_objectives_to_positive_values(data_table_arrays)


# Here I switch from numpy-centric to Python native.
# Because later I will probably want to write this algo in a compiled language.
shifted_data_table = from_numpy_arrays_to_nested_tuples(shifted)

import time

t0 = time.perf_counter()
pareto_front = find_pareto_front(shifted_data_table, epsilon=1e-7)

pareto_front_solution = reconstruct_solution_pareto_front(
    pareto_front=pareto_front,
    data_table=from_numpy_arrays_to_nested_tuples(
        pruned_target_var_arrays.data_weighted_by_area
    ),
)
t1 = time.perf_counter()
print(f"Python: {t1 - t0:.4f}s")

# %% Dynamic programming Rust
import pareto_dp

data_table_list = from_numpy_arrays_to_nested_lists(data_table_arrays)

t2 = time.perf_counter()
rust_front = pareto_dp.find_pareto_front(data=data_table_list, epsilon=1e-7)
t3 = time.perf_counter()

print(f"Rust:   {t3 - t2:.4f}s")

print(f"Speedup Rust vs Python: {(t1 - t0) / (t3 - t2):.1f}x")

# %% create random points
random_design_vectors = create_random_population(
    scenarios_cardinality=pruned_scenarios_cardinality, n_organisms=10000
)
random_target_vectors = [
    compute_objective(
        design_vector=vec,
        data_table=from_numpy_arrays_to_nested_tuples(data_table_arrays),
    )
    for vec in random_design_vectors
]


# %% visualize dynamic programming
from analysis.optimization.pareto_corner_plot import pareto_corner_plot


pareto_front_objectives = [point.target_vector for point in pareto_front_solution]

pareto_front_objectives_array = np.array(pareto_front_objectives)


fig = pareto_corner_plot(
    data=pareto_front_objectives_array,
    random_points=np.array(random_target_vectors),
    labels=list(target_var_arrays.variables),
    show_diagonal=False,
)
plt.show()

# %% Visualize Rust Pareto front

rust_front_objectives = [point.target_vector for point in rust_front]

rust_front_objectives_array = np.array(rust_front_objectives)


fig = pareto_corner_plot(
    data=rust_front_objectives_array,
    random_points=np.array(random_target_vectors),
    labels=list(target_var_arrays.variables),
    show_diagonal=False,
)
plt.show()
