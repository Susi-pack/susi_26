# %%
from susi.io.utils import read_json_file
from typing import Callable, Sequence, Literal, assert_never, NewType
from dataclasses import dataclass
import numpy as np
from numba import jit
import time
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
VARS_OF_INTEREST = (
    NetcdfVariablePath("/stand/volume"),
    NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"),
)

data_store: load_output.OutputDataStore = (
    load_output.read_netcdf_files_for_selected_variables(
        selected_variables=VARS_OF_INTEREST, metadata_by_stand=metadata_by_stand
    )
)


# %% compute total number of combinations
def product_of_elements_in_list(list_of_numbers: Sequence[int | float]) -> int | float:
    product = 1
    for element in list_of_numbers:
        product *= element
    return product


scenarios_cardinality: tuple[int, ...] = tuple(
    [len(scenarios) for stand, scenarios in data_store.scenarios.items()]
)
n_total_combinations = product_of_elements_in_list(scenarios_cardinality)

print("\nNumber of scenarios for each stand:")
print(scenarios_cardinality)
print(f"\nNumber of total combinations: {n_total_combinations:.2e}")

# %% interesting_variables by stand

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


PROPERTIES_OF_TARGET_VARIABLES = {
    load_output.NetcdfVariablePath("/stand/volume"): TargetVariableProperties(
        aggregation_method="mean_last_timestep",
        invert_optimization=True,
    ),
    load_output.NetcdfVariablePath(
        "/balance/C/soil_c_balance_co2eq"
    ): TargetVariableProperties(
        aggregation_method="mean_all",
        invert_optimization=True,
    ),
}


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

    data: list[np.ndarray]

    def validate(self):
        # 1st level corresponds to stands
        if len(self.data) != len(self.stands):
            raise ValueError(
                "TargetVariableArray not initialized properly. First dimension must be stands."
            )
        # rows correspond to scenarios; columns to target variables
        for n_stand, scenarios in enumerate(self.scenarios.values()):
            if self.data[n_stand].shape[0] != len(scenarios):
                raise ValueError(
                    f"TargetVariableArray not initialized properly. Rows must correspond to scenarios. Found mismatch in stand number {n_stand}"
                )
            if self.data[n_stand].shape[1] != len(self.variables):
                raise ValueError(
                    f"TargetVariableArray not initialized properly. Columns must correspond to variables. Found mismatch in stand number {n_stand}"
                )
            if self.data[n_stand].ndim != 2:
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
        data=data,
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

target_var_arrays = build_optimization_array(
    data_store=data_store,
    optimization_variables=PROPERTIES_OF_TARGET_VARIABLES,
    stand_areas=stand_areas_ha,
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


number_of_target_variables = len(VARS_OF_INTEREST)

random_organism = create_random_organism(scenarios_cardinality)
random_population = create_random_population(scenarios_cardinality, n_organisms=10)

random_organism_fitness = target_function_for_single_organism(
    configuration=random_organism,
    arrays_of_vars_of_interest=target_var_arrays.data,
    number_of_target_variables=number_of_target_variables,
)
random_population_fitness = target_function_for_population(
    configurations=random_population,
    arrays_of_vars_of_interest=target_var_arrays.data,
    number_of_target_variables=number_of_target_variables,
)

# %% number of total candidates that can be evaluated per second
# This helps to choose the right optimization algorithm
N_STANDS = len(target_var_arrays.stands)

start = time.time()

N_ITER_TO_EVALUATE = int(1e4)

for i in range(N_ITER_TO_EVALUATE):
    random_organism = create_random_organism(scenarios_cardinality)

    random_organism_fitness = target_function_for_single_organism(
        configuration=random_organism,
        arrays_of_vars_of_interest=target_var_arrays.data,
        number_of_target_variables=number_of_target_variables,
    )
end = time.time()

print(
    f"Total runtime to evaluate {N_ITER_TO_EVALUATE} iterations without numba: {end - start} seconds."
)


# Numba time.
# Run once to JIT function
random_organism = create_random_organism(scenarios_cardinality)

target_numba_function = make_target_function_numba_version(
    arrays_of_vars_of_interest=target_var_arrays.data,
    number_of_target_variables=number_of_target_variables,
    n_stands=N_STANDS,
)

_ = target_numba_function(configuration=random_organism)

# Now start timer
start = time.time()

for i in range(N_ITER_TO_EVALUATE):
    random_organism = create_random_organism(scenarios_cardinality)

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
    arrays_of_vars_of_interest=target_var_arrays.data,
    n_target_variables=number_of_target_variables,
    scenarios_cardinality=[
        c - 1 for c in scenarios_cardinality
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

# %% Visualize solutions

variable_labels = [var_path.split("/")[-1] for var_path in VARS_OF_INTEREST]


def compute_single_variable_minima(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.argmin(array, axis=0) for array in arrays]).transpose()


def compute_single_variable_maxima(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.argmax(array, axis=0) for array in arrays]).transpose()


# Minima and maxima for each dimension
single_var_minima = np.stack(
    [
        target_numba_function(m)
        for m in compute_single_variable_minima(target_var_arrays.data)
    ]
)
single_var_maxima = np.stack(
    [
        target_numba_function(m)
        for m in compute_single_variable_maxima(target_var_arrays.data)
    ]
)


random_points = np.stack(
    [
        target_numba_function(config)
        for config in create_random_population(
            scenarios_cardinality=scenarios_cardinality, n_organisms=1000
        )
    ]
)


# Multi-dimension scatter
plot = Scatter(tight_layout=True, labels=variable_labels, plot_3d=False, legend=True)
plot.add(res.F, label="Pareto", color="blue")
plot.add(single_var_minima, label="single_var_minima", color="red")
plot.add(single_var_maxima, label="single_var_maxima", color="green")
plot.add(random_points, label="random", color="orange")
plot.show()
