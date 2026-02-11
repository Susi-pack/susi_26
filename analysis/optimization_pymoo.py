# %%
from typing import Callable, Sequence
from dataclasses import dataclass, fields
from enum import Enum
import numpy as np
from pathlib import Path
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
import matplotlib.pyplot as plt

import susi.io.netcdf_utils as nc_utils
from susi.io.app_settings import AppSettings


# %% Load paroninkorpi metadata


def stand_foldername_from_number(stand_number: int) -> str:
    return f"paroninkorpi/stand_{stand_number:02d}"


def stand_folderpath_from_number(stand_number: int) -> Path:
    return AppSettings().output_folder / stand_foldername_from_number(stand_number)


N_STANDS = 21
stand_folderpaths = [
    stand_folderpath_from_number(stand_n) for stand_n in range(1, N_STANDS + 1)
]

metadata_by_stand = nc_utils.load_all_metadatas_from_folders(folders=stand_folderpaths)

# %% compute total number of combinations


def product_of_elements_in_list(list_of_numbers: Sequence[int | float]) -> int | float:
    product = 1
    for element in list_of_numbers:
        product *= element
    return product


scenarios_cardinality = tuple(
    [stand_metadata_df.shape[0] for stand_metadata_df in metadata_by_stand]
)
n_total_combinations = product_of_elements_in_list(scenarios_cardinality)

print("\nNumber of scenarios for each stand:")
print(scenarios_cardinality)
print(f"\nNumber of total combinations: {n_total_combinations:.2e}")


# %% Read netcdf file structure and choose (filter) variables
# All Susi netcdf files has the same structure.
# Pick one and get the group/variables hierarchical structure.
# This will be useful in choosing what variables to read later.
sample_netcdf_filepath = metadata_by_stand[0].iloc[0]["netcdf_output_filepath"]

all_variables = nc_utils.list_all_netcdf_variables(sample_netcdf_filepath)

# Filter the interesting variables
INTERESTING_VAR_PATHS: list[nc_utils.NetcdfVariablePath] = (
    "/stand/volume",
    "/balance/C/soil_c_balance_co2eq",
)

chosen_vars = nc_utils.choose_netcdf_vars_by_path(
    paths=INTERESTING_VAR_PATHS, all_variables=all_variables
)

# list (dimension n_sites) of dicts (scenarios for each site).
# Dicts hold arrays with the value of variables
chosen_variables_by_stand_and_scenario = (
    nc_utils.read_chosen_variables_from_netcdf_by_stands_and_scenarios(
        chosen_vars=chosen_vars, metadata_by_stand=metadata_by_stand
    )
)


# %% interesting_variables by stand
def get_last_year_values(var: np.ndarray) -> np.ndarray:
    return var[:, -1, :]


def average_of_last_year_values(var: np.ndarray) -> float:
    return np.mean(get_last_year_values(var))


def average_of_all_values(var: np.ndarray) -> float:
    return np.mean(var)


def get_scenario_name_from_path(path: str) -> str:
    return path.split("_")[-1]


class TimeSeriesAggregatingFunction(Enum):
    AVERAGE_OF_LAST_YEAR = average_of_last_year_values
    AVERAGE_OF_HISTORY = average_of_all_values


@dataclass
class TargetVariableProperties:
    func_to_aggregate_data: TimeSeriesAggregatingFunction
    invert_optimization: bool  # If True, this puts a minus sign in the value: turn maximization into minimization


def aggregate_time_series_to_float(
    time_series: np.ndarray, target_variable_properties: TargetVariableProperties
) -> float:
    inverter: int = -1 if target_variable_properties.invert_optimization else 1

    return inverter * target_variable_properties.func_to_aggregate_data(time_series)


def compute_target_variables_from_netcdf_values(
    netcdf_variables: list[nc_utils.NetcdfVariableValue],
    properties_of_target_variables: dict[
        nc_utils.NetcdfVariablePath, TargetVariableProperties
    ],
):
    return {
        netcdf_variable.path: aggregate_time_series_to_float(
            time_series=netcdf_variable.value,
            target_variable_properties=PROPERTIES_OF_TARGET_VARIABLES[
                netcdf_variable.path
            ],
        )
        for netcdf_variable in netcdf_variables
    }


PROPERTIES_OF_TARGET_VARIABLES = {
    "/stand/volume": TargetVariableProperties(
        func_to_aggregate_data=TimeSeriesAggregatingFunction.AVERAGE_OF_LAST_YEAR,
        invert_optimization=True,
    ),
    "/balance/C/soil_c_balance_co2eq": TargetVariableProperties(
        func_to_aggregate_data=TimeSeriesAggregatingFunction.AVERAGE_OF_HISTORY,
        invert_optimization=True,
    ),
}

# Get vars of interest
vars_of_interest_by_stand: list[
    dict[nc_utils.ScenarioName, nc_utils.TargetVariableDict]
] = [{} for _ in range(0, N_STANDS)]


for n_stand in range(N_STANDS):
    for scenario_name, netcdf_variables in chosen_variables_by_stand_and_scenario[
        n_stand
    ].items():
        vars_of_interest_by_stand[n_stand][scenario_name] = (
            compute_target_variables_from_netcdf_values(
                netcdf_variables=netcdf_variables,
                properties_of_target_variables=PROPERTIES_OF_TARGET_VARIABLES,
            )
        )
# %% Transformation between data representations: list of scenarios to list of numpys


def transform_list_of_scenarios_to_optimization_array_structure(
    vars_of_interest_by_stand: list[
        dict[nc_utils.ScenarioName, nc_utils.TargetVariableDict]
    ],
    n_stands: int,
    target_variable_paths: list[nc_utils.NetcdfVariablePath],
) -> nc_utils.ScenarioArrayData:
    target_variable_arrays: list[np.ndarray] = []

    scenario_names: list[list[nc_utils.ScenarioName]] = []

    n_target_variables = len(target_variable_paths)

    for n_stand in range(n_stands):
        vars_of_interest_single_stand = vars_of_interest_by_stand[n_stand]
        number_of_scenarios = len(vars_of_interest_single_stand)

        # Initialize the array that holds the variables' values
        target_var = np.ones((number_of_scenarios, n_target_variables)) * np.nan
        names_for_stand = []

        for row, (scenario_name, target_variable_dict) in enumerate(
            vars_of_interest_single_stand.items()
        ):
            names_for_stand.append(scenario_name)

            for col, var_path in enumerate(target_variable_paths):
                target_var[row, col] = target_variable_dict[var_path]

        # Check no field without filling
        # assert not np.isnan(target_var).any()

        target_variable_arrays.append(target_var)
        scenario_names.append(names_for_stand)

    return nc_utils.ScenarioArrayData(
        arrays=target_variable_arrays, scenario_names=scenario_names
    )


def transform_array_data_to_list_of_scenarios(
    array_data: nc_utils.ScenarioArrayData,
    n_stands: int,
    target_variable_paths: list[nc_utils.NetcdfVariablePath],
) -> list[dict[nc_utils.ScenarioName, nc_utils.TargetVariableDict]]:
    vars_of_interest_by_stand: list[
        dict[nc_utils.ScenarioName, nc_utils.TargetVariableDict]
    ] = []

    for n_stand in range(n_stands):
        scenarios_for_stand = {}

        for row_idx, scenarios_data in enumerate(array_data.arrays[n_stand]):
            scenario_name = array_data.scenario_names[n_stand][row_idx]
            target_variable_values = {
                target_var_path: scenarios_data[col_idx]
                for (col_idx, target_var_path) in enumerate(target_variable_paths)
            }

            scenarios_for_stand[scenario_name] = target_variable_values

        vars_of_interest_by_stand.append(scenarios_for_stand)

    return vars_of_interest_by_stand


# Usage example:
# Forward transformation
array_data = transform_list_of_scenarios_to_optimization_array_structure(
    vars_of_interest_by_stand=vars_of_interest_by_stand,
    n_stands=N_STANDS,
    target_variable_paths=INTERESTING_VAR_PATHS,
)

# We have both arrays and names in one object
arrays: list[np.ndarray] = array_data.arrays
names = array_data.scenario_names

# Reverse transformation
reconstructed = transform_array_data_to_list_of_scenarios(
    array_data=array_data,
    n_stands=N_STANDS,
    target_variable_paths=INTERESTING_VAR_PATHS,
)

assert reconstructed == vars_of_interest_by_stand


# %% target function
RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)


def create_random_organism(scenarios_cardinality: tuple[int]) -> np.ndarray:
    return np.array([rng.integers(low=0, high=m) for m in scenarios_cardinality])


def create_random_population(
    scenarios_cardinality: tuple[int], n_organisms: int
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
    assert chosen_vars.shape == (N_STANDS, number_of_target_variables)

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


number_of_target_variables = len(INTERESTING_VAR_PATHS)

random_organism = create_random_organism(scenarios_cardinality)
random_population = create_random_population(scenarios_cardinality, n_organisms=10)

random_organism_fitness = target_function_for_single_organism(
    configuration=random_organism,
    arrays_of_vars_of_interest=arrays,
    number_of_target_variables=number_of_target_variables,
)
random_population_fitness = target_function_for_population(
    configurations=random_population,
    arrays_of_vars_of_interest=arrays,
    number_of_target_variables=number_of_target_variables,
)

# %% number of total candidates that can be evaluated per second
# This helps to choose the right optimization algorithm

start = time.time()

N_ITER_TO_EVALUATE = int(1e4)

for i in range(N_ITER_TO_EVALUATE):
    random_organism = create_random_organism(scenarios_cardinality)

    random_organism_fitness = target_function_for_single_organism(
        configuration=random_organism,
        arrays_of_vars_of_interest=arrays,
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
    arrays_of_vars_of_interest=arrays,
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

# %% pymoo (not started yet)
# NOTE: pymoo was made for floats, not ints. Putting int values is cumbersome, although there are some tutorials in the docs.
# Perhaps use DEAP instead?
# Or write my own algo?


class MyProblem(Problem):
    def __init__(
        self,
        arrays_of_vars_of_interest: list[np.ndarray],
        n_target_variables: int,
        scenarios_cardinality: tuple[int],
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
    arrays_of_vars_of_interest=arrays,
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

variable_labels = [var_path.split("/")[-1] for var_path in INTERESTING_VAR_PATHS]


def compute_single_variable_minima(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.argmin(array, axis=0) for array in arrays]).transpose()


def compute_single_variable_maxima(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.argmax(array, axis=0) for array in arrays]).transpose()


# Minima and maxima for each dimension
single_var_minima = np.stack(
    [target_numba_function(m) for m in compute_single_variable_minima(arrays)]
)
single_var_maxima = np.stack(
    [target_numba_function(m) for m in compute_single_variable_maxima(arrays)]
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
