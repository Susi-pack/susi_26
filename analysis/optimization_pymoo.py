# %%
from typing import Callable, Sequence
from dataclasses import dataclass, fields, astuple
import numpy as np
from pathlib import Path
import netCDF4
import pandas as pd
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


# %% Read netcdf file structure


# All Susi netcdf files has the same structure.
# Pick one and get the group/variables hierarchical structure.
# This will be useful in choosing what variables to read later.
sample_netcdf_filepath = metadata_by_stand[0].iloc[0]["netcdf_output_filepath"]

all_variables = nc_utils.list_all_netcdf_variables(sample_netcdf_filepath)

var_values = nc_utils.read_value_netcdf_variable(
    netcdf_filepath=sample_netcdf_filepath, variable=all_variables[10]
)


# %% Read ncdf data into python dictionary with netcdf

# TODO: read each interesting variable from netcdf file in the same loop.
# Do so using a context manager.
# Otherwise, the netcdf files stay open and consume too much memory.
# So: with open file:
#       get necessary vars into list of dicts (above called vars_of_interest_by_stand)


def get_netcdf_variables(netcdf_filepath: Path) -> netCDF4.Dataset:
    with netCDF4.Dataset(netcdf_filepath, "r") as ds:
        variables = ds.get_variables_by_attributes()

    return variables


with netCDF4.Dataset(sample_netcdf_filepath, "r") as ds:
    variables = ds.variables
    print(variables)


sample_netcdf_variables = get_netcdf_variables(sample_netcdf_filepath)

get_netcdf_variables(sample_netcdf)


def read_netcdf_files_for_stand(
    metadata_df: pd.DataFrame,
) -> dict[str : netCDF4.Dataset]:
    # Netcdf data is saved in a dictionary where the experiment folder path is the key.
    data = {}
    for _, experiment_info in metadata_df.iterrows():
        data[experiment_info["experiment_id"]] = netCDF4.Dataset(
            experiment_info["netcdf_output_filepath"], "r"
        )

    return data


all_variables_by_stand = [0] * N_STANDS

for n_stand, metadata_df in enumerate(metadata_by_stand):
    all_variables_by_stand[n_stand] = read_netcdf_files_for_stand(metadata_df)


# %% interesting_variables by stand
def get_last_year_values(var: float) -> np.ndarray:
    return var[:, -1, :]


def get_scenario_name_from_path(path: str) -> str:
    return path.split("_")[-1]


@dataclass
class TargetVariables:
    volume: float
    soil_c_balance: float
    dwtyr: float


@dataclass
class Scenario:
    name: str  # DNM, default, fertilized, etc.
    target_variables: TargetVariables


# Container class to preserve both arrays and names
@dataclass
class ScenarioArrayData:
    arrays: list[np.ndarray]  # One array per stand
    scenario_names: list[list[str]]  # Names for each scenario in each stand


def transform_list_of_scenarios_to_array_data(
    vars_of_interest_by_stand: list[list[Scenario]], n_stands: int
) -> ScenarioArrayData:
    target_variables: list[np.ndarray] = []
    scenario_names: list[list[str]] = []

    number_of_target_variables = len(fields(TargetVariables))

    for n_stand in range(n_stands):
        vars_of_interest_for_stand = vars_of_interest_by_stand[n_stand]
        number_of_scenarios = len(vars_of_interest_for_stand)

        # Initialize the array that holds the variables' values
        target_var = np.ones((number_of_scenarios, number_of_target_variables)) * np.nan
        names_for_stand = []

        for row, scenario in enumerate(vars_of_interest_for_stand):
            # Convert dataclass to tuple automatically - works for any number of fields!
            target_var[row, :] = astuple(scenario.target_variables)
            names_for_stand.append(scenario.name)

        # Check no field without filling
        assert not np.isnan(target_var).any()

        target_variables.append(target_var)
        scenario_names.append(names_for_stand)

    return ScenarioArrayData(arrays=target_variables, scenario_names=scenario_names)


def transform_array_data_to_list_of_scenarios(
    array_data: ScenarioArrayData, n_stands: int
) -> list[list[Scenario]]:
    vars_of_interest_by_stand: list[list[Scenario]] = []

    # Get the field names from the dataclass
    target_var_fields = [f.name for f in fields(TargetVariables)]

    for n_stand in range(n_stands):
        scenarios_for_stand = []

        for row_idx, target_vars_of_stand in enumerate(array_data.arrays[n_stand]):
            # Create TargetVariables using field names dynamically
            target_vars_dict = {
                field_name: target_vars_of_stand[i]
                for i, field_name in enumerate(target_var_fields)
            }

            # Create Scenario object
            scenario = Scenario(
                name=array_data.scenario_names[n_stand][row_idx],
                target_variables=TargetVariables(**target_vars_dict),
            )
            scenarios_for_stand.append(scenario)

        vars_of_interest_by_stand.append(scenarios_for_stand)

    return vars_of_interest_by_stand


# Get vars of interest
vars_of_interest_by_stand: list[list[Scenario]] = [[] for _ in range(0, N_STANDS)]

for n_stand in range(N_STANDS):
    for scenario_name, netcdf_data in all_variables_by_stand[n_stand].items():
        # Get variable values
        volume = netcdf_data.groups["stand"].variables["volume"][:]

        # Negative sign because we actually want to maximize volume,
        # but it is a minimization problem
        total_last_year_volume = -np.sum(get_last_year_values(volume))

        soil_c_balance_co2eq = (
            netcdf_data.groups["balance"].groups["C"].variables["soil_c_balance_co2eq"]
        )
        total_last_year_soil_c_balance = np.sum(
            get_last_year_values(soil_c_balance_co2eq)
        )

        # Negative sign too
        dwtyr = netcdf_data.groups["strip"].variables["dwtyr"][:]
        average_last_year_dwtyr = -np.mean(get_last_year_values(dwtyr))

        # Add to list
        vars_of_interest_by_stand[n_stand].append(
            Scenario(
                name=scenario_name,
                target_variables=TargetVariables(
                    volume=total_last_year_volume,
                    soil_c_balance=total_last_year_soil_c_balance,
                    dwtyr=average_last_year_dwtyr,
                ),
            )
        )
# %% Transformation between data representations: list of scenarios to list of numpys
# Usage example:
# Forward transformation
array_data = transform_list_of_scenarios_to_array_data(
    vars_of_interest_by_stand, N_STANDS
)

# We have both arrays and names in one object
arrays: list[np.ndarray] = array_data.arrays
names = array_data.scenario_names

# Reverse transformation
reconstructed = transform_array_data_to_list_of_scenarios(array_data, N_STANDS)

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


number_of_target_variables = len(fields(TargetVariables))

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
    termination=("n_gen", 10000),
    seed=RANDOM_SEED,
    verbose=True,
)

# %% Visualize solutions


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


target_variable_names = [v.name for v in fields(TargetVariables)]

plt.figure(figsize=(7, 5))
plt.scatter(res.F[:, 0], res.F[:, 1], s=30, edgecolors="blue", label="Pareto")
plt.scatter(
    random_points[:, 0],
    random_points[:, 1],
    s=30,
    facecolors="none",
    edgecolors="orange",
    label="random",
)
plt.scatter(
    single_var_minima[:, 0],
    single_var_minima[:, 1],
    s=30,
    edgecolors="red",
    label="single_var_minima",
)
plt.scatter(
    single_var_maxima[:, 0],
    single_var_maxima[:, 1],
    s=30,
    edgecolors="green",
    label="single_var_maxima",
)
plt.xlabel(target_variable_names[0])
plt.ylabel(target_variable_names[1])

plt.title("Objective Space")

plt.legend()
plt.show()

# Multi-dimension scatter
plot = Scatter(
    tight_layout=True, labels=target_variable_names, plot_3d=False, legend=True
)
plot.add(res.F, label="Pareto", color="blue")
plot.add(single_var_minima, label="single_var_minima", color="red")
plot.add(single_var_maxima, label="single_var_maxima", color="green")
plot.add(random_points, label="random", color="orange")
plot.show()
