# %% imports
import argparse
from multiprocessing import Pool

from susi.io.execution_config import SimulationParams, MultipleSusis
from susi.core.susi_utils import read_FMI_weather
from susi.core.susi_main import Susi
from inputs.parameters import golden_test
from susi.io.metadata_model import SimulationMetaData
from susi.io.susi_parameter_model import SusiParams

# %% Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "n_parallel_processes", help="Number of parallel processes", type=int
)
cli_args = parser.parse_args()

# %% Create the scenarios


# This is the best way I found to create multiple parameter models based on one:
# First, define a function to be able to do this repeatedly
def create_strip_scenarios(base_params: SusiParams, L_value: float) -> SusiParams:
    """
    Example function to create new parameter models based on another one.
    This function changes the value of the parameter SusiParams.site_parameters.L,
    and returns a fully validated model.
    """

    # 1. Get parameters of the base model into a Python dictionary
    data = base_params.model_dump(exclude_computed_fields=True)

    # 2. Modify the Python dictionary. Here we choose to change the L parameter
    data["site_parameters"]["L"] = L_value

    # 3. Validate the model to check that you did not make a mistake
    return SusiParams.model_validate(data)


# Next, create the scenarios
long_strip = create_strip_scenarios(base_params=golden_test.PARAMETERS, L_value=60.0)
short_strip = create_strip_scenarios(base_params=golden_test.PARAMETERS, L_value=20.0)


# Finally, create the list of parameters that will go into the susi simulation
all_parameters = [
    SimulationParams(
        metadata=SimulationMetaData(),
        susi_params=short_strip,
    ),
    SimulationParams(metadata=SimulationMetaData(), susi_params=long_strip),
]

execution_config = MultipleSusis(
    simulation_parameter_list=all_parameters,
    n_parallel_processes=cli_args.n_parallel_processes,
)


def run_susi(simulation_parameters: SimulationParams) -> None:
    # Initiate susi class
    susi = Susi(simulation_parameters)

    # Create output folder where results go
    susi.create_output_folder()

    # Run simulation
    susi.run()

    # Save stuff
    susi.write_params_and_metadata()


# %% Execute parallel processing

pool = Pool(processes=execution_config.n_parallel_processes)
pool.map(func=run_susi, iterable=execution_config.simulation_parameter_list)
