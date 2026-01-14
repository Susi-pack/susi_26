# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:42 2020

@author: alauren
"""

# THIS files
from susi.io.execution_config import SimulationParams, MultipleSusis
from susi.core.susi_utils import read_FMI_weather
from susi.core.susi_main import Susi
from inputs.parameters import golden_test
from susi.io.metadata_model import SimulationMetaData

# ***************** local call for SUSI*****************************************************

# read weather input
forc = read_FMI_weather(
    ID=0,
    start_date=golden_test.PARAMETERS.simulation_config.start_date,
    end_date=golden_test.PARAMETERS.simulation_config.end_date,
    sourcefile=golden_test.PARAMETERS.weather_parameters.FMI_weather_filepath,
)
# Specifies all parameters needed for a single run
simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(), susi_params=golden_test.PARAMETERS
)


def call_susi(simu_params: SimulationParams, forc):
    # Initiate susi class
    susi = Susi(
        simulation_parameters=simu_params,
        weather_forcing=forc,
    )

    # Create output folder where results go
    susi.create_output_folder()

    # Run simulation
    susi.run()

    # Save stuff
    susi.write_params_and_metadata()


# %% Using normal single call

call_susi(simu_params=simulation_parameters, forc=forc)

# %% Using MultipleSusis
# We can also use MultipleSusis also to run a single susi instance.
# This will be more useful when running several, of course!
execution_config = MultipleSusis(
    n_runs=1, simulation_parameter_list=[simulation_parameters]
)

for n_run, simu_params in enumerate(execution_config.simulation_parameter_list):
    print(f"Starting SUSI run number {n_run}")

    call_susi(simu_params=simulation_parameters, forc=forc)

    print(f"Ended SUSI run number {n_run}")
