from multiprocessing import Pool

from susi.io.execution_config import SimulationParams, MultipleSusis
from susi.core.susi_main import Susi
from susi.io.metadata_model import SimulationMetaData

from susi.io.app_settings import AppSettings
from inputs.parameters import para_2021

_app_settings = AppSettings()

# ***************** local call for SUSI*****************************************************
folderName = (
    r"C:/Users/laurenan/OneDrive - University of Helsinki/SUSI/vesitase/vesitase_out/"
)

wpath = r"C:/Users/laurenan/OneDrive - University of Helsinki/SUSI/vesitase/vesitase_wfiles/"
mottipath = (
    r"C:/Users/laurenan/OneDrive - University of Helsinki/SUSI/vesitase/motti_files/"
)

SITE_LABELS = [
    "ansa21",
    "ansa26",
    "jaakkoin61",
    "jaakkoin62",
    "koira11",
    "koira12",
    "neva11",
    "neva14",
    "neva31",
    "neva34",
    "parkano11",
]


def create_all_simulation_params(site_labels: list[str]) -> list[SimulationParams]:

    all_parameters: list[SimulationParams] = []

    for site_label in site_labels:
        stand_label = site_label[:-2]
        scenario_label = site_label[-2:]

        all_parameters.append(
            SimulationParams(
                metadata=SimulationMetaData(
                    experiment_id="susi_2021",
                    stand_id=stand_label,
                    scenario_id=scenario_label,
                ),
                susi_params=para_2021.assign_susi_params_for_site(site_label),
            ),
        )
    return all_parameters


# Create list with all parameters to use in the simulations
all_parameters = create_all_simulation_params(SITE_LABELS)


# %%
# Finally, create the list of parameters that will go into the susi simulation

execution_config = MultipleSusis(
    simulation_parameter_list=all_parameters,
    n_parallel_processes=6,
)


def run_susi(simulation_parameters: SimulationParams) -> None:
    # Initiate susi class
    susi = Susi(simulation_parameters)

    # Run simulation
    susi.run()

    # Save stuff
    susi.write_params_and_metadata()


# %% Execute parallel processing
if __name__ == "__main__":
    with Pool(processes=execution_config.n_parallel_processes) as pool:
        pool.map(func=run_susi, iterable=execution_config.simulation_parameter_list)
