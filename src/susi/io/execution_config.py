from streamlit.delta_generator import Value
from pydantic import BaseModel, Field, model_validator
import json


from susi.io.extra_pydantic_types import PositiveInt
from susi.io.metadata_model import SimulationMetaData
from susi.io.susi_parameter_model import SusiParams


class SimulationParams(BaseModel):
    """
    Fully specifies the parameters needed for a single SUSI simulation.
    Contains SUSI parameters and metadata associated with each simulation.
    """

    susi_params: SusiParams
    metadata: SimulationMetaData


class MultipleSusis(BaseModel):
    """
    Highest abstraction layer for the input parameters.

    Enables the creation of multiple SUSI simulations.
    """

    n_parallel_processes: PositiveInt = Field(
        ge=1,
        le=40,
        frozen=True,
        description="Number of parallel processes to spawn. Cannot be greater than the number of simulations. Must be higher than 1, 40 at most (40 is the maximum number of cores in a single node, at least in CSC).",
    )
    simulation_parameter_list: list[SimulationParams] = Field(
        description="A list of parameters fully specifying each run. The length of the list determines the number of simulations than will be run, regardless of how many parallel processes are specified in `n_parallel_processes`. E.g., a `simulation_parameter_list` of length 2 will execute 2 full susi runs no matter the number of parallel processes specified (which in such case can only be either 1 or 2).",
    )

    def _check_not_more_processes_than_runs(self) -> None:
        """
        'n_parallel_processes' cannot be greater than 'n_runs'. There must be at most one process per run.
        """
        if self.n_parallel_processes > len(self.simulation_parameter_list):
            raise ValueError(
                "'n_parallel_processes' cannot be greater than 'n_runs'. There must be at most one process per run."
            )
        return None

    def _check_for_duplicated_susi_params(self) -> None:
        """
        We don't want to run two simulations with exactly the same parameters.
        This function checks for SUSI parameter duplicates in the list of runs.
        """
        seen = set()

        for simulation_run in self.simulation_parameter_list:
            # Convert to JSON string for hashing (handles nested structures)
            serialized = json.dumps(
                simulation_run.susi_params.model_dump(mode="json"), sort_keys=True
            )

            if serialized in seen:
                raise ValueError("Duplicate Susi Parameter models detected.")
            seen.add(serialized)
        return None

    def _check_for_duplicated_experiment_folder_paths(self) -> None:
        """
        We don't want two Susi simulations to write outputs to the same folder,
        for this would overwrite one with the other.
        This function checks for output folder path parameter duplicates
        in the metadatas of the list of runs.
        """
        seen = set()

        for simulation_run in self.simulation_parameter_list:
            experiment_folder_path = simulation_run.metadata.experiment_folder_path

            if experiment_folder_path in seen:
                raise ValueError("Duplicate experiment folder paths detected.")

            else:
                seen.add(experiment_folder_path)

        return None

    def _check_stand_and_scenario_ids_are_set(self) -> None:
        """
        All runs in MultipleSusi must have a stand_id and a scenario_id.
        """
        for simulation_run in self.simulation_parameter_list:
            stand_id = simulation_run.metadata.stand_id
            scenario_id = simulation_run.metadata.scenario_id

            if stand_id is None or scenario_id is None:
                raise ValueError(
                    "SimulationMetadata without `stand_id` and/or `scenario_id` detected."
                )

    def _check_single_experiment_id(self) -> None:
        seen = set()
        for simulation_run in self.simulation_parameter_list:
            experiment_id = simulation_run.metadata.experiment_id
            if experiment_id not in seen:
                seen.add(experiment_id)

        if len(seen) > 1:
            raise ValueError(
                f"All MultipleSusi runs must have the same experiment_id. Found the following instead: {seen}."
            )

    @model_validator(mode="after")
    def validate_configuration(self) -> "MultipleSusis":
        """Validate the entire model after all fields are set."""
        self._check_not_more_processes_than_runs()
        self._check_for_duplicated_susi_params()
        self._check_for_duplicated_experiment_folder_paths()
        self._check_stand_and_scenario_ids_are_set()
        self._check_single_experiment_id()

        return self
