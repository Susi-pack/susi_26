from typing import Optional
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator
import json


from susi.io.extra_pydantic_types import PositiveInt, StrictFrozenModel
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

    n_runs: PositiveInt = Field(frozen=True, description="Number of total SUSI runs.")
    n_parallel_processes: PositiveInt = Field(
        frozen=True,
        default=1,
        description="Number of parallel processes to spawn. Cannot be greater than n_runs.",
    )
    simulation_parameter_list: list[SimulationParams] = Field(
        frozen=True,
        description="A list of parameters fully specifying each run. It must have 'n_runs' number of elements. E.g., if only runing 1 simulation, 'n_runs'=1, and the length of the 'runs' list must be 1.",
    )

    @model_validator(mode="after")
    def check_not_more_processes_than_runs(self) -> Self:
        if self.n_parallel_processes > self.n_runs:
            raise ValueError(
                "'n_parallel_processes' cannot be greater than 'n_runs'. There must be at most one process per run."
            )
        return self

    def _check_for_duplicated_params(self) -> None:
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

    def _check_number_of_runs(self) -> None:
        if self.n_runs != len(self.simulation_parameter_list):
            raise ValueError(
                "The number of runs in the list 'runs' must be exactly 'n_runs'."
            )

    @model_validator(mode="after")
    def validate_configuration(self) -> "MultipleSusis":
        """Validate the entire model after all fields are set."""
        self._check_number_of_runs()
        self._check_for_duplicated_params()
        return self
