import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    SkipValidation,
    StrictBool,
    computed_field,
    field_validator,
)

from susi.io.extra_pydantic_types import (
    PositiveFloat,
    NonNegativeFloat,
    NonPositiveFloat,
)
from susi.io.utils import get_project_root


class StrictFrozenModel(BaseModel):
    """
    Defines a stricter Pydantic class
    """

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on assignment
        frozen=True,  # Force immutability
        extra="forbid",  # Forbid extra fields
        validate_default=True,  # Validate default values
        json_encoders={np.ndarray: lambda v: v.tolist()},
    )


class SimulationConfig(StrictFrozenModel):
    # Time
    start_date: datetime.datetime = Field(description="Simulation start date.")
    end_date: datetime.datetime = Field(description="Simulation end date.")

    # Forest
    # Age of different forest layers at the beginning of the simulation
    initial_dominant_stand_age_years: NonNegativeFloat = Field(
        description="Age of the dominant stand at the beginning of the simulation. This is set to all nodes in the strip."
    )
    initial_subdominant_stand_age_years: NonNegativeFloat = Field(
        description="Age of the subdominant layer at the beginning of the simulation. This is set to all nodes in the strip."
    )
    initial_understorey_age_years: NonNegativeFloat = Field(
        description="Age of the understorey at the beginning of the simulation. This is set to all nodes in the strip."
    )

    L: float = Field(description="Strip width, i.e., distance between ditches, m")

    @computed_field
    @property
    def n(self) -> int:
        """Number of computation nodes in the strip, 2-m width of node"""
        return int(self.L / 2)

    @property
    def age(self) -> SkipValidation[dict[str, np.ndarray]]:
        """Age of stand for all nodes along the strip"""
        return {
            "dominant": self.initial_dominant_stand_age_years * np.ones(self.n),
            "subdominant": self.initial_subdominant_stand_age_years * np.ones(self.n),
            "under": self.initial_understorey_age_years * np.ones(self.n),
        }


class WeatherParams(StrictFrozenModel):
    """
    Weather parameters
    """

    weather_filepath: FilePath = Field(description="Path to weather files.")


class MottiFileParams(StrictFrozenModel):
    """
    Motti files to read
    """

    path: DirectoryPath = Field(description="Motti files input file folder")
    dominant: dict[int, str] = Field(
        description="int: 0 if not in use. str: Motti file for the dominant layer."
    )
    subdominant: dict[int, str] = Field(
        description="int: 0 if not in use. str: Motti file for the subdominant layer."
    )
    under: dict[int, str] = Field(
        description="int: 0 if not in use. str: Motti file for the understorey layer."
    )


class SusiParams(StrictFrozenModel):
    """
    Parameter class to be stantiated.
    """

    params_schema_version: int = 1
    weather_parameters: WeatherParams
    motti_file_parameters: MottiFileParams
    simulation_config: SimulationConfig
