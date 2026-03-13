import datetime
from typing import Self
from pydantic import (
    Field,
    model_validator,
)

from susi.io.extra_pydantic_types import (
    StrictFrozenModel,
    NonNegativeFloat,
)


class NutrientFertilizationParameters(StrictFrozenModel):
    """
    Nutrient fertilization parameters
    """

    dose: NonNegativeFloat = Field(
        description="Dose of compound in fertilizer, kg ha-1"
    )
    decay_k: NonNegativeFloat = Field(description="Decay rate, yr-1")
    eff: NonNegativeFloat = Field(description="Nutrient use efficiency")


class FertilizationParameters(StrictFrozenModel):
    """
    Canopy parameters
    """

    application_year: int = 2201
    N: NutrientFertilizationParameters
    P: NutrientFertilizationParameters
    K: NutrientFertilizationParameters
    pH_increment: NonNegativeFloat = 1.0


class SusiParams(StrictFrozenModel):
    """
    Parameter class to be stantiated.
    """

    start_date: datetime.datetime = Field(description="Simulation start date.")
    end_date: datetime.datetime = Field(description="Simulation end date.")
    fertilization: FertilizationParameters | None = None

    @model_validator(mode="after")
    def check_fertilization_within_bounds(self) -> Self:
        # Now 'self' is SusiParams, which CAN see both children

        if self.fertilization is not None:
            app_year = self.fertilization.application_year
            if not (self.start_date.year <= app_year <= self.end_date.year):
                raise ValueError(f"Fertilization year {app_year} is out of bounds!")
        return self


SusiParams(
    start_date=datetime.datetime(year=2020, month=1, day=1),
    end_date=datetime.datetime(year=2025, month=1, day=1),
)
