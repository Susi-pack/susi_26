import datetime
from typing import Self, Literal, Annotated, Union, assert_never
from pydantic import (
    Field,
    model_validator,
    computed_field,
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


class BaseFertilization(StrictFrozenModel):
    application_year: int = 2201
    N: NutrientFertilizationParameters
    P: NutrientFertilizationParameters
    K: NutrientFertilizationParameters


class StandardFertilization(BaseFertilization):
    type: Literal["standard"] = "standard"
    pH_increment: NonNegativeFloat = 1.0


class AshFertilization(BaseFertilization):
    type: Literal["ash"] = "ash"
    dose: NonNegativeFloat

    @computed_field
    @property
    def pH_increment(self) -> float:
        return self.dose**2


Fertilization = Annotated[
    Union[AshFertilization, StandardFertilization], Field(discriminator="type")
]


class SusiParams(StrictFrozenModel):
    """
    Parameter class to be stantiated.
    """

    start_date: datetime.datetime = Field(description="Simulation start date.")
    end_date: datetime.datetime = Field(description="Simulation end date.")
    fertilization: Fertilization | None = None

    @model_validator(mode="after")
    def check_fertilization_within_bounds(self) -> Self:
        # Now 'self' is SusiParams, which CAN see both children

        if self.fertilization is not None:
            app_year = self.fertilization.application_year
            if not (self.start_date.year <= app_year <= self.end_date.year):
                raise ValueError(f"Fertilization year {app_year} is out of bounds!")
        return self


# %% test

date20 = datetime.datetime(year=2020, month=1, day=1)
date25 = datetime.datetime(year=2025, month=1, day=1)

nutr = NutrientFertilizationParameters(dose=0.3, decay_k=1.0, eff=2.4)
ash_fert = AshFertilization(application_year=2021, dose=0.2, N=nutr, P=nutr, K=nutr)
std_fert = StandardFertilization(application_year=2021, N=nutr, P=nutr, K=nutr)

susi_nofert = SusiParams(start_date=date20, end_date=date25)
susi_ashfert = SusiParams(start_date=date20, end_date=date25, fertilization=ash_fert)
susi_stdfert = SusiParams(start_date=date20, end_date=date25, fertilization=std_fert)

match susi_stdfert.fertilization:
    case AshFertilization():
        print("ash")
    case StandardFertilization():
        print("std")
    case None:
        print("None")
    case _ as unreachable:
        assert_never(unreachable)
