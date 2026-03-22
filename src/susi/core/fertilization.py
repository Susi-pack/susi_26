# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:31:27 2022

@author: alauren
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import assert_never, Literal, get_args
from dataclasses import dataclass

from susi.io.susi_parameter_model import (
    AshFertilizationParameters,
    StandardNPKFertilizationParameters,
    FertilizationParameters,
)

Nutrient = Literal["N", "P", "K"]


@dataclass(frozen=True)
class FertilizationEffect:
    """
    Return type
    """

    is_active: bool
    pH_increment: float
    nutrient_release: dict[Nutrient, np.ndarray]


class AbstractFertilization(ABC):
    """
    Abstract Base Class for fertilization models.
    It describes what the API for the models should be.
    The functions declared here are required to exist in the
    classes that inherit from it.

    Always instantiate one of its children; never this abstract class.
    """

    def __init__(self, n_cols: int, fpara: FertilizationParameters | None):
        self.ncols = n_cols
        # Nutrient release from the fertilizer.
        # Initialized as zero = no release.
        self.fpara = fpara

    @abstractmethod
    def compute_ph_effect(self, years_since_fertilization: int) -> float: ...

    @abstractmethod
    def compute_nutrient_release(
        self, years_since_fertilization: int
    ) -> dict[Nutrient, np.ndarray]: ...

    def _years_since_fertilization(self, year: int) -> int | None:
        if self.fpara is None:
            return None
        return year - self.fpara.application_year

    def _zero_release(self) -> dict[Nutrient, np.ndarray]:
        return {nutrient: np.zeros(self.ncols) for nutrient in get_args(Nutrient)}

    def is_active(self, years_since: int | None) -> bool:
        """
        Fertilization activates once the simulation year is at or above
        the specified fertilization year.
        """
        return years_since is not None and years_since >= 0

    def compute_effect(self, year: int) -> FertilizationEffect:
        years_since_fertilization = self._years_since_fertilization(year)

        if self.is_active(years_since=years_since_fertilization):
            return FertilizationEffect(
                is_active=True,
                pH_increment=self.compute_ph_effect(years_since_fertilization),
                nutrient_release=self.compute_nutrient_release(
                    years_since_fertilization
                ),
            )

        return FertilizationEffect(
            is_active=False,
            pH_increment=0.0,
            nutrient_release=self._zero_release(),
        )


class StandardNPKFertilization(AbstractFertilization):
    def compute_ph_effect(self, years_since_fertilization: int) -> float:
        return self.fpara.pH_increment * np.exp(-0.1 * years_since_fertilization)

    def compute_nutrient_release(
        self, years_since_fertilization: int
    ) -> dict[Nutrient, np.ndarray]:
        assert years_since_fertilization >= 0
        release = {}
        for nutr in ["N", "P", "K"]:
            fertilization_nutrient = getattr(self.fpara, nutr)

            nut_efficiency = fertilization_nutrient.eff
            dose = fertilization_nutrient.dose
            decay_k = fertilization_nutrient.decay_k
            release[nutr] = (
                (
                    dose * np.exp(-decay_k * years_since_fertilization)
                    - dose * np.exp(-decay_k * (years_since_fertilization + 1))
                )
                * nut_efficiency
                * np.ones(self.ncols)
            )
        return release


class AshFertilization(AbstractFertilization):
    def compute_ph_effect(self, years_since_fertilization: int) -> float:
        raise NotImplementedError()

    def compute_nutrient_release(
        self, years_since_fertilization: int
    ) -> dict[Nutrient, np.ndarray]:
        raise NotImplementedError()


class NoFertilization(AbstractFertilization):
    """
    Class returned when there is no fertilization.
    Its purpose is to be able to return something in case
    no fertilization happens.
    Otherwise, it is useless.
    """

    def compute_ph_effect(self, years_since_fertilization: int) -> float:
        return 0.0

    def compute_nutrient_release(
        self, years_since_fertilization: int
    ) -> dict[Nutrient, np.ndarray]:
        return self._zero_release()

    def compute_effect(self, year: int) -> FertilizationEffect:
        return FertilizationEffect(
            is_active=False,
            pH_increment=0.0,
            nutrient_release=self._zero_release(),
        )


def initialize_fertilization(
    fertilization_params: FertilizationParameters | None, n_cols: int
) -> AbstractFertilization:
    """
    Factory function to create the appropriate fertilization instance.
    """
    match fertilization_params:
        case None:
            return NoFertilization(n_cols=n_cols, fpara=None)
        case AshFertilizationParameters():
            return AshFertilization(n_cols=n_cols, fpara=fertilization_params)
        case StandardNPKFertilizationParameters():
            return StandardNPKFertilization(n_cols=n_cols, fpara=fertilization_params)
        case _:
            assert_never(fertilization_params)
