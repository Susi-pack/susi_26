# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:31:27 2022

@author: alauren
"""

import numpy as np
from abc import ABC
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

    def __init__(self, n_cols: int, fpara: FertilizationParameters | None = None):
        self.ncols = n_cols
        self.fpara = fpara

    # ---- Core API ----
    def compute_effect(self, year: int) -> FertilizationEffect:
        """
        Template method: handles the inactive guard,
        delegates active logic to subclasses.
        """
        years_since = self._years_since_fertilization(year)

        if self.is_active(years_since):
            return self._compute_active_effect(years_since)

        # else, if not active:
        return FertilizationEffect(
            is_active=False,
            pH_increment=0.0,
            nutrient_release=self._zero_release(),
        )

    # ---- Subclasses must implement this ----
    def _compute_active_effect(
        self, years_since_fertilization: int
    ) -> FertilizationEffect: ...

    # ---- Shared helpers ----
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
        if years_since is None:
            return False
        return years_since >= 0


class StandardNPKFertilization(AbstractFertilization):
    def __init__(self, n_cols: int, fpara: StandardNPKFertilizationParameters):
        super().__init__(n_cols=n_cols, fpara=fpara)

    def compute_ph_effect(self, years_since_fertilization: int) -> float:
        return self.fpara.pH_increment * np.exp(-0.1 * years_since_fertilization)

    def compute_nutrient_release(
        self, years_since_fertilization: int
    ) -> dict[Nutrient, np.ndarray]:

        release = {}
        for nutr, fertilization_nutrient in [
            ("N", self.fpara.N),
            ("P", self.fpara.P),
            ("K", self.fpara.K),
        ]:
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

    def _compute_active_effect(
        self, years_since_fertilization: int
    ) -> FertilizationEffect:

        return FertilizationEffect(
            is_active=True,
            pH_increment=self.compute_ph_effect(years_since_fertilization),
            nutrient_release=self.compute_nutrient_release(years_since_fertilization),
        )


class AshFertilization(AbstractFertilization):
    def __init__(
        self,
        n_cols: int,
        fpara: AshFertilizationParameters,
        n_years_to_simulate_since_fertilization: int,
    ):
        super().__init__(n_cols=n_cols, fpara=fpara)

        # Ash fertilization curves are precomputed,
        # and accessed later for the yearly values
        self.pH_history, self.K_release_history, self.P_release_history = (
            self._precompute_ash_fertilization(
                n_years_to_simulate_since_fertilization, fpara
            )
        )

    def _precompute_ash_fertilization(
        self,
        n_years_to_simulate_since_fertilization: int,
        fpara: AshFertilizationParameters,
    ):
        """
        Calculates the disintegration of ash grains and the release of P and K from ash and the ewffect on soil pH
        Inputs:
            n_years_to_simulate_since_fertilization: number of years since fertilization until simulation ends
            fpara: AshFertilizationParameters
        """

        dt = 1  # Yearly timestep. Must be =1 to be the same as SUSI.
        t = np.arange(0, n_years_to_simulate_since_fertilization, dt)

        # Result table initialization
        K_release = np.zeros(len(t))
        P_release = np.zeros(len(t))
        pH_history = np.zeros(len(t))

        # initial storages in ash (kg)
        K_storage = fpara.K_in_ash
        P_storage = fpara.P_in_ash

        # 1. Lasketaan montako raetta meillä on alussa
        volume_of_single_grain = (4 / 3) * np.pi * (fpara.grain_radius**3)
        mass_of_single_grain = fpara.density * volume_of_single_grain
        n_particles_0 = fpara.fertilizer_dose / mass_of_single_grain

        current_fertilizer_mass = fpara.fertilizer_dose

        for i in range(len(t)):
            # 2. Murenemisfunktio: hiukkasten lukumäärä kasvaa ajan neliössä
            # n = n_0 * (1 + alpha * t^2)
            n_particles = n_particles_0 * (
                1 + fpara.particle_cracking_rate * (t[i] ** fpara.time_exp)
            )

            # 3. Lasketaan yksittäisen murentuneen hiukkasen massa tällä hetkellä
            m_particle = current_fertilizer_mass / n_particles

            # 4. Lasketaan hiukkasen säde ja kokonaispinta-ala
            # r = (3m / 4*pi*rho)^(1/3)
            r_eff = ((3 * m_particle) / (4 * np.pi * fpara.density)) ** (1 / 3)
            A_particle = 4 * np.pi * (r_eff**2)
            A_total = n_particles * A_particle

            # 5. Lasketaan vapautuva ravinne (dm = k * A * dt) tämä on massan laskenta
            dm = fpara.dissolution_rate * A_total * dt
            dm = min(dm, current_fertilizer_mass)  # Ei voi vapauttaa enemmän kuin on

            # 2. Kaliumin vapautuminen (nopeampi)
            dK = fpara.K_dissolution_rate * A_total * dt
            dK = min(dK, K_storage)
            K_storage -= dK

            # 3. Fosforin vapautuminen (hitaampi)
            dP = fpara.P_dissolution_rate * A_total * dt
            dP = min(dP, P_storage)
            P_storage -= dP

            K_release[i] = dK / dt
            P_release[i] = dP / dt

            # 6.lasketaan pHn nousu
            dissolved_ash = fpara.fertilizer_dose - current_fertilizer_mass
            # this is cumulative increment from the begiining of the simulation
            # TODO: set the value 2.5/15000 as a constant in fpara
            pH_increment = dissolved_ash * (2.5 / 15000)

            pH_history[i] = pH_increment

            # Päivitys seuraavalle kierrokselle
            current_fertilizer_mass -= dm

        return pH_history, K_release, P_release

    def _compute_active_effect(
        self, years_since_fertilization: int
    ) -> FertilizationEffect:
        return FertilizationEffect(
            is_active=True,
            pH_increment=self.ph_history[years_since_fertilization],
            nutrient_release={
                "N": np.zeros(self.ncols),
                "P": self.P_release_history[years_since_fertilization]
                * np.ones(self.ncols),
                "K": self.K_release_history[years_since_fertilization]
                * np.ones(self.ncols),
            },
        )


class NoFertilization(AbstractFertilization):
    """
    Class returned when there is no fertilization.
    Its purpose is to be able to return something in case
    no fertilization happens.
    Otherwise, it is useless.
    """

    def __init__(self, n_cols: int):
        super().__init__(n_cols=n_cols, fpara=None)

    def is_active(self, years_since: int) -> bool:
        return False

    def _compute_active_effect(
        self, years_since_fertilization: int
    ) -> FertilizationEffect:
        # No need to implement this for this class.
        raise NotImplementedError


def initialize_fertilization(
    fertilization_params: FertilizationParameters | None,
    n_cols: int,
    simulation_end_year: int,
) -> AbstractFertilization:
    """
    Factory function to create the appropriate fertilization instance.
    """
    match fertilization_params:
        case None:
            return NoFertilization(n_cols=n_cols)
        case AshFertilizationParameters():
            return AshFertilization(
                n_cols=n_cols,
                fpara=fertilization_params,
                n_years_to_simulate_since_fertilization=simulation_end_year
                - fertilization_params.application_year,
            )
        case StandardNPKFertilizationParameters():
            return StandardNPKFertilization(n_cols=n_cols, fpara=fertilization_params)
        case _:
            assert_never(fertilization_params)
