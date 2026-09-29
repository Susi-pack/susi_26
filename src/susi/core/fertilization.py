"""
Created on Sat Mar  5 19:31:27 2022

@author: alauren
"""

from dataclasses import dataclass
from typing import assert_never

import numpy as np

from susi.io.susi_parameter_model import (
    AshFertilizationParameters,
    FertilizationParameters,
    NutrientFertilizationParameters,
    StandardNPKFertilizationParameters,
)


@dataclass(frozen=True)
class FertilizationEffect:
    pH_increment: float  # level above the site's baseline pH, not a yearly change
    N_release: float  # kg/ha/yr
    P_release: float
    K_release: float


NO_EFFECT = FertilizationEffect(
    pH_increment=0.0, N_release=0.0, P_release=0.0, K_release=0.0
)


# Dispatch based on parameters
def fertilization_effect_in_year(
    params: FertilizationParameters | None, year: int
) -> FertilizationEffect:
    if params is None or year < params.application_year:
        return NO_EFFECT
    years_since_application = year - params.application_year
    match params:
        case StandardNPKFertilizationParameters():
            return first_order_decay_effect(params, years_since_application)
        case AshFertilizationParameters():
            return ash_effect(params, years_since_application)
        case _:
            assert_never(params)


def _first_order_release(
    nutrient_params: NutrientFertilizationParameters, years_since_application: int
) -> float:
    return (
        nutrient_params.dose
        * np.exp(-nutrient_params.decay_k * years_since_application)
        - nutrient_params.dose
        * np.exp(-nutrient_params.decay_k * (years_since_application + 1))
    ) * nutrient_params.eff


def first_order_decay_effect(
    params: StandardNPKFertilizationParameters, years_since_application: int
) -> FertilizationEffect:
    return FertilizationEffect(
        pH_increment=params.pH_increment
        * np.exp(-params.pH_decay_k * years_since_application),
        N_release=_first_order_release(
            nutrient_params=params.N,
            years_since_application=years_since_application,
        ),
        P_release=_first_order_release(
            nutrient_params=params.P,
            years_since_application=years_since_application,
        ),
        K_release=_first_order_release(
            nutrient_params=params.K,
            years_since_application=years_since_application,
        ),
    )


def ash_effect(
    params: AshFertilizationParameters, years_since_application: int
) -> FertilizationEffect:
    """
    Calculates the disintegration of ash grains and the release of P and K from ash and the effect on soil pH

    Note: recomputes until current year at every call.
    It could be precomputed too, but the runtime cost is milliseconds, and
    we chose the simplicity of the common interface.
    """

    dt = 1  # Yearly timestep. Must be =1 to be the same as SUSI.
    t = np.arange(0, years_since_application + 1, dt)

    # Result table initialization
    K_release = np.zeros(len(t))
    P_release = np.zeros(len(t))
    pH_history = np.zeros(len(t))

    # initial storages in ash (kg)
    K_storage = params.K_in_ash
    P_storage = params.P_in_ash

    # 1. Lasketaan montako raetta meillä on alussa
    volume_of_single_grain = (4 / 3) * np.pi * (params.grain_radius**3)
    mass_of_single_grain = params.density * volume_of_single_grain
    n_particles_0 = params.fertilizer_dose / mass_of_single_grain

    current_fertilizer_mass = params.fertilizer_dose

    # Start-of-year convention: the state at the start of each year drives that year.
    # K and P release use the grain surface area at the start of the year, and the
    # pH increment uses the ash dissolved by the start of the year
    # (current_fertilizer_mass is only updated at the end of the iteration).
    # So the pH increment is 0 in the application year.
    for i in range(len(t)):
        # 2. Murenemisfunktio: hiukkasten lukumäärä kasvaa ajan neliössä
        # n = n_0 * (1 + alpha * t^2)
        n_particles = n_particles_0 * (
            1 + params.particle_cracking_rate * (t[i] ** params.time_exp)
        )

        # 3. Lasketaan yksittäisen murentuneen hiukkasen massa tällä hetkellä
        m_particle = current_fertilizer_mass / n_particles

        # 4. Lasketaan hiukkasen säde ja kokonaispinta-ala
        # r = (3m / 4*pi*rho)^(1/3)
        r_eff = ((3 * m_particle) / (4 * np.pi * params.density)) ** (1 / 3)
        A_particle = 4 * np.pi * (r_eff**2)
        A_total = n_particles * A_particle

        # 5. Lasketaan vapautuva ravinne (dm = k * A * dt) tämä on massan laskenta
        dm = params.dissolution_rate * A_total * dt
        dm = min(dm, current_fertilizer_mass)  # Ei voi vapauttaa enemmän kuin on

        # 2. Kaliumin vapautuminen (nopeampi)
        dK = params.K_dissolution_rate * A_total * dt
        dK = min(dK, K_storage)
        K_storage -= dK

        # 3. Fosforin vapautuminen (hitaampi)
        dP = params.P_dissolution_rate * A_total * dt
        dP = min(dP, P_storage)
        P_storage -= dP

        K_release[i] = dK / dt
        P_release[i] = dP / dt

        # 6.lasketaan pHn nousu
        dissolved_ash = params.fertilizer_dose - current_fertilizer_mass
        # this is cumulative increment from the begining of the simulation
        pH_increment = dissolved_ash * params.pH_increment_per_dissolved_ash

        pH_history[i] = pH_increment

        # Päivitys seuraavalle kierrokselle
        current_fertilizer_mass -= dm

    return FertilizationEffect(
        pH_increment=pH_history[-1],
        N_release=0.0,
        P_release=P_release[-1],
        K_release=K_release[-1],
    )
