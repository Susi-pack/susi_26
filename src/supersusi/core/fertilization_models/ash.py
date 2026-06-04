from dataclasses import dataclass
import numpy as np

from supersusi.core.fertilization_types import State, Inputs, Outputs
from supersusi.core.fertilization_models.no_fertilization import unfertilized_output
from supersusi.io.susi_parameter_model import AshFertilizationParameters


@dataclass(frozen=True)
class Params:
    n_cols: int
    simulation_end_year: int
    fpara: AshFertilizationParameters


@dataclass(frozen=True)
class ComputedConstants:
    pH_history: np.ndarray
    K_release_history: np.ndarray
    P_release_history: np.ndarray


def compute_constants(params: Params) -> ComputedConstants:
    """
    Calculates the disintegration of ash grains and the release of P and K from ash and the ewffect on soil pH
    """
    fpara = params.fpara
    # Number of years since fertilization until simulation ends
    n_years_to_simulate_since_fertilization = (
        params.simulation_end_year - fpara.application_year
    )

    dt = 1  # Yearly timestep. Must be =1 to be the same as SUSI.
    t = np.arange(0, n_years_to_simulate_since_fertilization, dt)

    # Result table initialization.
    K_release_history = np.zeros(len(t))
    P_release_history = np.zeros(len(t))
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

        K_release_history[i] = dK / dt
        P_release_history[i] = dP / dt

        # 6.lasketaan pHn nousu
        dissolved_ash = fpara.fertilizer_dose - current_fertilizer_mass
        # this is cumulative increment from the begiining of the simulation
        # TODO: set the value 2.5/15000 as a constant in fpara
        pH_increment = dissolved_ash * (2.5 / 15000)

        pH_history[i] = pH_increment

        # Päivitys seuraavalle kierrokselle
        current_fertilizer_mass -= dm

    return ComputedConstants(
        pH_history=pH_history,
        K_release_history=K_release_history,
        P_release_history=P_release_history,
    )


def run_timestep(
    params: Params, computed_constants: ComputedConstants, inputs: Inputs
) -> tuple[State, Outputs]:
    if inputs.years_since_fertilization < 0:
        return State(), unfertilized_output(n_cols=params.n_cols)
    return State(), Outputs(
        pH_increment=computed_constants.pH_history[inputs.years_since_fertilization],
        nutrient_release={
            "N": np.zeros(params.n_cols),
            "P": computed_constants.P_release_history[inputs.years_since_fertilization]
            * np.ones(params.n_cols),
            "K": computed_constants.K_release_history[inputs.years_since_fertilization]
            * np.ones(params.n_cols),
        },
    )
