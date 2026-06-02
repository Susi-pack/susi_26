from dataclasses import dataclass
import numpy as np

from susi.io.susi_parameter_model import StandardNPKFertilizationParameters
from supersusi.core.fertilization_types import State, Inputs, Nutrient
from supersusi.core.fertilization_models.no_fertilization import unfertilized_state


@dataclass(frozen=True)
class Params:
    n_cols: int
    fpara: StandardNPKFertilizationParameters


def run_timestep(params: Params, inputs: Inputs) -> State:
    if inputs.years_since_fertilization < 0:
        return unfertilized_state(n_cols=params.n_cols)
    return State(
        pH_increment=_compute_ph_effect(
            pH_increment_param=params.fpara.pH_increment,
            years_since_fertilization=inputs.years_since_fertilization,
        ),
        nutrient_release=_compute_nutrient_release(
            params=params,
            years_since_fertilization=inputs.years_since_fertilization,
        ),
    )


def _compute_ph_effect(pH_increment_param, years_since_fertilization: int) -> float:
    return pH_increment_param * np.exp(-0.1 * years_since_fertilization)


def _compute_nutrient_release(
    params: Params, years_since_fertilization: int
) -> dict[Nutrient, np.ndarray]:

    release = {}
    for nutr, fertilization_nutrient in [
        ("N", params.fpara.N),
        ("P", params.fpara.P),
        ("K", params.fpara.K),
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
            * np.ones(params.n_cols)
        )
    return release
