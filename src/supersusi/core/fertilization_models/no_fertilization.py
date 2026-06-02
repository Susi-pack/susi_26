# Applies no fertilization effects.

from dataclasses import dataclass
import numpy as np

from supersusi.core.fertilization_types import State, Inputs


@dataclass(frozen=True)
class Params:
    n_cols: int


def unfertilized_state(n_cols: int) -> State:
    return State(
        pH_increment=0.0,
        nutrient_release={nutr: np.zeros(n_cols) for nutr in ("N", "P", "K")},
    )


def run_timestep(params: Params, inputs: Inputs) -> State:
    return unfertilized_state(params.n_cols)
