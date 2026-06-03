# Applies no fertilization effects.

from dataclasses import dataclass
import numpy as np

from supersusi.core.fertilization_types import State, Inputs, Outputs


@dataclass(frozen=True)
class Params:
    n_cols: int


def unfertilized_output(n_cols: int) -> Outputs:
    return Outputs(
        pH_increment=0.0,
        nutrient_release={nutr: np.zeros(n_cols) for nutr in ("N", "P", "K")},
    )


def run_timestep(params: Params, inputs: Inputs) -> tuple[State, Outputs]:
    return State(), unfertilized_output(params.n_cols)
