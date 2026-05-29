from dataclasses import dataclass, field
import pandas as pd
import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class State:
    ch4: Float[Array, " n_cols"] = field(doc="Annual node-wise kg CH4 ha-1 year-1")
    ch4_as_co2eq: Float[Array, " n_cols"] = field(
        doc="Annual CH4 kg  ha-1 year-1 in CO2-eq"
    )


@dataclass(frozen=True)
class DynamicInputs:
    year: int = field(doc="Simulation year")
    dfwt: pd.DataFrame = field(doc="Daily WT dataframe")


@dataclass(frozen=True)
class StaticInputs:
    None


@dataclass(frozen=True)
class Params:
    None


def initialize(n_cols: int) -> State:
    return State(ch4=jnp.zeros(shape=n_cols), ch4_as_co2eq=jnp.zeros(shape=n_cols))


def run_timestep(dynamic_inputs: DynamicInputs) -> State:

    # convert to cm positive down
    wt = (
        dynamic_inputs.dfwt[
            str(dynamic_inputs.year) + "-05-01" : str(dynamic_inputs.year) + "-10-31"
        ]
        .mean()
        .values
        * -100.0
    )

    # Ojanen et al. 2010, Fig. 6, convert to kg CH4 /ha/year
    ch4 = (-0.378 + 12.3 * jnp.exp(-0.121 * wt)) * 10.0

    return State(ch4=ch4, ch4_as_co2eq=_methane_to_co2eq(ch4))


def _methane_to_co2eq(ch4):
    # convert to kg CH4 to kg CO2-eq.
    # Reference: SGWP100 coefficient https://ghgprotocol.org/sites/default/files/2024-08/Global-Warming-Potential-Values%20%28August%202024%29.pdf
    return ch4 * 27.0
