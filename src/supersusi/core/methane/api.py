from dataclasses import dataclass, field
import pandas as pd
import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class MethaneState:
    ch4: Float[Array, " n_cols"] = field(doc="Annual node-wise kg CH4 ha-1 year-1")
    ch4_as_co2eq: Float[Array, " n_cols"] = field(
        doc="Annual CH4 kg  ha-1 year-1 in CO2-eq"
    )


@dataclass(frozen=True)
class MethaneDynamicInputs:
    year: int = field(doc="Simulation year")
    dfwt: pd.DataFrame = field(doc="Daily WT dataframe")


@dataclass(frozen=True)
class MethaneStaticInputs:
    None


@dataclass(frozen=True)
class MethaneParams:
    None


def initialize(n_cols: int) -> MethaneState:
    return MethaneState(
        ch4=jnp.zeros(shape=n_cols), ch4_as_co2eq=jnp.zeros(shape=n_cols)
    )


def step(dynamic_inputs: MethaneDynamicInputs) -> MethaneState:

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

    return MethaneState(ch4=ch4, ch4_as_co2eq=ch4 * 27.0)
