from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float


@dataclass(frozen=True)
class Params:
    dt: float = field(doc="Canopy model timestep [s]")

    # Per-cell constants
    cf: Float[np.ndarray, " n"] = field(doc="Canopy closure fraction [-]")
    lai_decid_max: Float[np.ndarray, " n"] = field(
        doc="Maximum annual deciduous 1-sided leaf area index [m2 m-2]"
    )

    # Interception
    wmax: float = field(doc="Interception capacity per LAI for rain [mm]")
    wmaxsnow: float = field(doc="Interception capacity per LAI for snow [mm]")

    # Snow
    kmelt: float = field(doc="Melt coefficient (in open) [mm s-1 K-1]")
    kfreeze: float = field(doc="Freezing coefficient [mm s-1 K-1]")
    r: float = field(doc="Maximum fraction of liquid water in snowpack [-]")

    # Physiology
    amax_init: float = field(
        doc="Initial maximum photosynthetic rate [umol m-2(leaf) s-1]"
    )
    g1_conif: float = field(doc="Stomatal conductance parameter, conifers")
    g1_decid: float = field(doc="Stomatal conductance parameter, deciduous")
    kp: float = field(doc="PAR attenuation coefficient [-]")
    q50: float = field(doc="Half-saturation of leaf light response [W m-2]")
    gsoil: float = field(doc="Soil surface conductance when fully wet [m s-1]")

    # Flow / aerodynamics
    zmeas: float = field(doc="Wind measurement height above canopy [m]")
    zground: float = field(doc="Reference height above ground [m]")
    zo_ground: float = field(doc="Ground roughness length [m]")

    # Phenology — temperature acclimation
    smax: float = field(doc="Phenology modifier sensitivity [degC]")
    tau: float = field(doc="Temperature acclimation time constant [d]")
    xo: float = field(doc="Temperature threshold for acclimation [degC]")
    fmin: float = field(doc="Minimum phenology modifier (residual photosynthetic capacity) [-]")

    # Phenology — deciduous leaf dynamics
    lai_decid_min: float = field(doc="Minimum deciduous LAI fraction [-]")
    ddo: float = field(doc="Degree-day onset for leaf growth [degC]")
    ddur: float = field(doc="Duration of leaf growth phase [d]")
    sso: float = field(doc="Senescence onset day of year")
    sdur: float = field(doc="Senescence duration [d]")


@dataclass(frozen=True)
class ComputedConstants:
    pass


@dataclass(frozen=True)
class State:
    W: Float[np.ndarray, " n"] = field(doc="Canopy water storage [mm]")
    SWE: Float[np.ndarray, " n"] = field(doc="Snow water equivalent [mm]")
    SWEi: Float[np.ndarray, " n"] = field(doc="Ice in snowpack [mm]")
    SWEl: Float[np.ndarray, " n"] = field(doc="Liquid water in snowpack [mm]")
    X: Float[np.ndarray, " n"] = field(doc="Temperature acclimation state [degC]")
    amax: Float[np.ndarray, " n"] = field(
        doc="Photosynthetic capacity [umol m-2(leaf) s-1]"
    )
    DDsum: Float[np.ndarray, " n"] = field(doc="Degree-day sum [degC]")
    growth_stage: Float[np.ndarray, " n"] = field(
        doc="Phenology growth stage counter [-]"
    )
    senesc_stage: Float[np.ndarray, " n"] = field(
        doc="Phenology senescence stage counter [-]"
    )
    LAIdecid: Float[np.ndarray, " n"] = field(doc="Deciduous leaf area index [m2 m-2]")


@dataclass(frozen=True)
class Outputs:
    potinf: Float[np.ndarray, " n"] = field(doc="Potential infiltration to soil [m]")
    trfall: Float[np.ndarray, " n"] = field(doc="Throughfall [m]")
    interc: Float[np.ndarray, " n"] = field(doc="Interception [m]")
    evap: Float[np.ndarray, " n"] = field(doc="Canopy evaporation [m]")
    et: Float[np.ndarray, " n"] = field(doc="Total evapotranspiration [m]")
    transpi: Float[np.ndarray, " n"] = field(doc="Transpiration [m]")
    efloor: Float[np.ndarray, " n"] = field(doc="Forest floor evaporation [m]")
    mbe: Float[np.ndarray, " n"] = field(doc="Mass balance error [m]")
    swe: Float[np.ndarray, " n"] = field(doc="Snow water equivalent [mm]")


@dataclass(frozen=True)
class Inputs:
    doy: int = field(doc="Day of year")
    Ta: Float[np.ndarray, " n"] = field(doc="Air temperature [degC]")
    Prec: Float[np.ndarray, " n"] = field(doc="Precipitation rate [mm s-1]")
    Rg: Float[np.ndarray, " n"] = field(doc="Global radiation [W m-2]")
    Par: Float[np.ndarray, " n"] = field(
        doc="Photosynthetically active radiation [W m-2]"
    )
    VPD: Float[np.ndarray, " n"] = field(doc="Vapor pressure deficit [kPa]")
    hc: Float[np.ndarray, " n"] = field(doc="Canopy height [m]")
    LAIconif: Float[np.ndarray, " n"] = field(doc="Conifer leaf area index [m2 m-2]")
    U: Float[np.ndarray, " n"] = field(doc="Wind speed [m s-1]")
    CO2: Float[np.ndarray, " n"] = field(doc="CO2 concentration [ppm]")
    Rew: Float[np.ndarray, " n"] = field(doc="Relative extractable water [-]")
    beta: Float[np.ndarray, " n"] = field(doc="Soil evaporation resistance factor [-]")
    P: Float[np.ndarray, " n"] = field(doc="Air pressure [Pa]")


# def compute_initial_state(params: Params, computed_constants: ComputedConstants) -> State:
#     ...

# def assemble_inputs(...) -> Inputs:
#     ...

# def run_timestep(
#     params: Params,
#     computed_constants: ComputedConstants,
#     input: Inputs,
#     state: State,
# ) -> tuple[State, Outputs]:
#     ...
