from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float


@dataclass(frozen=True)
class Params:
    org_depth: Float[np.ndarray, " n"] = field(doc="Depth of organic top layer [m]")
    org_poros: Float[np.ndarray, " n"] = field(doc="Porosity of organic layer [-]")
    org_fc: Float[np.ndarray, " n"] = field(doc="Field capacity of organic layer [-]")
    org_rw: Float[np.ndarray, " n"] = field(
        doc="Critical volumetric moisture content for decreasing phase in Ef [-]"
    )
    pond_storage_max: Float[np.ndarray, " n"] = field(
        doc="Maximum pond storage depth [m]"
    )
    org_sat: Float[np.ndarray, " n"] = field(
        doc="Initial organic layer saturation ratio [-]"
    )
    pond_storage_initial: Float[np.ndarray, " n"] = field(
        doc="Initial pond storage depth [m]"
    )


@dataclass(frozen=True)
class ComputedConstants:
    Wsto_top_max: Float[np.ndarray, " n"] = field(
        doc="Maximum water storage in organic top layer [m]"
    )
    h_pond_max: Float[np.ndarray, " n"] = field(doc="Maximum pond storage [m]")


@dataclass(frozen=True)
class State:
    Wsto_top: Float[np.ndarray, " n"] = field(
        doc="Water storage in organic top layer [m]"
    )
    h_pond: Float[np.ndarray, " n"] = field(doc="Pond storage depth [m]")
    Ree: Float[np.ndarray, " n"] = field(
        doc="Relative evaporation rate for next timestep [-]"
    )


@dataclass(frozen=True)
class InterceptionInputs:
    potinf: Float[np.ndarray, " n"] = field(doc="Throughfall reaching moss layer [m]")
    evap: Float[np.ndarray, " n"] = field(
        doc="Potential evaporation from moss layer [m]"
    )


@dataclass(frozen=True)
class InterceptionOutputs:
    potinf: Float[np.ndarray, " n"] = field(
        doc="Water passing through to soil after interception [m]"
    )
    evap: Float[np.ndarray, " n"] = field(doc="Actual evaporation from moss layer [m]")
    mbe: Float[np.ndarray, " n"] = field(doc="Mass balance error from interception [m]")


@dataclass(frozen=True)
class ReturnflowInputs:
    rflow: Float[np.ndarray, " n"] = field(
        doc="Return flow from soil that cannot be stored [m]"
    )
    interception_mbe: Float[np.ndarray, " n"] = field(
        doc="Mass balance error from the interception step, to be accumulated [m]"
    )


@dataclass(frozen=True)
class ReturnflowOutputs:
    surface_runoff: Float[np.ndarray, " n"] = field(doc="Surface runoff generated [m]")
    mbe: Float[np.ndarray, " n"] = field(doc="Mass balance error from return flow [m]")


def compute_constants(params: Params) -> ComputedConstants:
    return ComputedConstants(
        Wsto_top_max=params.org_fc * params.org_depth,
        h_pond_max=params.pond_storage_max,
    )


def compute_initial_state(
    params: Params, computed_constants: ComputedConstants
) -> State:
    Wsto_top = computed_constants.Wsto_top_max * params.org_sat
    Wliq_top = params.org_poros * Wsto_top / computed_constants.Wsto_top_max
    Ree = np.maximum(0.0, np.minimum(0.98 * Wliq_top / params.org_rw, 1.0))
    return State(
        Wsto_top=Wsto_top,
        h_pond=params.pond_storage_initial,
        Ree=Ree,
    )


def assemble_interception_inputs(
    potinf: Float[np.ndarray, " n"],
    evap: Float[np.ndarray, " n"],
) -> InterceptionInputs:
    return InterceptionInputs(potinf=potinf, evap=evap)


def assemble_returnflow_inputs(
    rflow: Float[np.ndarray, " n"],
    interception_mbe: Float[np.ndarray, " n"],
) -> ReturnflowInputs:
    return ReturnflowInputs(rflow=rflow, interception_mbe=interception_mbe)


def run_interception(
    computed_constants: ComputedConstants,
    previous_state: State,
    input: InterceptionInputs,
) -> tuple[State, InterceptionOutputs]:
    Wsto_top_max = computed_constants.Wsto_top_max

    potinf0 = input.potinf.copy()
    Wsto_top_ini = previous_state.Wsto_top.copy()
    pond_ini = previous_state.h_pond.copy()

    potinf = input.potinf + pond_ini
    h_pond = previous_state.h_pond - pond_ini

    interc = np.maximum(0.0, Wsto_top_max - previous_state.Wsto_top) * (
        1.0 - np.exp(-(potinf / Wsto_top_max))
    )
    potinf -= interc
    Wsto_top = previous_state.Wsto_top + interc
    evap = np.minimum(input.evap, Wsto_top)
    Wsto_top -= evap

    mbe = (pond_ini - h_pond) + (Wsto_top_ini - Wsto_top) + potinf0 - potinf - evap

    return State(Wsto_top=Wsto_top, h_pond=h_pond, Ree=previous_state.Ree), InterceptionOutputs(
        potinf=potinf,
        evap=evap,
        mbe=mbe,
    )


def run_returnflow(
    params: Params,
    computed_constants: ComputedConstants,
    previous_state: State,
    input: ReturnflowInputs,
) -> tuple[State, ReturnflowOutputs]:
    Wsto_top_max = computed_constants.Wsto_top_max
    h_pond_max = computed_constants.h_pond_max

    to_top_layer = np.minimum(input.rflow, Wsto_top_max - previous_state.Wsto_top)
    Wsto_top = previous_state.Wsto_top + to_top_layer
    to_pond = np.minimum(input.rflow - to_top_layer, h_pond_max - previous_state.h_pond)
    h_pond = previous_state.h_pond + to_pond
    surface_runoff = input.rflow - to_top_layer - to_pond

    mbe = input.interception_mbe + input.rflow - to_top_layer - to_pond - surface_runoff

    Wliq_top = params.org_fc * Wsto_top / Wsto_top_max
    Ree = np.maximum(0.0, np.minimum(0.98 * Wliq_top / params.org_rw, 1.0))

    return State(Wsto_top=Wsto_top, h_pond=h_pond, Ree=Ree), ReturnflowOutputs(
        surface_runoff=surface_runoff,
        mbe=mbe,
    )
