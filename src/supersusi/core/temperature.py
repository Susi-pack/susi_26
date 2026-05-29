from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as linalg

from jaxtyping import Float


@dataclass(frozen=True)
class Params:
    n_layers_hydro: int
    dz: float
    timestep: float
    n_subtimesteps: int
    D: float
    heat_of_vaporization: float


@dataclass(frozen=True)
class State:
    T_soil: np.ndarray = field(
        doc="Peat temperature at different depths [deg C]. Dimensions: (nscenarios, ndays, nLyrs)"
    )


@dataclass(frozen=True)
class DynamicInputs:
    T_air: float = field(doc="Air temperature [deg C]")
    swe: float = field(doc="Snow water equivalent [m]")
    efloor: float = field(doc="Evaporation from surface layer [m]")


@dataclass(frozen=True)
class StaticInputs:
    z: Float[np.ndarray, " n_layers"] = field(
        doc="depth of the layers' central point [m]"
    )
    n_layers: int
    A: Float[np.ndarray, "n_layers+1 n_layers+1"] = field(
        doc="Matrix of the linear system resulting from ODE discretization. It is tridiagonal."
    )
    heat_capacity: float = field(doc="[J m-3]")
    T_air_mean: float = field(
        doc="Air temperature, coming from weather data [deg C]. Used as lower boundary condition for peat temperature."
    )


def compute_static_inputs(params: Params, T_air_mean: float) -> StaticInputs:
    n_layers = params.n_layers_hydro + 30

    return StaticInputs(
        z=np.cumsum(np.ones(n_layers) * params.dz) - params.dz / 2.0,
        heat_capacity=3860000.0 * params.dz,
        n_layers=n_layers,
        A=_create_linear_system_matrix(
            timestep=params.timestep,
            n_subtimesteps=params.n_subtimesteps,
            dz=params.dz,
            D=params.D,
            n_layers=n_layers,
        ),
        T_air_mean=T_air_mean,
    )


def compute_initial_state(static_inputs: StaticInputs) -> State:
    return State(T_soil=np.ones(static_inputs.n_layers + 1) * static_inputs.T_air_mean)


def step(
    params: Params,
    static_inputs: StaticInputs,
    dynamic_inputs=DynamicInputs,
    state=State,
) -> State:

    # Cooling by evaporation
    e_consumed = (
        dynamic_inputs.efloor
        * 1000
        * params.heat_of_vaporization
        / params.n_subtimesteps
    )
    T_cool = -e_consumed / static_inputs.heat_capacity
    if dynamic_inputs.swe > 0.01:
        T_air = max(-5.0, dynamic_inputs.T_air)
    else:
        T_air = dynamic_inputs.T_air + T_cool

    u = np.zeros(static_inputs.n_layers + 1)
    T_soil = state.T_soil.copy()
    for _ in range(0, params.n_subtimesteps):
        b = T_soil.copy()
        b[0] = T_air  # top boundary condition
        b[-1] = static_inputs.T_air_mean  # bottom boundary condition
        u[:] = linalg.solve(static_inputs.A, b, assume_a="tridiagonal")
        T_soil = u
    return State(T_soil=u)


def _create_linear_system_matrix(
    timestep: float, n_subtimesteps: int, dz: float, D: float, n_layers: int
):
    t = np.linspace(0, timestep, n_subtimesteps + 1)  # mesh points in time
    dt = t[1] - t[0]  # timestep in seconds, s

    F = D * dt / dz**2

    # Create the main matrix
    A = (
        np.diag([1 + 2 * F] * (n_layers + 1))
        + np.diag([-F] * n_layers, k=1)
        + np.diag([-F] * n_layers, k=-1)
    )

    # Insert boundary conditions, Diritchlet (constant value) boundaries
    A[0, 0] = 1
    A[n_layers, n_layers] = 1

    return A
