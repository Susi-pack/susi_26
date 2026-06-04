from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float

from supersusi.core.susi_utils import peat_hydrol_properties, CWTr
from supersusi.core.mosslayer import ReturnflowOutputs


@dataclass(frozen=True)
class Params:
    nLyrs: int = field(doc="Number of soil layers")
    dzLyr: float = field(doc="Vertical layer thickness [m]")
    vonP: bool = field(doc="Use von Post scale flag")
    vonP_top: Float[np.ndarray, " n_top"] = field(
        doc="von Post humification at top layers"
    )
    vonP_bottom: int = field(doc="von Post humification at bottom layers")
    peat_type: list[str] = field(doc="Peat type per layer (e.g., 'A', 'S')")
    peat_type_bottom: list[str] = field(doc="Peat type for bottom layers (fallback)")
    bd_top: Float[np.ndarray, " n_top"] | None = field(
        default=None, doc="Bulk density at top [g cm-3]"
    )
    bd_bottom: float = field(default=0.0, doc="Bulk density at bottom [g cm-3]")
    anisotropy: float = field(
        default=1.0, doc="Hydraulic conductivity anisotropy factor"
    )
    L: float = field(default=10.0, doc="Strip width / ditch distance [m]")
    n: int = field(default=10, doc="Number of computation nodes")
    slope: float = field(default=0.0, doc="Slope [%]")
    initial_h: float = field(default=0.0, doc="Initial water table depth [m]")
    dt: float = field(default=1.0, doc="Time step [days]")
    implic: float = field(default=1.0, doc="Implicit factor (0=FE, 1=BE, 0.5=CN)")
    DrIrr: bool = field(default=False, doc="Drainage/irrigation flag")


@dataclass(frozen=True)
class ComputedConstants:
    dz: Float[np.ndarray, " nLyrs"] = field(doc="Layer thickness [m]")
    z: Float[np.ndarray, " nLyrs"] = field(doc="Depth of layer center points [m]")
    pF: Float[np.ndarray, " nLyrs 4"] = field(
        doc="van Genuchten parameters [ThetaS, ThetaR, alpha, n]"
    )
    Ksat: Float[np.ndarray, " nLyrs"] = field(
        doc="Saturated hydraulic conductivity [m s-1]"
    )
    Kmap: Float[np.ndarray, " n nLyrs"] = field(
        doc="Ksat tiled across nodes (n, nLyrs) [m s-1]"
    )
    dy: float = field(doc="Node width [m]")
    ele: Float[np.ndarray, " n"] = field(doc="Surface elevation in y-direction [m]")
    dwtToSto: Callable[[np.ndarray], np.ndarray] = field(
        doc="Water storage as function of water table depth"
    )
    stoToGwl: Callable[[np.ndarray], np.ndarray] = field(
        doc="Water table depth as function of storage"
    )
    dwtToTra: Callable[[np.ndarray], np.ndarray] = field(
        doc="Transmissivity as function of water table depth"
    )
    C: Callable[[np.ndarray], np.ndarray] = field(
        doc="Storage coefficient as function of water table depth"
    )
    dwtToRat: Callable[[np.ndarray], np.ndarray] = field(
        doc="Air-filled porosity ratio as function of water table depth"
    )
    dwtToAfp: Callable[[np.ndarray], np.ndarray] = field(
        doc="Air-filled porosity in root zone as function of water table depth"
    )


@dataclass(frozen=True)
class State:
    H: Float[np.ndarray, " n"] = field(doc="Hydraulic head relative to datum [m]")


@dataclass(frozen=True)
class ExfilOutputs:
    exfil: Float[np.ndarray, " n"] = field(
        doc="Water that cannot fit in soil pores [m]"
    )
    S: Float[np.ndarray, " n"] = field(
        doc="Actual source/sink after capping by air volume [m]"
    )


@dataclass(frozen=True)
class TimestepInputs:
    h0ts_west: float = field(doc="West boundary ditch depth [m]")
    h0ts_east: float = field(doc="East boundary ditch depth [m]")
    S: Float[np.ndarray, " n"] = field(
        doc="Source/sink after capping by air volume [m]"
    )
    surface_runoff: Float[np.ndarray, " n"] = field(
        doc="Surface runoff from mosslayer [m]"
    )


@dataclass(frozen=True)
class TimestepOutputs:
    roff: float = field(doc="Total runoff [m]")
    roffwest: float = field(doc="Runoff from west ditch [m]")
    roffeast: float = field(doc="Runoff from east ditch [m]")
    air_ratio: Float[np.ndarray, " n"] = field(doc="Air-filled porosity ratio")
    afp: Float[np.ndarray, " n"] = field(
        doc="Air-filled porosity in root zone [m3 m-3]"
    )


@dataclass(frozen=True)
class NumericalBuffer:
    """Pre-allocated workspace for the strip solver's iterative timestep.

    The solver functions (``run_timestep``, ``_amatrix``, ``_bound_const``)
    mutate the *contents* of these arrays in-place each timestep.
    Pre-allocating once per scenario avoids repeated ``np.zeros()`` calls
    in the hot path (5114+ timesteps).

    ``frozen=True`` prevents accidental rebinding of the array *references*
    (e.g. ``buffer.A = something_else``).  Numpy array *contents* are
    always mutable regardless of the frozen setting, which is the intended
    usage — the arrays are zeroed and refilled every call.

    Attributes
    ----------
    A  : np.ndarray, shape (n, n)
        Coefficient matrix of the tridiagonal system.
        Zero-filled at the start of each ``run_timestep`` call, then
        written by ``_amatrix`` and ``_bound_const`` every solver iteration.
    hs : np.ndarray, shape (n,)
        Right-hand side vector.  Reserved for a future refactoring of
        ``_right_side`` to write in-place, avoiding a per-call allocation.
    """

    A: Float[np.ndarray, " n n"]
    hs: Float[np.ndarray, " n"]


@dataclass(frozen=True)
class ResidenceTimeOutput:
    n: int = field(doc="Number of computation nodes")
    residence_time: Float[np.ndarray, " n"] = field(
        doc="Residence time of water from column to ditch [days]"
    )
    ixwest: tuple = field(doc="Indices of columns discharging to west ditch")
    ixeast: tuple = field(doc="Indices of columns discharging to east ditch")


def compute_constants(params: Params) -> ComputedConstants:
    dz = np.ones(params.nLyrs) * params.dzLyr
    z = np.cumsum(dz) - dz / 2.0
    if params.vonP:
        lenvp = len(params.vonP_top)
        vonP = np.ones(params.nLyrs) * params.vonP_bottom
        vonP[:lenvp] = params.vonP_top
        ptype = params.peat_type_bottom * params.nLyrs
        lenpt = len(params.peat_type)
        ptype[:lenpt] = params.peat_type
        pF, Ksat = peat_hydrol_properties(vonP, var="H", ptype=ptype)
    else:
        assert params.bd_top is not None
        lenbd = len(params.bd_top)
        bd = np.ones(params.nLyrs) * params.bd_bottom
        bd[:lenbd] = params.bd_top
        ptype = params.peat_type_bottom * params.nLyrs
        lenpt = len(params.peat_type)
        ptype[:lenpt] = params.peat_type
        pF, Ksat = peat_hydrol_properties(bd, var="bd", ptype=ptype)

    for i in range(params.nLyrs):
        if z[i] < 0.41:
            Ksat[i] = Ksat[i] * params.anisotropy

    (
        dwtToSto,
        stoToGwl,
        dwtToTra,
        C,
        dwtToRat,
        dwtToAfp,
    ) = CWTr(params.nLyrs, z, dz, pF, Ksat, direction="negative")

    dy = params.L / params.n
    sl = params.slope
    lev = 1.0
    ele = np.linspace(0, params.L * sl / 100.0, params.n) + lev
    Kmap = np.tile(Ksat, (params.n, 1))

    return ComputedConstants(
        dz=dz,
        z=z,
        pF=pF,
        Ksat=Ksat,
        Kmap=Kmap,
        dy=float(dy),
        ele=ele,
        dwtToSto=dwtToSto,
        stoToGwl=stoToGwl,
        dwtToTra=dwtToTra,
        C=C,
        dwtToRat=dwtToRat,
        dwtToAfp=dwtToAfp,
    )


def compute_initial_state(params: Params, constants: ComputedConstants) -> State:
    H = constants.ele + params.initial_h
    return State(H=H)


def assemble_timestep_inputs(
    h0ts_west: float,
    h0ts_east: float,
    exfil_out: ExfilOutputs,
    moss_rf_out: ReturnflowOutputs,
) -> TimestepInputs:

    return TimestepInputs(
        h0ts_west=h0ts_west,
        h0ts_east=h0ts_east,
        S=exfil_out.S,
        surface_runoff=moss_rf_out.surface_runoff,
    )


def compute_exfil(
    state: State,
    constants: ComputedConstants,
    h0ts_west: float,
    h0ts_east: float,
    p: Float[np.ndarray, " n"],
) -> ExfilOutputs:
    n = len(state.H)
    Htmp = state.H.copy()
    dwt = Htmp - constants.ele
    S = p.copy()
    dwt[0] = h0ts_west
    dwt[n - 1] = h0ts_east

    airv = np.maximum(
        constants.dwtToSto(np.zeros(n)) - constants.dwtToSto(dwt), np.zeros(n)
    )
    S = np.where(S > airv, airv, S)
    exfil = p - S
    return ExfilOutputs(exfil=exfil, S=S)


def _gmean_tr(Tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(Tr)
    trwest = np.maximum(Tr[: n - 1] * Tr[1:], 0.0)
    Trwest = np.sqrt(trwest)
    Trwest = np.append(Trwest, 0.0)
    treast = np.maximum(Tr[1:] * Tr[: n - 1], 0.0)
    Treast = np.sqrt(treast)
    Treast = np.insert(Treast, 0, 0.0)
    return Trwest, Treast


def _hadjacent(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(H)
    Hwest = H[: n - 1]
    Hwest = np.append(Hwest, 0.0)
    Heast = H[1:]
    Heast = np.insert(Heast, 0, 0.0)
    return Hwest, Heast


def _amatrix(
    A: np.ndarray,
    n: int,
    implic: float,
    Trwest: np.ndarray,
    Treast: np.ndarray,
    alfa: np.ndarray,
) -> np.ndarray:
    i, j = np.indices(A.shape)
    A[i == j] = implic * (Trwest + Treast) + alfa
    A[i == j + 1] = -implic * Trwest[: n - 1]
    A[i == j - 1] = -implic * Treast[1:]
    return A


def _bound_const(A: np.ndarray, n: int) -> np.ndarray:
    A[0, 0] = 1
    A[0, 1] = 0.0
    A[n - 1, n - 1] = 1.0
    A[n - 1, n - 2] = 0.0
    return A


def _right_side(
    S: np.ndarray,
    dt: float,
    dy: float,
    implic: float,
    alfa: np.ndarray,
    H: np.ndarray,
    Trminus0: np.ndarray,
    Hminus: np.ndarray,
    Trplus0: np.ndarray,
    Hplus: np.ndarray,
    DrIrr: bool,
    Htmp1: np.ndarray,
    ele: np.ndarray,
    h0_west: float,
    h0_east: float,
) -> np.ndarray:
    hs = (
        S * dt * dy**2
        + alfa * H
        + (1 - implic) * (Trminus0 * Hminus)
        - (1 - implic) * (Trminus0 + Trplus0) * H
        + (1 - implic) * (Trplus0 * Hplus)
    )
    n = len(Htmp1)

    if not DrIrr:
        hs[0] = Htmp1[1] if Htmp1[0] > Htmp1[1] else min(ele[0] + h0_west, Htmp1[1])
        hs[n - 1] = (
            Htmp1[n - 2]
            if Htmp1[n - 1] > Htmp1[n - 2]
            else min(ele[n - 1] + h0_east, Htmp1[n - 2])
        )
    else:
        hs[0] = ele[0] + h0_west
        hs[n - 1] = ele[n - 1] + h0_east
    return hs


def _runoff(
    H: np.ndarray,
    Trminus: np.ndarray,
    Trplus: np.ndarray,
    dt: float,
    dy: float,
    L: float,
) -> tuple[float, float]:
    roffwest = ((H[1] - H[0]) / dy * Trminus[0] * dt) / L
    roffeast = (H[-2] - H[-1]) / dy * Trplus[-1] * dt / L
    return roffwest, roffeast


def run_timestep(
    params: Params,
    constants: ComputedConstants,
    state: State,
    inputs: TimestepInputs,
    buffer: NumericalBuffer,
) -> tuple[State, TimestepOutputs]:
    n = params.n
    Htmp = state.H.copy()
    Htmp1 = state.H.copy()
    dwt = Htmp - constants.ele
    dwt[0] = inputs.h0ts_west
    dwt[n - 1] = inputs.h0ts_east

    Tr0 = constants.dwtToTra(dwt)
    Trminus0, Trplus0 = _gmean_tr(Tr0)
    Hminus, Hplus = _hadjacent(state.H)

    A = buffer.A
    A.fill(0)
    for _it in range(100):
        Tr0 = np.maximum(constants.dwtToTra(state.H - constants.ele), 0.0)
        Tr1 = np.maximum(constants.dwtToTra(Htmp1 - constants.ele), 0.0)
        CC = constants.C(Htmp1 - constants.ele)
        Trminus1, Trplus1 = _gmean_tr(Tr1)
        alfa = CC * constants.dy**2 / params.dt
        A = _amatrix(A, n, params.implic, Trminus1, Trplus1, alfa)
        A = _bound_const(A, n)
        hs = _right_side(
            inputs.S,
            params.dt,
            constants.dy,
            params.implic,
            alfa,
            state.H,
            Trminus0,
            Hminus,
            Trplus0,
            Hplus,
            params.DrIrr,
            Htmp1,
            constants.ele,
            inputs.h0ts_west,
            inputs.h0ts_east,
        )
        Htmp1 = np.linalg.multi_dot([np.linalg.inv(A), hs])
        Htmp1 = np.where(Htmp1 > constants.ele, constants.ele, Htmp1)
        conv = max(np.abs(Htmp1 - Htmp))
        Htmp = Htmp1.copy()
        if conv < 1.0e-7:
            break

    H_new = Htmp1.copy()
    roffwest, roffeast = _runoff(
        H_new, Trminus1, Trplus1, params.dt, constants.dy, params.L
    )
    roff = roffwest + roffeast
    new_dwt = H_new - constants.ele
    air_ratio = constants.dwtToRat(new_dwt)
    afp = constants.dwtToAfp(new_dwt)

    return State(H=H_new), TimestepOutputs(
        roff=roff,
        roffwest=roffwest,
        roffeast=roffeast,
        air_ratio=air_ratio,
        afp=afp,
    )


def compute_residence_time(
    params: Params,
    constants: ComputedConstants,
    mean_dwt: Float[np.ndarray, " n"],
) -> ResidenceTimeOutput:
    n = params.n
    porosity = 0.9
    K = 10.0 ** (-4) * 86400

    dist = np.arange(0, params.L, constants.dy)
    H = constants.ele + mean_dwt
    rtime = constants.dy / (K * np.gradient(H, dist) / porosity)

    ixwest = np.where(rtime > 0)
    ixeast = np.where(rtime < 0)

    timetoditch = np.zeros(n)
    timetoditch[ixwest] = np.cumsum(rtime[ixwest])
    timetoditch[ixeast] = np.flip(np.cumsum(np.flip(rtime[ixeast] * -1)))

    return ResidenceTimeOutput(
        n=n, residence_time=timetoditch, ixwest=ixwest, ixeast=ixeast
    )


def make_numerical_buffer(params: Params) -> NumericalBuffer:
    """
    Allocate scratch arrays for solution of the differential equation.

    Call this once per scenario and pass the result to every
    ``run_timestep`` call to avoid re-allocation overhead.

    (This is done to allocate an array to memory once,
    instead of allocating it at each timestep, which is less
    memory efficient).
    """
    n = params.n
    return NumericalBuffer(A=np.zeros((n, n)), hs=np.zeros(n))


def drain_depth_development(length, hdr, hdr20y):
    """
    Computes daily level of drain bottom thru the time of the simulation. Model adjusted from Hannu Hökkä drain model.
    Input:
        - drain depth in the beginning of the simulation (m, negative down)
        - drain depth after 20 yrs (m, negative down)
        - length of simulation in days
    Output:
        - daily drain bottom level (m, negative down)
    """
    timeyrs = np.linspace(
        0, length / 365.0, length
    )  # time vector telling the time elapsed from the simulation beginning time, yrs
    h0ts = (
        (-100 * hdr20y + 100.0 * hdr) / np.log(20.0) * np.log(timeyrs + 1.0) - 100 * hdr
    ) / -100.0  # Ditch model Hökkä
    return h0ts
