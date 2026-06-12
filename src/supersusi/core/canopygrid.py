from dataclasses import dataclass, field, replace

import numpy as np
from jaxtyping import Float

from supersusi.io.forcing_weather import WeatherForcings

eps = np.finfo(float).eps


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
    fmin: float = field(
        doc="Minimum phenology modifier (residual photosynthetic capacity) [-]"
    )

    # Initial conditions
    initial_W: Float[np.ndarray, " n"] = field(doc="Initial canopy water storage [mm]")
    initial_SWE: Float[np.ndarray, " n"] = field(
        doc="Initial snow water equivalent [mm]"
    )

    # Atmospheric pressure
    P: float = field(default=101300.0, doc="Air pressure [Pa]")

    # Defaults (constant across timesteps)
    U: float = field(default=2.0, doc="Wind speed [m s-1]")
    CO2: float = field(default=380.0, doc="CO2 concentration [ppm]")


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
    Ta: Float[np.ndarray, " n"] = field(doc="Air temperature [degC]")
    Prec: Float[np.ndarray, " n"] = field(doc="Precipitation rate [mm s-1]")
    Rg: Float[np.ndarray, " n"] = field(doc="Global radiation [W m-2]")
    Par: Float[np.ndarray, " n"] = field(
        doc="Photosynthetically active radiation [W m-2]"
    )
    VPD: Float[np.ndarray, " n"] = field(doc="Vapor pressure deficit [kPa]")
    hc: Float[np.ndarray, " n"] = field(doc="Canopy height [m]")
    LAIconif: Float[np.ndarray, " n"] = field(doc="Conifer leaf area index [m2 m-2]")
    Rew: Float[np.ndarray, " n"] = field(doc="Relative extractable water [-]")
    beta: Float[np.ndarray, " n"] = field(doc="Soil evaporation resistance factor [-]")


def compute_initial_state(
    params: Params,
) -> State:
    return State(
        W=params.initial_W,
        SWE=params.initial_SWE,
        SWEi=params.initial_SWE,
        SWEl=np.zeros_like(params.cf),
        X=np.zeros_like(params.cf),
        amax=np.full_like(params.cf, fill_value=params.amax_init),
    )


def assemble_inputs(
    forcings: WeatherForcings,
    hc: Float[np.ndarray, " n"],
    LAIconif: Float[np.ndarray, " n"],
    Rew: Float[np.ndarray, " n"],
    beta: Float[np.ndarray, " n"],
) -> Inputs:
    n = hc.shape[0]
    return Inputs(
        Ta=np.full(n, forcings.T),
        Prec=np.full(n, forcings.Prec / 86400.0),  # mm/day → mm/s
        Rg=np.full(n, forcings.Rg),
        Par=np.full(n, forcings.Par),
        VPD=np.full(n, forcings.VPD),
        hc=hc,
        LAIconif=LAIconif,
        Rew=Rew,
        beta=beta,
    )


def run_timestep(
    params: Params,
    previous_state: State,
    input: Inputs,
) -> tuple[State, Outputs]:
    # Deciduous LAI assumed constant (=lai_decid_max).
    # Seasonal dynamics (_lai_dynamics) not currently wired.
    lai = input.LAIconif + params.lai_decid_max

    EPSILON = 0.01
    cf = params.cf + EPSILON

    Rn = np.maximum(2.57 * lai / (2.57 * lai + 0.57) - 0.2, 0.55) * input.Rg

    X_new = previous_state.X + 1.0 / params.tau * (input.Ta - previous_state.X)
    S = np.maximum(X_new - params.xo, 0.0)
    fPheno = np.maximum(params.fmin, np.minimum(S / params.smax, 1.0))

    Ra, _, Ras, _, _, _ = _aerodynamics(
        lai,
        input.hc,
        params.U,
        w=0.01,
        zmeas=params.zmeas,
        zg=params.zground,
        zos=params.zo_ground,
    )

    W_new, SWEi_new, SWEl_new, PotInf, Trfall, Evap, Interc, MBE = _canopy_water_snow(
        params,
        params.dt,
        input.Ta,
        input.Prec,
        Rn,
        input.VPD,
        params.U,
        Ra,
        lai,
        cf,
        previous_state.W,
        previous_state.SWEi,
        previous_state.SWEl,
    )
    SWE_new = SWEi_new + SWEl_new

    Transpi, Efloor, _ = _dry_canopy_et(
        params,
        lai,
        params.lai_decid_max,
        input.LAIconif,
        previous_state.amax,
        input.VPD,
        input.Par,
        Rn,
        input.Ta,
        params.CO2,
        input.Rew,
        input.beta,
        fPheno,
        SWE_new,
        Ra,
        Ras,
    )

    transpi = Transpi * params.dt
    efloor = Efloor * params.dt
    et = transpi + efloor

    new_state = replace(
        previous_state,
        W=W_new,
        SWE=SWE_new,
        SWEi=SWEi_new,
        SWEl=SWEl_new,
        X=X_new,
    )

    outputs = Outputs(
        potinf=PotInf * 1e-3,
        trfall=Trfall * 1e-3,
        interc=Interc * 1e-3,
        evap=Evap * 1e-3,
        et=et * 1e-3,
        transpi=transpi * 1e-3,
        efloor=efloor * 1e-3,
        mbe=MBE * 1e-3,
        swe=SWE_new,
    )

    return new_state, outputs


def update_amax(nutstat, state: State) -> State:
    new_amax = state.amax * 1.0 + (nutstat - 1)  # 1.035

    return replace(state, amax=new_amax)


def _e_sat(
    T: Float[np.ndarray, " n"], P: float = 101300.0
) -> tuple[
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
]:
    NT = 273.15
    cp = 1004.67

    Lambda = 1e3 * (3147.5 - 2.37 * (T + NT))
    esa = 1e3 * (0.6112 * np.exp((17.67 * T) / (T + 273.16 - 29.66)))

    s = 17.502 * 240.97 * esa / ((240.97 + T) ** 2)
    g = P * cp / (0.622 * Lambda)
    return esa, s, g


def _penman_monteith(
    AE: Float[np.ndarray, " n"],
    D: Float[np.ndarray, " n"],
    T: Float[np.ndarray, " n"],
    Gs: Float[np.ndarray, " n"],
    Ga: Float[np.ndarray, " n"],
    P: float = 101300.0,
    units: str = "W",
) -> Float[np.ndarray, " n"]:
    cp = 1004.67
    rho = 1.25
    Mw = 18e-3
    _, s, g = _e_sat(T, P)
    L = 1e3 * (3147.5 - 2.37 * (T + 273.15))

    x = (s * AE + rho * cp * Ga * D) / (s + g * (1.0 + Ga / Gs))

    if units == "mm":
        x = x / L
    if units == "mol":
        x = x / L / Mw

    x = np.maximum(x, 0.0)
    return x


def _aerodynamics(
    LAI: Float[np.ndarray, " n"],
    hc: Float[np.ndarray, " n"],
    Uo: float,
    w: float = 0.01,
    zmeas: float = 2.0,
    zg: float = 0.5,
    zos: float = 0.01,
) -> tuple[
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
]:
    zm = hc + zmeas
    kv = 0.4
    beta = 285.0
    alpha = LAI / 2.0
    d = 0.66 * hc
    zom = 0.123 * hc
    zov = 0.1 * zom
    zosv = 0.1 * zos

    ustar = Uo * kv / np.log((zm - d) / zom)
    Uh = ustar / kv * np.log((hc - d) / zom)

    zn = np.minimum(zg / hc, 1.0)
    Ug = Uh * np.exp(alpha * (zn - 1.0))

    ra = 1.0 / (kv**2.0 * Uo) * np.log((zm - d) / zom) * np.log((zm - d) / zov)
    rb = 1.0 / LAI * beta * ((w / Uh) * (alpha / (1.0 - np.exp(-alpha / 2.0)))) ** 0.5

    ras = 1.0 / (kv**2.0 * Ug) * (np.log(zg / zos)) * np.log(zg / (zosv))

    ra = ra + rb
    return ra, rb, ras, ustar, Uh, Ug


def _canopy_water_snow(
    params: Params,
    dt: float,
    T: Float[np.ndarray, " n"],
    Prec: Float[np.ndarray, " n"],
    AE: Float[np.ndarray, " n"],
    D: Float[np.ndarray, " n"],
    U: float,
    Ra: Float[np.ndarray, " n"],
    lai: Float[np.ndarray, " n"],
    cf: Float[np.ndarray, " n"],
    W: Float[np.ndarray, " n"],
    SWEi: Float[np.ndarray, " n"],
    SWEl: Float[np.ndarray, " n"],
) -> tuple[
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
]:
    Tmin = 0.0
    Tmax = 1.0
    Tmelt = 0.0

    Wmax = params.wmax * lai
    Wmaxsnow = params.wmaxsnow * lai

    Kmelt = params.kmelt - 1.64 * cf / dt
    Kfreeze = params.kfreeze

    tau = np.exp(-params.kp * lai)
    gridshape = np.shape(lai)

    if np.shape(T) != gridshape:
        T = np.ones(gridshape) * T
        Prec = np.ones(gridshape) * Prec
        AE = np.ones(gridshape) * AE
        D = np.ones(gridshape) * D
        Ra = np.ones(gridshape) * Ra

    Prec = Prec * dt

    Lv = 1e3 * (3147.5 - 2.37 * (T + 273.15))
    Ls = Lv + 3.3e5

    erate = np.zeros(gridshape)
    ixs = np.where((Prec == 0) & (T <= Tmin))
    ixr = np.where((Prec == 0) & (T > Tmin))
    Ga = 1.0 / Ra

    Ce = 0.01 * ((W + eps) / Wmaxsnow) ** (-0.4)
    Sh = 1.79 + 3.0 * U**0.5
    gi = Sh * W * Ce / 7.68 + eps

    erate[ixs] = (
        dt
        / Ls[ixs]
        * _penman_monteith(
            (1.0 - tau[ixs]) * AE[ixs],
            1e3 * D[ixs],
            T[ixs],
            gi[ixs],
            Ga[ixs],
            units="W",
        )
    )

    gs = 1e6 * np.ones(gridshape)
    erate[ixr] = (
        dt
        / Lv[ixr]
        * _penman_monteith(
            (1.0 - tau[ixr]) * AE[ixr],
            1e3 * D[ixr],
            T[ixr],
            gs[ixr],
            Ga[ixr],
            units="W",
        )
    )

    fW = np.zeros(gridshape)
    fS = np.zeros(gridshape)
    fW[T >= Tmax] = 1.0
    fS[T <= Tmin] = 1.0
    ix = np.where((T > Tmin) & (T < Tmax))
    fW[ix] = (T[ix] - Tmin) / (Tmax - Tmin)
    fS[ix] = 1.0 - fW[ix]

    Unload = np.zeros(gridshape)
    Interc = np.zeros(gridshape)
    Melt = np.zeros(gridshape)
    Freeze = np.zeros(gridshape)

    Wo = W.copy()
    SWEo = SWEi + SWEl

    ix = T >= Tmax
    Unload[ix] = np.maximum(W[ix] - Wmax[ix], 0.0)
    W = W - Unload

    ix = T < Tmin
    Interc[ix] = (Wmaxsnow[ix] - W[ix]) * (
        1.0 - np.exp(-(cf[ix] / Wmaxsnow[ix]) * Prec[ix])
    )

    ix = T >= Tmin
    Interc[ix] = np.maximum(0.0, (Wmax[ix] - W[ix])) * (
        1.0 - np.exp(-(cf[ix] / Wmax[ix]) * Prec[ix])
    )
    W = W + Interc

    Trfall = Prec + Unload - Interc

    Evap = np.minimum(erate, W)
    W = W - Evap

    ix = np.where(T >= Tmelt)
    Melt[ix] = np.minimum(SWEi[ix], Kmelt[ix] * dt * (T[ix] - Tmelt))
    ix = np.where(T < Tmelt)
    Freeze[ix] = np.minimum(SWEl[ix], Kfreeze * dt * (Tmelt - T[ix]))

    Sice = np.maximum(0.0, SWEi + fS * Trfall + Freeze - Melt)
    Sliq = np.maximum(0.0, SWEl + fW * Trfall - Freeze + Melt)

    PotInf = np.maximum(0.0, Sliq - Sice * params.r)
    Sliq = np.maximum(0.0, Sliq - PotInf)

    SWEi_new = Sice
    SWEl_new = Sliq
    SWE_new = SWEi_new + SWEl_new

    MBE = (W + SWE_new) - (Wo + SWEo) - (Prec - Evap - PotInf)

    return W, SWEi_new, SWEl_new, PotInf, Trfall, Evap, Interc, MBE


def _dry_canopy_et(
    params: Params,
    lai: Float[np.ndarray, " n"],
    lai_decid: Float[np.ndarray, " n"],
    lai_conif: Float[np.ndarray, " n"],
    amax: Float[np.ndarray, " n"],
    D: Float[np.ndarray, " n"],
    Qp: Float[np.ndarray, " n"],
    AE: Float[np.ndarray, " n"],
    Ta: Float[np.ndarray, " n"],
    CO2: float,
    Rew: Float[np.ndarray, " n"],
    beta: Float[np.ndarray, " n"],
    fPheno: Float[np.ndarray, " n"],
    SWE: Float[np.ndarray, " n"],
    Ra: Float[np.ndarray, " n"],
    Ras: Float[np.ndarray, " n"],
) -> tuple[
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
]:
    rhoa = params.P / (8.31 * (Ta + 273.15))

    Amax = 1.0 / lai * (lai_conif * amax + lai_decid * amax)

    g1 = 1.0 / lai * (lai_conif * params.g1_conif + lai_decid * params.g1_decid)

    tau = np.exp(-params.kp * lai)

    fQ = (
        1.0
        / params.kp
        * np.log(
            (params.kp * Qp + params.q50)
            / (params.kp * Qp * np.exp(-params.kp * lai) + params.q50 + eps)
        )
    )

    fRew = Rew
    fCO2 = 1.0 - 0.387 * np.log(CO2 / 380.0)

    gs = 1.6 * (1.0 + g1 / np.sqrt(D)) * Amax / CO2 / rhoa

    Gc = gs * fQ * fRew * fCO2 * fPheno
    Gc[np.isnan(Gc)] = eps

    Tr = _penman_monteith(
        (1.0 - tau) * AE, 1e3 * D, Ta, Gc, 1.0 / Ra, P=params.P, units="mm"
    )
    Tr[Tr < 0] = 0.0

    Gcs = np.full_like(lai, params.gsoil)

    Efloor = beta * _penman_monteith(
        tau * AE, 1e3 * D, Ta, Gcs, 1.0 / Ras, P=params.P, units="mm"
    )
    Efloor[SWE > 0] = 0.0

    return Tr, Efloor, Gc
