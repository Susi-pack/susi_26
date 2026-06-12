# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:41:03 2022

@author: alauren

Functional rewrite of the organic matter decomposition module (Esom).

The substance parameter defines what this instance is for:
    'Mass'  — organic matter decomposition (combined CO2 and DOC)
    'N'     — nitrogen dynamics
    'P'     — phosphorus dynamics
    'K'     — potassium dynamics

Mass instance accounts for combined CO2 and DOC release from organic matter.
Lauren et al 2012, Lappalainen et al. 2018 ja Laurén et al 2019
L: DOC / CO2-C = 0.1; F,H, peat DOC / CO2-C = 0.05
Added here to Mass instance release k1: L to out 0.1, k2 F to out 0.05 ; H to peat 0.05

"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import diags
from jaxtyping import Float

from supersusi.core.susi_utils import peat_hydrol_properties, wrc
from supersusi.core.strip import ResidenceTimeOutput


@dataclass(frozen=True)
class SubstanceParams:
    substance: str = field(
        doc="'Mass' — organic matter decomposition (combined CO2 and DOC), 'N' — nitrogen dynamics, 'P' — phosphorus dynamics, 'K' — potassium dynamics"
    )
    n: int = field(doc="number of spatial columns")
    nLyrs: int = field(doc="number of soil layers")
    dz: Float[np.ndarray, " nLyrs"] = field(doc="layer thickness (m)")
    bd: Float[np.ndarray, " nLyrs"] = field(doc="bulk density (g cm-3)")
    sfc: Float[np.ndarray, " n"] = field(doc="site fertility class, integers 1–6")
    h_mor: float = field(doc="mor humus thickness (m)")
    rho_mor: float = field(doc="bulk density of the mor (kg m-3)")
    enable_peattop: bool = field(doc="enable decomposition in top peat layer")
    enable_peatmiddle: bool = field(doc="enable decomposition in middle peat layer")
    enable_peatbottom: bool = field(doc="enable decomposition in bottom peat layer")
    nutc_k1: float = field(doc="nutrient release modifier k1 (mass release comparison)")
    nutc_k2: float = field(doc="nutrient release modifier k2 (mass release comparison)")
    nutc_k6: float = field(doc="nutrient release modifier k6 (mass release comparison)")
    contpara: dict = field(
        doc="peat N/P/K content by SFC class and specification, unit gravimetric %"
    )
    contpara_mor: float = field(
        doc="initial concentration in mor layer, gravimetric %, Tammjinen et al. Plant and Soil 259: 51–58, 2004"
    )
    dph: dict = field(doc="pH by site fertility class (1→3.9 … 6→3.0)")
    contpara_override: float | None = field(
        default=None, doc="overrides peat N/P/K content for all SFC classes"
    )
    bound1: float = field(
        default=0.2, doc="boundary between top and middle peat layers (m)"
    )
    bound2: float = field(
        default=0.4, doc="boundary between middle and bottom peat layers (m)"
    )
    sfc_specification: int = field(
        default=2, doc="peat content specification index (MTkg1/MTkg2)"
    )
    ash: float = field(default=5.0, doc="litter ash content in gravimetric %")
    litterN: float = field(default=1.2, doc="litter N content in gravimetric %")
    frac_L: float = field(
        default=0.1, doc="share of undecomposed litter (L) from mor thickness (0…1)"
    )
    frac_F: float = field(
        default=0.2,
        doc="share of partly decomposed F material from mor thickness (0…1)",
    )
    frac_H: float = field(
        default=0.7, doc="share of humified H material from mor thickness (0…1)"
    )
    frac_leaf: float = field(
        default=0.2, doc="share of non-woody material from L and F (0…1)"
    )
    frac_woody: float = field(
        default=0.8, doc="share of woody material from L and F (0…1)"
    )
    nitrogen: float = field(default=0.5, doc="total N content in gravimetric %")
    lignin: float = field(default=25.0, doc="lignin content in gravimetric %")
    adjust: float = field(default=1.0, doc="adjustment factor for lignin/N ratio")


@dataclass(frozen=True)
class SubstanceComputedConstants:
    z: Float[np.ndarray, " nLyrs"] = field(doc="depth of each layer center point (m)")
    idtop: np.ndarray = field(doc="indices of top peat layers (z < bound1)")
    idmiddle: np.ndarray = field(
        doc="indices of middle peat layers (bound1 ≤ z < bound2)"
    )
    idbottom: np.ndarray = field(doc="indices of bottom peat layers (z ≥ bound2)")
    pF: dict = field(doc="water retention pF curves per layer")
    wtToVfAir_top: interp1d = field(
        doc="water table → volume fraction of air, top layer"
    )
    wtToVfAir_middle: interp1d = field(
        doc="water table → volume fraction of air, middle layer"
    )
    wtToVfAir_bottom: interp1d = field(
        doc="water table → volume fraction of air, bottom layer"
    )
    t2: interp1d = field(doc="temperature function for k2")
    t3: interp1d = field(doc="temperature function for k3")
    t4: interp1d = field(doc="temperature function for k4")
    t5: interp1d = field(doc="temperature function for k5")
    t6: interp1d = field(doc="temperature function for k6 (peat top)")
    t7: interp1d = field(doc="temperature function for k7/k8/k9 (peat layers)")
    phi1236: interp1d = field(doc="moisture function for k1, k2, k3, k6")
    phi4: interp1d = field(doc="moisture function for k4")
    phi5: interp1d = field(doc="moisture function for k5")
    mu_k1: float = field(doc="woody decomposition modifier for k1 from lignin/N ratio")
    mu_k2: float = field(doc="woody decomposition modifier for k2 from lignin/N ratio")
    mu_k3: float = field(doc="woody decomposition modifier for k3 from lignin/N ratio")


@dataclass(frozen=True)
class Params:
    mass: SubstanceParams
    n: SubstanceParams
    p: SubstanceParams
    k: SubstanceParams


@dataclass(frozen=True)
class ComputedConstants:
    mass: SubstanceComputedConstants
    n: SubstanceComputedConstants
    p: SubstanceComputedConstants
    k: SubstanceComputedConstants


@dataclass(frozen=True)
class State:
    M: Float[np.ndarray, "1 n 11"] = field(
        doc="mass matrix: 0=L0L leaf litter input, 1=L0W woody input, 2=LL leaf litter, 3=LW woody litter, 4=FL leaf F material, 5=FW woody F material, 6=H humus, 7=P1 peat 0-30cm, 8=P2 peat 30-60cm, 9=P3 peat 60cm-bottom, 10=Out cumulative output (kg m-2)"
    )
    i: int = field(doc="day counter")
    previous_mass: Float[np.ndarray, " n"] = field(
        doc="previous year's cumulative output scaled by 10000"
    )
    pH: Float[np.ndarray, "1 n"] = field(doc="soil pH per column")


@dataclass(frozen=True)
class Inputs:
    tair_ts: Float[np.ndarray, " days"] = field(doc="daily air temperatures (deg C)")
    tp_top_ts: Float[np.ndarray, " days"] = field(
        doc="peat temperature at -0.125 m depth (deg C)"
    )
    tp_middle_ts: Float[np.ndarray, " days"] = field(
        doc="peat temperature at -0.4 m depth (deg C)"
    )
    tp_bottom_ts: Float[np.ndarray, " days"] = field(
        doc="peat temperature at -0.75 m depth (deg C)"
    )
    water_tables: Float[np.ndarray, "days n"] = field(doc="daily water table depth (m)")
    nonwoodylitter: Float[np.ndarray, " n"] = field(
        doc="leaf and fine root litter input (kg m-2)"
    )
    woodylitter: Float[np.ndarray, " n"] = field(
        doc="branch and coarse root litter input (kg m-2)"
    )


@dataclass(frozen=True)
class YearOutputs:
    out: Float[np.ndarray, " n"] = field(doc="annual mass output (kg ha-1)")
    out_root_lyr: Float[np.ndarray, " n"] = field(
        doc="mass output from root layer (kg ha-1)"
    )
    out_below_root_lyr: Float[np.ndarray, " n"] = field(
        doc="mass output below root layer (kg ha-1)"
    )
    nonwoodylitter: Float[np.ndarray, " n"] = field(
        doc="leaf and fine root litter (kg m-2)"
    )
    woodylitter: Float[np.ndarray, " n"] = field(
        doc="branch and coarse root litter (kg m-2)"
    )
    daily_cumulative_out: Float[np.ndarray, "n days"] | None = field(
        default=None, doc="daily cumulative output (Mass only)"
    )


@dataclass(frozen=True)
class DOCExportOutputs:
    hmw: Float[np.ndarray, " n"] = field(doc="high molecular weight DOC (kg C)")
    lmw: Float[np.ndarray, " n"] = field(doc="low molecular weight DOC (kg C)")
    hmwtoditch: Float[np.ndarray, " n"] = field(
        doc="HMW DOC after biodegradation (kg C)"
    )
    lmwtoditch: Float[np.ndarray, " n"] = field(
        doc="LMW DOC after biodegradation (kg C)"
    )
    hmw_to_west: float = field(doc="HMW DOC export to west ditch")
    hmw_to_east: float = field(doc="HMW DOC export to east ditch")
    lmw_to_west: float = field(doc="LMW DOC export to west ditch")
    lmw_to_east: float = field(doc="LMW DOC export to east ditch")


def _build_substance_params(
    substance: str,
    n: int,
    nLyrs: int,
    dz: Float[np.ndarray, " nLyrs"],
    bd: Float[np.ndarray, " nLyrs"],
    sfc: Float[np.ndarray, " n"],
    h_mor: float,
    rho_mor: float,
    enable_peattop: bool,
    enable_peatmiddle: bool,
    enable_peatbottom: bool,
    peat_override: float | None = None,
) -> SubstanceParams:
    nutcpara = {
        "Mass": {"k1": 1.05, "k2": 1.05, "k6": 1.05},
        "N": {"k1": 0.1, "k2": 0.5, "k6": 0.175},
        "P": {"k1": 1.1, "k2": 0.45, "k6": 0.3},
        "K": {"k1": 1.5, "k2": 1.5, "k6": 1.5},
    }
    contpara = {
        "Mass": {
            2: {1: 100.0, 2: 100.0},
            3: {1: 100.0, 2: 100.0},
            4: {1: 100.0, 2: 100.0},
            5: {1: 100.0, 2: 100.0},
        },
        "N": {
            2: {1: 1.9, 2: 1.9},
            3: {1: 1.6, 2: 1.6},
            4: {1: 1.4, 2: 1.4},
            5: {1: 1.2, 2: 1.2},
        },
        "P": {
            2: {1: 0.1, 2: 0.1},
            3: {1: 0.08, 2: 0.08},
            4: {1: 0.06, 2: 0.06},
            5: {1: 0.05, 2: 0.05},
        },
        "K": {
            2: {1: 0.045, 2: 0.045},
            3: {1: 0.04, 2: 0.038},
            4: {1: 0.037, 2: 0.034},
            5: {1: 0.032, 2: 0.032},
        },
    }
    contpara_mor = {
        "Mass": 100.0,
        "N": 1.5,
        "P": 0.099,
        "K": 0.089,
    }
    dph = {
        1: 3.9,
        2: 3.8,
        3: 3.5,
        4: 3.2,
        5: 3.1,
        6: 3.0,
    }

    nutc = nutcpara[substance]
    contp = contpara[substance]
    cp_mor = contpara_mor[substance]

    if peat_override is not None:
        sfcs = np.unique(sfc)
        for s in sfcs:
            s_int = int(s)
            if s_int in contp:
                contp[s_int][1] = peat_override
                contp[s_int][2] = peat_override

    return SubstanceParams(
        substance=substance,
        n=n,
        nLyrs=nLyrs,
        dz=dz,
        bd=bd,
        sfc=sfc,
        h_mor=h_mor,
        rho_mor=rho_mor,
        enable_peattop=enable_peattop,
        enable_peatmiddle=enable_peatmiddle,
        enable_peatbottom=enable_peatbottom,
        nutc_k1=nutc["k1"],
        nutc_k2=nutc["k2"],
        nutc_k6=nutc["k6"],
        contpara=contp,
        contpara_mor=cp_mor,
        dph=dph,
        contpara_override=peat_override,
    )


def build_params(
    n: int,
    nLyrs: int,
    dzLyr: float,
    vonP: bool,
    vonP_top: list[int],
    vonP_bottom: int,
    bd_top: list[float] | None,
    bd_bottom: float,
    sfc: Float[np.ndarray, " n"],
    h_mor: float,
    rho_mor: float,
    enable_peattop: bool,
    enable_peatmiddle: bool,
    enable_peatbottom: bool,
    peatN: float | None,
    peatP: float | None,
    peatK: float | None,
) -> Params:
    dz = np.ones(nLyrs) * dzLyr
    if vonP:
        vpost = vonP_bottom * np.ones(nLyrs)
        vpost[: len(vonP_top)] = vonP_top
        bd = 0.035 + 0.0159 * vpost
    else:
        bd = bd_bottom * np.ones(nLyrs)
        if bd_top is not None:
            bd[: len(bd_top)] = bd_top

    return Params(
        mass=_build_substance_params(
            "Mass",
            n,
            nLyrs,
            dz,
            bd,
            sfc,
            h_mor,
            rho_mor,
            enable_peattop,
            enable_peatmiddle,
            enable_peatbottom,
        ),
        n=_build_substance_params(
            "N",
            n,
            nLyrs,
            dz,
            bd,
            sfc,
            h_mor,
            rho_mor,
            enable_peattop,
            enable_peatmiddle,
            enable_peatbottom,
            peat_override=peatN,
        ),
        p=_build_substance_params(
            "P",
            n,
            nLyrs,
            dz,
            bd,
            sfc,
            h_mor,
            rho_mor,
            enable_peattop,
            enable_peatmiddle,
            enable_peatbottom,
            peat_override=peatP,
        ),
        k=_build_substance_params(
            "K",
            n,
            nLyrs,
            dz,
            bd,
            sfc,
            h_mor,
            rho_mor,
            enable_peattop,
            enable_peatmiddle,
            enable_peatbottom,
            peat_override=peatK,
        ),
    )


def _build_wt_to_vf_air(
    bd_slice: Float[np.ndarray, " lyrs"],
    z_slice: Float[np.ndarray, " lyrs"],
    dz_slice: Float[np.ndarray, " lyrs"],
) -> interp1d:
    gwl = np.linspace(0, -6, 150)
    pF_slice, _ = peat_hydrol_properties(bd_slice, var="bd", ptype="A")
    water_sto = np.array(
        [np.sum(wrc(pF_slice, x=np.minimum(z_slice + g, 0.0)) * dz_slice) for g in gwl]
    )
    volume_fraction_of_air = (water_sto[0] - water_sto) / water_sto[0]
    return interp1d(
        gwl,
        volume_fraction_of_air,
        fill_value=(volume_fraction_of_air[0], volume_fraction_of_air[-1]),
        bounds_error=False,
    )


def _get_rates(
    params: SubstanceParams,
    cc: SubstanceComputedConstants,
    tair: float,
    tp_top: float,
    tp_middle: float,
    tp_bottom: float,
    wn: Float[np.ndarray, " n"],
    peat_w1: Float[np.ndarray, " n"],
    peat_w2: Float[np.ndarray, " n"],
    peat_w3: Float[np.ndarray, " n"],
    pH: Float[np.ndarray, "1 n"],
) -> tuple[
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
    Float[np.ndarray, " n"],
]:
    """Compute decomposition rate arrays k1…k9 from current environmental conditions."""
    # pH function, Romul documentation Table 1
    nu = np.clip(0, 0.701 * pH - 1.6018 - 0.038 * pH**2, 1)

    k1 = (
        (0.002 + 0.00009 * params.ash + 0.003 * params.litterN)
        * min(0.1754 * np.exp(0.0871 * tair), 1.0)
        * cc.phi1236(wn)
        * nu
    )
    k2 = np.clip(
        (0.00114 - 0.00028 * params.litterN) * cc.t2(tair) * cc.phi1236(wn) * nu,
        0.0,
        1.0,
    )
    k3 = np.clip(
        (0.04 - 0.003 * params.litterN) * cc.t3(tair) * cc.phi1236(wn), 0.0, 1.0
    )
    k4 = 0.005 * params.litterN * cc.t4(tair) * cc.phi4(wn)
    k5 = 0.007 * cc.t5(tair) * cc.phi5(wn)
    k6 = 0.001 * cc.t6(tp_top) * cc.phi1236(wn) * nu

    # -> temperature separately for top peat 30 cm (take from 15 cm)
    # -> air filled porosity separately for the top and bottom ()
    k7 = (
        0.00045 * cc.t7(tp_top) * peat_w1 * params.enable_peattop
    )  # Lappalainen et al 2018, gamma/VfAir slightly decomposed peat
    k8 = (
        0.0001 * cc.t7(tp_middle) * peat_w2 * params.enable_peatmiddle
    )  # Lappalainen et al. 2018 gamma/VfAir highly decomposed
    k9 = (
        0.0001 * cc.t7(tp_bottom) * peat_w3 * params.enable_peatbottom * 0.1
    )  # 0.25  #0.5

    return k1, k2, k3, k4, k5, k6, k7, k8, k9


def _decompose(
    params: SubstanceParams,
    cc: SubstanceComputedConstants,
    k1: Float[np.ndarray, " n"],
    k2: Float[np.ndarray, " n"],
    k3: Float[np.ndarray, " n"],
    k4: Float[np.ndarray, " n"],
    k5: Float[np.ndarray, " n"],
    k6: Float[np.ndarray, " n"],
    k7: Float[np.ndarray, " n"],
    k8: Float[np.ndarray, " n"],
    k9: Float[np.ndarray, " n"],
    M: Float[np.ndarray, "1 n 11"],
) -> Float[np.ndarray, "1 n 11"]:
    """Advance the mass matrix M one timestep via sparse matrix multiplication."""
    diagonal_shape = (1, params.n, 11)

    # Staying fraction
    k_diag = np.zeros(diagonal_shape)
    k_diag[:, :, 2] = 1 - (params.nutc_k1 * k1 + k3)
    k_diag[:, :, 3] = 1 - (params.nutc_k1 * k1 * cc.mu_k1 + k3 * cc.mu_k3)
    k_diag[:, :, 4] = 1 - (params.nutc_k2 * k2 + k4 + k5)
    k_diag[:, :, 5] = 1 - (params.nutc_k2 * k2 * cc.mu_k2 + k4 + k5)
    k_diag[:, :, 6] = 1 - (params.nutc_k6 * k6)
    k_diag[:, :, 7] = 1 - (params.nutc_k6 * k7)
    k_diag[:, :, 8] = 1 - (params.nutc_k6 * k8)
    k_diag[:, :, 9] = 1 - (params.nutc_k6 * k9)
    k_diag[:, :, 10] = 1
    k_diag = np.ravel(k_diag)

    k_low0 = np.zeros(diagonal_shape)
    k_low0[:, :, 6] = k4 + k5  # to H from FW
    k_low0[:, :, 10] = params.nutc_k6 * k9  # to Out from P3
    k_low0 = np.ravel(k_low0)

    k_low1 = np.zeros(diagonal_shape)
    k_low1[:, :, 2] = 1  # to LL from L0L
    k_low1[:, :, 3] = 1  # to LW from L0W
    k_low1[:, :, 4] = k3  # to FL from LL
    k_low1[:, :, 5] = k3 * cc.mu_k3  # to FW from LW
    k_low1[:, :, 6] = k4 + k5  # to H from FL
    k_low1[:, :, 10] = params.nutc_k6 * k8  # to Out from P2
    k_low1 = np.ravel(k_low1)

    k_low2 = np.zeros(diagonal_shape)
    k_low2[:, :, 10] = params.nutc_k6 * k7  # to Out from P1
    k_low2 = np.ravel(k_low2)

    k_low3 = np.zeros(diagonal_shape)
    k_low3[:, :, 10] = params.nutc_k6 * k6  # to Out from H
    k_low3 = np.ravel(k_low3)

    k_low4 = np.zeros(diagonal_shape)
    k_low4[:, :, 10] = params.nutc_k2 * k2 * cc.mu_k2  # to Out from FW
    k_low4 = np.ravel(k_low4)

    k_low5 = np.zeros(diagonal_shape)
    k_low5[:, :, 10] = params.nutc_k2 * k2  # to Out from FL
    k_low5 = np.ravel(k_low5)

    k_low6 = np.zeros(diagonal_shape)
    k_low6[:, :, 10] = params.nutc_k1 * k1 * cc.mu_k1  # to Out from LW
    k_low6 = np.ravel(k_low6)

    k_low7 = np.zeros(diagonal_shape)
    k_low7[:, :, 10] = params.nutc_k1 * k1  # to Out from LL
    k_low7 = np.ravel(k_low7)

    length = len(k_diag)
    kmat = diags(
        diagonals=[
            k_diag,
            k_low0[1:],
            k_low1[2:],
            k_low2[3:],
            k_low3[4:],
            k_low4[5:],
            k_low5[6:],
            k_low6[7:],
            k_low7[8:],
        ],  # Create the main matrix
        offsets=[0, -1, -2, -3, -4, -5, -6, -7, -8],
        shape=(length, length),
        format="csr",
    )

    M_tmp = np.ravel(M)
    M_tmp = kmat @ M_tmp
    return np.reshape(M_tmp, (1, params.n, 11))


def compute_all_constants(params: Params) -> ComputedConstants:
    return ComputedConstants(
        mass=compute_constants(params.mass),
        n=compute_constants(params.n),
        p=compute_constants(params.p),
        k=compute_constants(params.k),
    )


# ── Public API ───────────────────────────────────────────────────────────────────


def compute_constants(params: SubstanceParams) -> SubstanceComputedConstants:
    z = np.cumsum(params.dz) - params.dz / 2.0

    idtop = np.where(z < params.bound1)[0]
    idmiddle = np.where((z >= params.bound1) & (z < params.bound2))[0]
    idbottom = np.where(z >= params.bound2)[0]

    pF, _ = peat_hydrol_properties(params.bd, var="bd", ptype="A")

    wtToVfAir_top = _build_wt_to_vf_air(params.bd[idtop], z[idtop], params.dz[idtop])
    wtToVfAir_middle = _build_wt_to_vf_air(
        params.bd[idmiddle], z[idmiddle], params.dz[idmiddle]
    )
    wtToVfAir_bottom = _build_wt_to_vf_air(
        params.bd[idbottom], z[idbottom], params.dz[idbottom]
    )

    t2 = interp1d([-40, -5, -1, 25, 35, 60], [0, 0, 0.2, 1.53, 1.53, 0])
    t3 = interp1d([-40, -3, 0, 7, 60], [0, 0, 1.3, 1.3, 0])
    t4 = interp1d([-40, -5, 1, 20, 40, 80], [0, 0, 0.2, 1, 1, 0])
    t5 = interp1d([-40, -5, 1, 13, 25, 50], [0, 0, 0.2, 1, 1, 0])
    t6 = interp1d([-40, -5, 1, 27.5, 35, 60], [0, 0, 0.2, 1.95, 1.95, 0])
    t7 = interp1d(
        [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 8],
    )

    phi1236 = interp1d(
        [
            0.02,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.417,
            1.333,
            1.4,
            1.6,
            1.8,
            2.0,
            2.2,
            2.4,
            2.6,
            2.8,
            4,
        ],
        [
            0,
            0.004,
            0.026,
            0.074,
            0.154,
            0.271,
            0.432,
            0.64,
            0.899,
            1.0,
            1.0,
            0.844,
            0.508,
            0.305,
            0.184,
            0.111,
            0.067,
            0.04,
            0.024,
            0,
        ],
    )
    phi4 = interp1d([0, 0.133, 1.333, 2.333, 4], [0, 1, 1, 0, 0])
    phi5 = interp1d([0, 0.067, 0.5, 2.333, 4, 10], [0, 0, 1, 1, 0, 0])

    ln_ratio = params.lignin / params.nitrogen
    mu_k1 = 0.092 * ln_ratio**-0.7396 * params.adjust
    mu_k2 = 0.0027 * ln_ratio**-0.3917 * params.adjust
    mu_k3 = 0.062 * ln_ratio**-0.3972 * params.adjust

    return SubstanceComputedConstants(
        z=z,
        idtop=idtop,
        idmiddle=idmiddle,
        idbottom=idbottom,
        pF=pF,
        wtToVfAir_top=wtToVfAir_top,
        wtToVfAir_middle=wtToVfAir_middle,
        wtToVfAir_bottom=wtToVfAir_bottom,
        t2=t2,
        t3=t3,
        t4=t4,
        t5=t5,
        t6=t6,
        t7=t7,
        phi1236=phi1236,
        phi4=phi4,
        phi5=phi5,
        mu_k1=mu_k1,
        mu_k2=mu_k2,
        mu_k3=mu_k3,
    )


def compute_initial_state(
    params: SubstanceParams, cc: SubstanceComputedConstants
) -> State:
    M = np.zeros((1, params.n, 11))

    LL_mass = (
        params.h_mor
        * params.frac_L
        * params.rho_mor
        * params.frac_leaf
        * params.contpara_mor
        / 100.0
    )
    LW_mass = (
        params.h_mor
        * params.frac_L
        * params.rho_mor
        * params.frac_woody
        * params.contpara_mor
        / 100.0
    )
    FL_mass = (
        params.h_mor
        * params.frac_F
        * params.rho_mor
        * params.frac_leaf
        * params.contpara_mor
        / 100.0
    )
    FW_mass = (
        params.h_mor
        * params.frac_F
        * params.rho_mor
        * params.frac_woody
        * params.contpara_mor
        / 100.0
    )
    H_mass = params.h_mor * params.frac_H * params.rho_mor * params.contpara_mor / 100.0

    M[:, :, 2] = LL_mass
    M[:, :, 3] = LW_mass
    M[:, :, 4] = FL_mass
    M[:, :, 5] = FW_mass
    M[:, :, 6] = H_mass

    for scode in np.unique(params.sfc):
        scode_int = int(scode)
        mask = params.sfc == scode
        cont_array = (
            np.ones(params.nLyrs)
            * params.contpara[scode_int][params.sfc_specification]
            / 100.0
        )
        M[0, mask, 7] = np.sum(
            params.bd[cc.idtop] * params.dz[cc.idtop] * 1000.0 * cont_array[cc.idtop]
        )  # Peat material storage, top, kg m-2
        M[0, mask, 8] = np.sum(
            params.bd[cc.idmiddle]
            * params.dz[cc.idmiddle]
            * 1000.0
            * cont_array[cc.idmiddle]
        )  # Peat material storage, middle, kg m-2
        M[0, mask, 9] = np.sum(
            params.bd[cc.idbottom]
            * params.dz[cc.idbottom]
            * 1000.0
            * cont_array[cc.idbottom]
        )  # Peat material storage, bottom, kg m-2

    pH = np.zeros((1, params.n))
    for scode in np.unique(params.sfc):
        scode_int = int(scode)
        mask = params.sfc == scode
        pH[0, mask] = params.dph[scode_int]

    return State(
        M=M,
        i=0,
        previous_mass=np.zeros(params.n),
        pH=pH,
    )


def update_soil_pH(
    pH: Float[np.ndarray, "1 n"],
    params: SubstanceParams,
    increment: float,
) -> Float[np.ndarray, "1 n"]:
    new_pH = pH.copy()
    for scode in np.unique(params.sfc):
        mask = params.sfc == scode
        new_pH[0, mask] = params.dph[int(scode)] + increment
    return new_pH


def assemble_inputs(
    tair_ts: Float[np.ndarray, " days"],
    tp_top_ts: Float[np.ndarray, " days"],
    tp_middle_ts: Float[np.ndarray, " days"],
    tp_bottom_ts: Float[np.ndarray, " days"],
    water_tables: Float[np.ndarray, "days n"],
    nonwoodylitter: Float[np.ndarray, " n"],
    woodylitter: Float[np.ndarray, " n"],
) -> Inputs:
    return Inputs(
        tair_ts=tair_ts,
        tp_top_ts=tp_top_ts,
        tp_middle_ts=tp_middle_ts,
        tp_bottom_ts=tp_bottom_ts,
        water_tables=water_tables,
        nonwoodylitter=nonwoodylitter,
        woodylitter=woodylitter,
    )


def run_yr(
    params: SubstanceParams,
    cc: SubstanceComputedConstants,
    previous_state: State,
    inputs: Inputs,
) -> tuple[State, YearOutputs]:
    days = len(inputs.tair_ts)

    M = previous_state.M.copy()
    i = previous_state.i
    previous_mass = previous_state.previous_mass.copy()
    pH = previous_state.pH.copy()

    P2_ini = M[0, :, 8] * 10000.0  # middle layer peat mass at the beginning of yr
    P3_ini = M[0, :, 9] * 10000.0  # bottom layer peat mass at the beginning of yr

    L0L = np.zeros((1, params.n))  # Litterfall time series (leaf & fineroots), kg m-2
    L0L[0, :] = inputs.nonwoodylitter
    L0W = np.zeros((1, params.n))  # Litterfall time series (woody litter), kg m-2
    L0W[0, :] = inputs.woodylitter

    track_daily = params.substance == "Mass"
    if track_daily:
        daily_cumulative_out = np.zeros((params.n, days))

    for day in range(days):
        tair = inputs.tair_ts[day]  # Daily air temperatures, deg C
        tp_top = inputs.tp_top_ts[day]  # Peat temperature -0.125 m depth
        tp_middle = inputs.tp_middle_ts[day]  # Peat temperature -0.4 m depth
        tp_bottom = inputs.tp_bottom_ts[day]  # Peat temperature -0.75 m depth
        wts = inputs.water_tables[day, :]

        # Physical conditions in the peat profile
        wn = (
            wrc(cc.pF[0], wts) / wrc(cc.pF[0], -0.3)
        )  # Relative water content with respect to field capacity, pF[0] refers to water retention characteristics in the topmost layer
        peat_w1 = cc.wtToVfAir_top(
            wts
        )  # Call interpolation function WT -> volume fraction of air
        peat_w2 = cc.wtToVfAir_middle(
            wts
        )  # Call interpolation function WT -> volume fraction of air
        peat_w3 = cc.wtToVfAir_bottom(
            wts
        )  # Call interpolation function WT -> volume fraction of air

        k1, k2, k3, k4, k5, k6, k7, k8, k9 = _get_rates(
            params,
            cc,
            tair,
            tp_top,
            tp_middle,
            tp_bottom,
            wn,
            peat_w1,
            peat_w2,
            peat_w3,
            pH,
        )

        if day == 243:  # end of August — fresh litter input
            M[:, :, 0] = L0L  # fresh litter leaves and fine roots, kg m-2
            M[:, :, 1] = L0W  # woody litter branches and coarse roots, kg m-2

        M = _decompose(params, cc, k1, k2, k3, k4, k5, k6, k7, k8, k9, M)
        if track_daily:
            daily_cumulative_out[:, day] = M[0, :, 10]
        i += 1

    out = M[0, :, 10] * 10000.0 - previous_mass
    P2_out = P2_ini - M[0, :, 8] * 10000.0
    P3_out = P3_ini - M[0, :, 9] * 10000.0
    next_previous_mass = M[0, :, 10] * 10000.0

    out_root_lyr = out - P2_out - P3_out
    out_below_root_lyr = P2_out  # + self.P3_out

    new_state = State(M=M, i=i, previous_mass=next_previous_mass, pH=pH)
    outputs = YearOutputs(
        out=out,
        out_root_lyr=out_root_lyr,
        out_below_root_lyr=out_below_root_lyr,
        nonwoodylitter=inputs.nonwoodylitter,
        woodylitter=inputs.woodylitter,
        daily_cumulative_out=daily_cumulative_out if track_daily else None,
    )
    return new_state, outputs


def compose_export(
    daily_cumulative_out: Float[np.ndarray, "n days"],
    diag: ResidenceTimeOutput,
    peat_T: Float[np.ndarray, " days"],
    n: int,
) -> DOCExportOutputs:
    # To get total export, sum the left and right ditches
    mass_to_c = 0.5
    doc = np.zeros(n)
    for y in range(n):
        doc[y] = (
            np.sum(
                0.066
                * np.exp(-0.061 * peat_T)
                * np.gradient(daily_cumulative_out[y, :])
            )
            * 10000
            * mass_to_c
        )

    lmwtohmwshare = 0.04
    hmw = (1 - lmwtohmwshare) * doc
    lmw = lmwtohmwshare * doc

    hmwtoditch = hmw * np.exp(
        -0.0004 * diag.residence_time
    )  # biodegradation parameters from Kalbiz et al 2003
    lmwtoditch = lmw * np.exp(-0.15 * diag.residence_time)

    hmw_to_west = (
        len(np.ravel(diag.ixwest)) / diag.n * np.mean(hmwtoditch[np.ravel(diag.ixwest)])
    )
    hmw_to_east = (
        len(np.ravel(diag.ixeast)) / diag.n * np.mean(hmwtoditch[np.ravel(diag.ixeast)])
    )
    lmw_to_west = (
        len(np.ravel(diag.ixwest)) / diag.n * np.mean(lmwtoditch[np.ravel(diag.ixwest)])
    )
    lmw_to_east = (
        len(np.ravel(diag.ixeast)) / diag.n * np.mean(lmwtoditch[np.ravel(diag.ixeast)])
    )

    return DOCExportOutputs(
        hmw=hmw,
        lmw=lmw,
        hmwtoditch=hmwtoditch,
        lmwtoditch=lmwtoditch,
        hmw_to_west=hmw_to_west,
        hmw_to_east=hmw_to_east,
        lmw_to_west=lmw_to_west,
        lmw_to_east=lmw_to_east,
    )
