# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:37:43 2022

@author: alauren
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from supersusi.core.allometry import (
    AllometryFunctions,
    Params as AllometryParams,
    build_allometry_interpolation_functions,
)
from supersusi.core.susi_utils import assimilation_yr


@dataclass(frozen=True)
class Params:
    name: str = field(doc="'dominant' / 'subdominant' / 'under'")
    ncols: int
    nlyrs: np.ndarray = field(doc="unique allometry zone IDs in this layer")
    sfc: np.ndarray = field(doc="site fertility class per column")


@dataclass(frozen=True)
class ComputedConstants:
    allodic: dict[int, AllometryFunctions] = field(
        doc="per-zone allometry, built via build_allometry_interpolation_functions"
    )
    ixs: dict[int, np.ndarray] = field(doc="column indices per zone")
    tree_species: np.ndarray = field(doc="1=Pine, 2=Spruce, 3=Birch")


@dataclass(frozen=True)
class State:
    agearr: np.ndarray = field(doc="years")
    biomass: np.ndarray = field(doc="kg/tree")
    remaining_share: np.ndarray = field(doc="thinning fraction 0..1")
    leafmass: np.ndarray = field(doc="pre-growth leaf mass kg/tree (age-based at init, biomass-based after year 1)")


@dataclass(frozen=True)
class Outputs:
    stems: np.ndarray = field(doc="trees/ha")
    basalarea: np.ndarray = field(doc="m2/tree")
    hdom: np.ndarray = field(doc="dominant height, m")
    Dg: np.ndarray = field(doc="mean diameter, cm")
    volume: np.ndarray = field(doc="m3/tree")
    leafarea: np.ndarray = field(doc="one-sided, m2/m2 per tree")
    leafmass: np.ndarray = field(doc="kg/tree")
    volumegrowth: np.ndarray = field(doc="m3/tree/yr")
    logvolume: np.ndarray
    pulpvolume: np.ndarray
    yi: np.ndarray
    NPP: np.ndarray = field(doc="kg/tree/yr")
    NPP_pot: np.ndarray
    new_lmass: np.ndarray
    leaf_litter: np.ndarray
    C_consumption: np.ndarray
    Nleafdemand: np.ndarray
    Nleaf_litter: np.ndarray
    N_leaf: np.ndarray
    Pleafdemand: np.ndarray
    Pleaf_litter: np.ndarray
    P_leaf: np.ndarray
    Kleafdemand: np.ndarray
    Kleaf_litter: np.ndarray
    K_leaf: np.ndarray
    finerootlitter: np.ndarray
    n_finerootlitter: np.ndarray
    p_finerootlitter: np.ndarray
    k_finerootlitter: np.ndarray
    nonwoodylitter: np.ndarray
    n_nonwoodylitter: np.ndarray
    p_nonwoodylitter: np.ndarray
    k_nonwoodylitter: np.ndarray
    woodylitter: np.ndarray
    n_woodylitter: np.ndarray
    p_woodylitter: np.ndarray
    k_woodylitter: np.ndarray
    woody_litter_mort: np.ndarray
    n_woody_litter_mort: np.ndarray
    p_woody_litter_mort: np.ndarray
    k_woody_litter_mort: np.ndarray
    non_woody_litter_mort: np.ndarray
    n_non_woody_litter_mort: np.ndarray
    p_non_woody_litter_mort: np.ndarray
    k_non_woody_litter_mort: np.ndarray
    n_demand: np.ndarray
    p_demand: np.ndarray
    k_demand: np.ndarray
    basNdemand: np.ndarray
    basPdemand: np.ndarray
    basKdemand: np.ndarray
    leafmax: np.ndarray
    leafmin: np.ndarray


@dataclass(frozen=True)
class CuttingOutputs:
    harvested_volume: np.ndarray
    harvested_log_volume: np.ndarray
    harvested_pulp_volume: np.ndarray
    harvested_biomass: np.ndarray
    harvested_stems: np.ndarray
    nonwoody_lresid: np.ndarray
    n_nonwoody_lresid: np.ndarray
    p_nonwoody_lresid: np.ndarray
    k_nonwoody_lresid: np.ndarray
    woody_lresid: np.ndarray
    n_woody_lresid: np.ndarray
    p_woody_lresid: np.ndarray
    k_woody_lresid: np.ndarray


@dataclass(frozen=True)
class LeafDynamicsOutputs:
    new_lmass: np.ndarray
    leaf_litter: np.ndarray
    C_consumption: np.ndarray
    Nleafdemand: np.ndarray
    Nleaf_litter: np.ndarray
    N_leaf: np.ndarray
    Pleafdemand: np.ndarray
    Pleaf_litter: np.ndarray
    P_leaf: np.ndarray
    Kleafdemand: np.ndarray
    Kleaf_litter: np.ndarray
    K_leaf: np.ndarray
    leafmax: np.ndarray
    leafmin: np.ndarray


@dataclass(frozen=True)
class Inputs:
    photopara: object
    forc: pd.DataFrame = field(doc="annual weather")
    wt: pd.DataFrame = field(doc="annual water table")
    afp: pd.DataFrame = field(doc="annual air-filled porosity")
    previous_nut_stat: np.ndarray
    nut_stat: np.ndarray
    lai_above: np.ndarray


def _leaf_dynamics(
    bm,
    bm_increment,
    current_leafmass,
    previous_nut_stat,
    nut_stat,
    agenow,
    allometry_funcs,
    species,
    printOpt=False,
) -> LeafDynamicsOutputs:
    """
    input:
         bm, current biomass, array, kg/tree without leaves
         bm_increment, array, total NPP kg/tree
         incoming columns are of single tree species canopy layers
    returns LeafDynamicsOutputs (new_lmass, leaf_litter, C_consumption,
            N/P/K demand/litter/content)
    """
    # ******** Parameters *****************
    nuts = {
        "Pine": {"Foliage": {"N": [10.0, 20.0], "P": [1.0, 2.2], "K": [5.0, 6.5]}},
        "Spruce": {
            "Foliage": {"N": [10.0, 20.0], "P": [1.0, 2.2], "K": [3.5, 6.0]}
        },
        "Birch": {"Foliage": {"N": [10.0, 20.0], "P": [1.0, 2.2], "K": [3.5, 6.0]}},
    }

    retrans = {"N": 0.69, "P": 0.73, "K": 0.8}  # Nieminen Helmisaari 1996 Tree Phys
    species_codes = {1: "Pine", 2: "Spruce", 3: "Birch"}
    spe = species_codes[species[0]]  # Take the first item, all are same here
    longevityLeaves = {
        "Pine": [5.0, 3.5],
        "Spruce": [8.0, 5.0],
        "Birch": [1.0, 1.0],
    }  # yrs, life span of leaves and fine roots

    N_con = interp1d(
        np.array([0.66, 1.5]),
        np.array(nuts[spe]["Foliage"]["N"]),
        fill_value=tuple(nuts[spe]["Foliage"]["N"]),
        bounds_error=False,
    )
    P_con = interp1d(
        np.array([0.66, 1.5]),
        np.array(nuts[spe]["Foliage"]["P"]),
        fill_value=tuple(nuts[spe]["Foliage"]["P"]),
        bounds_error=False,
    )
    K_con = interp1d(
        np.array([0.66, 1.5]),
        np.array(nuts[spe]["Foliage"]["K"]),
        fill_value=tuple(nuts[spe]["Foliage"]["K"]),
        bounds_error=False,
    )

    longevity = interp1d(
        np.array([0.66, 1.5]),
        np.array(longevityLeaves[spe]),
        fill_value=tuple(longevityLeaves[spe]),
        bounds_error=False,
    )

    n = len(nut_stat)
    # ************ Biomass and litter**************
    """ Change units to /tree here"""
    leafbase0 = allometry_funcs.biomass_to_stand.leaf_mass(
        bm
    )  # table growth leaf mass in the beginning of timestep
    leafbase1 = allometry_funcs.biomass_to_stand.leaf_mass(
        bm + bm_increment
    )  # table growth leaf mass in the end of timestep

    # -------------------------------------------------
    leafmax = leafbase1 * 1.5
    leafmin = leafbase1 / 1.5

    net_ch = leafbase1 - leafbase0
    gr_demand = leafbase1 * nut_stat - current_leafmass  # actual leaf mass - current

    max_ch = (leafmax - leafmin) / (2 * longevity(nut_stat))

    allowed_ch = np.zeros(n, dtype=float)
    lmass_ch = np.zeros(n, dtype=float)

    ix1 = np.where(gr_demand >= 0)
    ix0 = np.where(gr_demand < 0)
    allowed_ch[ix1] = net_ch[ix1] + max_ch[ix1]
    allowed_ch[ix0] = net_ch[ix0] - max_ch[ix0]
    lmass_ch[ix1] = np.minimum(allowed_ch[ix1], gr_demand[ix1])
    lmass_ch[ix0] = np.maximum(allowed_ch[ix0], gr_demand[ix0])

    new_lmass = current_leafmass + lmass_ch
    new_lmass = np.minimum(leafmax, new_lmass)
    new_lmass = np.maximum(leafmin, new_lmass)
    leaf_litter = new_lmass / longevity(nut_stat)
    C_consumption = lmass_ch + leaf_litter

    N_net = np.maximum(
        np.zeros(n),
        N_con(nut_stat) / 1000.0 * new_lmass
        - N_con(previous_nut_stat) / 1000.0 * current_leafmass,
    )
    Nleaf_litter = leaf_litter * (1 - retrans["N"]) * N_con(nut_stat) / 1000.0
    Ndemand = N_net + Nleaf_litter
    N_leaf = N_con(nut_stat) / 1000.0 * new_lmass

    P_net = np.maximum(
        np.zeros(n),
        P_con(nut_stat) / 1000.0 * new_lmass
        - P_con(previous_nut_stat) / 1000.0 * current_leafmass,
    )
    Pleaf_litter = leaf_litter * (1 - retrans["P"]) * P_con(nut_stat) / 1000.0
    Pdemand = P_net + Pleaf_litter
    P_leaf = P_con(nut_stat) / 1000.0 * new_lmass

    K_net = np.maximum(
        np.zeros(n),
        K_con(nut_stat) / 1000.0 * new_lmass
        - K_con(previous_nut_stat) / 1000.0 * current_leafmass,
    )
    Kleaf_litter = leaf_litter * (1 - retrans["K"]) * K_con(nut_stat) / 1000.0
    Kdemand = K_net + Kleaf_litter
    K_leaf = K_con(nut_stat) / 1000.0 * new_lmass

    if printOpt:
        print("********************************************")
        print("bm", bm)
        print("bm_increment", bm_increment)
        print("nutstat ", nut_stat)
        print("leafmin", leafmin)
        print("leafbase0", leafbase0)
        print("leafbase1", leafbase1)
        print("leafmax", leafmax)
        print("net_change", net_ch)
        print("demanded growth", gr_demand)
        print("max_change", max_ch)
        print("allowed change", allowed_ch)
        print("leaf mass change", lmass_ch)
        print("new leaf mass", new_lmass)
        print("leaf_litter", leaf_litter)
        print("basic consumption", C_consumption)
        print("Ndemand ", N_net + Nleaf_litter)
        print("Nlitter", Nleaf_litter)
        print("Nnet", N_net, N_con(nut_stat), N_con(previous_nut_stat))
        print("Pdemand ", P_net + Pleaf_litter)
        print("Plitter", Pleaf_litter)
        print("Pnet", P_net, P_con(nut_stat), P_con(previous_nut_stat))
        print("Kdemand ", K_net + Kleaf_litter)
        print("Klitter", Kleaf_litter)
        print("Knet", K_net, K_con(nut_stat), K_con(previous_nut_stat))
        print("*******************************************")
        print("nitrogen content,", N_con(nut_stat))
        print("leaf longevity", longevity(nut_stat))

    return LeafDynamicsOutputs(
        new_lmass=new_lmass,
        leaf_litter=leaf_litter,
        C_consumption=C_consumption,
        Nleafdemand=Ndemand,
        Nleaf_litter=Nleaf_litter,
        N_leaf=N_leaf,
        Pleafdemand=Pdemand,
        Pleaf_litter=Pleaf_litter,
        P_leaf=P_leaf,
        Kleafdemand=Kdemand,
        Kleaf_litter=Kleaf_litter,
        K_leaf=K_leaf,
        leafmax=leafmax,
        leafmin=leafmin,
    )


def apply_allometry(
    biomass: np.ndarray,
    agearr: np.ndarray,
    remaining_share: np.ndarray,
    cc: ComputedConstants,
) -> Outputs:
    """Map core state (biomass, agearr, remaining_share) → all derived allometric outputs.

    Pure function. Growth fields (NPP, leaf_litter, …) are set to 0.
    nonwoodylitter = finerootlitter (leaf_litter not available here; grow_stand overrides).
    """
    ncols = len(biomass)
    z = np.zeros

    # Growth fields — all zero
    growth_zeros = dict(
        NPP=z(ncols),
        NPP_pot=z(ncols),
        new_lmass=z(ncols),
        leaf_litter=z(ncols),
        C_consumption=z(ncols),
        Nleafdemand=z(ncols),
        Nleaf_litter=z(ncols),
        N_leaf=z(ncols),
        Pleafdemand=z(ncols),
        Pleaf_litter=z(ncols),
        P_leaf=z(ncols),
        Kleafdemand=z(ncols),
        Kleaf_litter=z(ncols),
        K_leaf=z(ncols),
        volumegrowth=z(ncols),
        leafmax=z(ncols),
        leafmin=z(ncols),
    )

    # Allometric fields — will be filled per zone
    stems = z(ncols)
    basalarea = z(ncols)
    hdom = z(ncols)
    Dg = z(ncols)
    volume = z(ncols)
    leafarea = z(ncols)
    leafmass = z(ncols)
    logvolume = z(ncols)
    pulpvolume = z(ncols)
    yi = z(ncols)
    finerootlitter = z(ncols)
    n_finerootlitter = z(ncols)
    p_finerootlitter = z(ncols)
    k_finerootlitter = z(ncols)
    woodylitter = z(ncols)
    n_woodylitter = z(ncols)
    p_woodylitter = z(ncols)
    k_woodylitter = z(ncols)
    woody_litter_mort = z(ncols)
    n_woody_litter_mort = z(ncols)
    p_woody_litter_mort = z(ncols)
    k_woody_litter_mort = z(ncols)
    non_woody_litter_mort = z(ncols)
    n_non_woody_litter_mort = z(ncols)
    p_non_woody_litter_mort = z(ncols)
    k_non_woody_litter_mort = z(ncols)
    n_demand = z(ncols)
    p_demand = z(ncols)
    k_demand = z(ncols)
    basNdemand = z(ncols)
    basPdemand = z(ncols)
    basKdemand = z(ncols)

    for zone_id, af in cc.allodic.items():
        ixs = cc.ixs[zone_id]
        bm = biomass[ixs]
        rs = remaining_share[ixs]

        bts = af.biomass_to_stand
        nd = af.nutrient_demand
        lm = af.litter_mass
        nl = af.nutrient_litter
        mm = af.mortality_mass
        nm = af.nutrient_mortality
        yv = af.yield_volume

        stems[ixs] = bts.stems(bm) * rs
        basalarea[ixs] = bts.ba(bm)
        hdom[ixs] = bts.hdom(bm)
        Dg[ixs] = bts.dg(bm)
        leafarea[ixs] = bts.lai(bm)
        leafmass[ixs] = bts.leaf_mass(bm)
        volume[ixs] = bts.vol(bm)
        yi[ixs] = bts.yi(bm)

        logvolume[ixs] = yv.vol_to_logs(volume[ixs])
        pulpvolume[ixs] = yv.vol_to_pulp(volume[ixs])

        n_demand[ixs] = nd.n_demand(bm)
        p_demand[ixs] = nd.p_demand(bm)
        k_demand[ixs] = nd.k_demand(bm)
        basNdemand[ixs] = nd.n_leaf_demand(bm)
        basPdemand[ixs] = nd.p_leaf_demand(bm)
        basKdemand[ixs] = nd.k_leaf_demand(bm)

        finerootlitter[ixs] = lm.fine_root_litter(bm)
        n_finerootlitter[ixs] = nl.n_fine_root_litter(bm)
        p_finerootlitter[ixs] = nl.p_fine_root_litter(bm)
        k_finerootlitter[ixs] = nl.k_fine_root_litter(bm)

        woodylitter[ixs] = lm.woody_litter(bm)
        n_woodylitter[ixs] = nl.n_woody_litter(bm)
        p_woodylitter[ixs] = nl.p_woody_litter(bm)
        k_woodylitter[ixs] = nl.k_woody_litter(bm)

        woody_litter_mort[ixs] = mm.woody(bm)
        n_woody_litter_mort[ixs] = nm.n_mortality_woody(bm)
        p_woody_litter_mort[ixs] = nm.p_mortality_woody(bm)
        k_woody_litter_mort[ixs] = nm.k_mortality_woody(bm)

        # BUG: The + leaves term below is a NO-OP because it sits on the next line,
        #      outside the implicit bracket continuation. Python treats it as a
        #      separate expression statement (+leaves) whose result is discarded.
        #      This means stand/non_woody_litter_mort OMITS the leaves mortality
        #      component, including only fine-root mortality.
        #      See OOP src/susi/core/canopylayer.py lines 419-422.
        non_woody_litter_mort[ixs] = mm.fine_root(bm)
        + mm.leaves(bm)
        # BUG: Same pattern for N, P, K — the + term on the next line is a no-op.
        n_non_woody_litter_mort[ixs] = nm.n_mortality_fine_root(bm)
        + nm.n_mortality_leaves(bm)
        p_non_woody_litter_mort[ixs] = nm.p_mortality_fine_root(bm)
        + nm.p_mortality_leaves(bm)
        k_non_woody_litter_mort[ixs] = nm.k_mortality_fine_root(bm)
        + nm.k_mortality_leaves(bm)

    return Outputs(
        stems=stems,
        basalarea=basalarea,
        hdom=hdom,
        Dg=Dg,
        volume=volume,
        leafarea=leafarea,
        leafmass=leafmass,
        logvolume=logvolume,
        pulpvolume=pulpvolume,
        yi=yi,
        finerootlitter=finerootlitter,
        n_finerootlitter=n_finerootlitter,
        p_finerootlitter=p_finerootlitter,
        k_finerootlitter=k_finerootlitter,
        nonwoodylitter=finerootlitter,
        n_nonwoodylitter=n_finerootlitter,
        p_nonwoodylitter=p_finerootlitter,
        k_nonwoodylitter=k_finerootlitter,
        woodylitter=woodylitter,
        n_woodylitter=n_woodylitter,
        p_woodylitter=p_woodylitter,
        k_woodylitter=k_woodylitter,
        woody_litter_mort=woody_litter_mort,
        n_woody_litter_mort=n_woody_litter_mort,
        p_woody_litter_mort=p_woody_litter_mort,
        k_woody_litter_mort=k_woody_litter_mort,
        non_woody_litter_mort=non_woody_litter_mort,
        n_non_woody_litter_mort=n_non_woody_litter_mort,
        p_non_woody_litter_mort=p_non_woody_litter_mort,
        k_non_woody_litter_mort=k_non_woody_litter_mort,
        n_demand=n_demand,
        p_demand=p_demand,
        k_demand=k_demand,
        basNdemand=basNdemand,
        basPdemand=basPdemand,
        basKdemand=basKdemand,
    **growth_zeros,
)


def compute_constants(
    params: Params,
    allometry_data: dict[int, pd.DataFrame],
    species_id: dict[int, int],
) -> ComputedConstants:
    """Build per-zone AllometryFunctions and tree_species array.

    For each non-zero zone in params.nlyrs, builds interpolation functions
    using the median site fertility class for that zone.
    """
    ixs: dict[int, np.ndarray] = {}
    for z in params.nlyrs:
        if z > 0:
            ixs[z] = np.where(params.nlyrs == z)[0]

    allodic: dict[int, AllometryFunctions] = {}
    tree_species = np.zeros(params.ncols, dtype=np.int32)
    for z, af_ixs in ixs.items():
        sfc_median = int(np.median(params.sfc[af_ixs]))
        allodic[z] = build_allometry_interpolation_functions(
            AllometryParams(species=species_id[z], site_fertility_class=sfc_median),
            allometry_data[z],
        )
        tree_species[af_ixs] = species_id[z]

    return ComputedConstants(allodic=allodic, ixs=ixs, tree_species=tree_species)


def compute_initial_state(
    params: Params,
    cc: ComputedConstants,
    agearr: np.ndarray,
    nut_stat: np.ndarray,
) -> tuple[State, Outputs]:
    """Initialise State and Outputs matching OOP initialize_domain().

    Starts from age-based biomass (same as apply_allometry), then overrides
    with OOP's mixed formulas: basalarea←ageToBa, hdom←ageToHdom,
    leafmass←ageToLeaves×nut_stat, leafarea←bmToLAI×nut_stat.
    OOP init zeros all litter/mortality/demand fields except those set above.
    """
    biomass = np.zeros(params.ncols)
    for z, ix in cc.ixs.items():
        biomass[ix] = cc.allodic[z].age_based.bm(agearr[ix])

    out = apply_allometry(biomass, agearr, np.ones(params.ncols), cc)

    # Override fields where OOP init uses different formulas
    basalarea = np.zeros(params.ncols)
    hdom = np.zeros(params.ncols)
    init_leafmass = np.zeros(params.ncols)
    leafarea = np.zeros(params.ncols)
    for z, ix in cc.ixs.items():
        age = agearr[ix]
        ns = nut_stat[ix]
        bm = biomass[ix]
        basalarea[ix] = cc.allodic[z].age_based.ba(age)
        hdom[ix] = cc.allodic[z].age_based.hdom(age)
        init_leafmass[ix] = cc.allodic[z].age_based.leaves(age) * ns
        leafarea[ix] = cc.allodic[z].biomass_to_stand.lai(bm) * ns

    # Zero fields that OOP init leaves at zero
    zero = np.zeros(params.ncols)

    state = State(
        agearr=agearr.copy(), biomass=biomass, remaining_share=np.ones(params.ncols),
        leafmass=init_leafmass.copy(),
    )
    return state, replace(out,
        basalarea=basalarea, hdom=hdom, leafmass=init_leafmass, leafarea=leafarea,
        # Litter
        finerootlitter=zero, n_finerootlitter=zero,
        p_finerootlitter=zero, k_finerootlitter=zero,
        nonwoodylitter=zero, n_nonwoodylitter=zero,
        p_nonwoodylitter=zero, k_nonwoodylitter=zero,
        woodylitter=zero, n_woodylitter=zero,
        p_woodylitter=zero, k_woodylitter=zero,
        woody_litter_mort=zero, n_woody_litter_mort=zero,
        p_woody_litter_mort=zero, k_woody_litter_mort=zero,
        non_woody_litter_mort=zero, n_non_woody_litter_mort=zero,
        p_non_woody_litter_mort=zero, k_non_woody_litter_mort=zero,
        # Growth fields
        NPP=zero, NPP_pot=zero, new_lmass=zero, leaf_litter=zero, C_consumption=zero,
        Nleafdemand=zero, Nleaf_litter=zero, N_leaf=zero,
        Pleafdemand=zero, Pleaf_litter=zero, P_leaf=zero,
        Kleafdemand=zero, Kleaf_litter=zero, K_leaf=zero,
        # Nutrient demands (keep basN/P/Kdemand from apply_allometry)
        n_demand=zero, p_demand=zero, k_demand=zero,
        # Other
        logvolume=zero, pulpvolume=zero, yi=zero, volumegrowth=zero,
        leafmax=zero, leafmin=zero,
    )


def grow_stand(
    state: State,
    cc: ComputedConstants,
    inputs: Inputs,
) -> tuple[State, Outputs]:
    """Annual growth: photosynthesis, leaf dynamics, biomass increment.

    Pure function. Returns (new_state, outputs) where outputs has:
    - growth fields (NPP, leaf_litter, C_consumption, …) filled by leaf dynamics
    - allometric fields (stems, volume, …) at the new biomass
    - nonwoodylitter = finerootlitter(new biomass) + leaf_litter
    """
    ncols = len(state.agearr)
    prev = apply_allometry(state.biomass, state.agearr, state.remaining_share, cc)

    # NPP — assimilation_yr returns kg/ha/yr
    npp_ha, npp_pot_ha = assimilation_yr(
        inputs.photopara,
        inputs.forc,
        inputs.wt,
        inputs.afp,
        prev.leafarea * 2 * prev.stems,
        inputs.lai_above * 2,
    )
    stems_safe = np.maximum(prev.stems, 1e-30)
    npp_per_tree = npp_ha * inputs.nut_stat / stems_safe
    npp_pot_per_tree = npp_pot_ha * inputs.nut_stat / stems_safe
    # BUG: OOP writes nan for NPP/NPP_pot when stems=0 (np.divide with
    #      out=np.nan, where=stems>0). Replicate to match golden file.
    npp_per_tree = np.where(prev.stems > 0, npp_per_tree, np.nan)
    npp_pot_per_tree = np.where(prev.stems > 0, npp_pot_per_tree, np.nan)

    # Per-zone: leaf dynamics + delta calc
    delta = np.zeros(ncols)
    new_lmass = np.zeros(ncols)
    leaf_litter = np.zeros(ncols)
    c_consumption = np.zeros(ncols)
    n_leaf = np.zeros(ncols)
    p_leaf = np.zeros(ncols)
    k_leaf = np.zeros(ncols)
    nleafdemand = np.zeros(ncols)
    pleafdemand = np.zeros(ncols)
    kleafdemand = np.zeros(ncols)
    nleaf_litter = np.zeros(ncols)
    pleaf_litter = np.zeros(ncols)
    kleaf_litter = np.zeros(ncols)
    leafmax = np.zeros(ncols)
    leafmin = np.zeros(ncols)

    for zid, ix in cc.ixs.items():
        af = cc.allodic[zid]
        bm_z = state.biomass[ix]
        fr_lit = af.litter_mass.fine_root_litter(bm_z)
        wd_lit = af.litter_mass.woody_litter(bm_z)

        ld = _leaf_dynamics(
            bm_z,
            npp_per_tree[ix],
            state.leafmass[ix],
            inputs.previous_nut_stat[ix],
            inputs.nut_stat[ix],
            state.agearr[ix],
            af,
            cc.tree_species[ix],
        )

        new_lmass[ix] = ld.new_lmass
        leaf_litter[ix] = ld.leaf_litter
        c_consumption[ix] = ld.C_consumption
        nleafdemand[ix] = ld.Nleafdemand
        nleaf_litter[ix] = ld.Nleaf_litter
        n_leaf[ix] = ld.N_leaf
        pleafdemand[ix] = ld.Pleafdemand
        pleaf_litter[ix] = ld.Pleaf_litter
        p_leaf[ix] = ld.P_leaf
        kleafdemand[ix] = ld.Kleafdemand
        kleaf_litter[ix] = ld.Kleaf_litter
        k_leaf[ix] = ld.K_leaf
        leafmax[ix] = ld.leafmax
        leafmin[ix] = ld.leafmin

        delta[ix] = npp_per_tree[ix] - ld.C_consumption - fr_lit - wd_lit

    new_biomass = state.biomass + np.maximum(delta, 0.0)
    new_agearr = state.agearr + 1

    out = apply_allometry(new_biomass, new_agearr, state.remaining_share, cc)

    # OOP's self.leafmass = self.new_lmass aliases leafmass→new_lmass, then
    # self.update() overwrites both with allometry-based leafmass.
    # Match this by using out.leafmass for new_lmass.
    merged = replace(
        out,
        NPP=npp_per_tree,
        NPP_pot=npp_pot_per_tree,
        new_lmass=out.leafmass,
        leaf_litter=leaf_litter,
        C_consumption=c_consumption,
        Nleafdemand=nleafdemand,
        Nleaf_litter=nleaf_litter,
        N_leaf=n_leaf,
        Pleafdemand=pleafdemand,
        Pleaf_litter=pleaf_litter,
        P_leaf=p_leaf,
        Kleafdemand=kleafdemand,
        Kleaf_litter=kleaf_litter,
        K_leaf=k_leaf,
        leafmass=out.leafmass,
        volumegrowth=out.volume - prev.volume,
        nonwoodylitter=out.finerootlitter + leaf_litter,
        n_nonwoodylitter=out.n_finerootlitter + nleaf_litter,
        p_nonwoodylitter=out.p_finerootlitter + pleaf_litter,
        k_nonwoodylitter=out.k_finerootlitter + kleaf_litter,
        leafmax=leafmax,
        leafmin=leafmin,
    )

    return State(new_agearr, new_biomass, state.remaining_share, out.leafmass), merged


def cut_stand(
    state: State,
    cc: ComputedConstants,
    out: Outputs,
    nut_stat: np.ndarray,
    to_ba: float = 0.5,
) -> tuple[State, CuttingOutputs]:
    """Thinning (to_ba >= 1) or clear-cut (to_ba < 1).

    Thinning: retain ``to_ba`` m²/ha of basal area, update remaining_share.
    Clear-cut: reset age to 1, re-initialise state from allometry.
    Logging residues and harvested volumes are per-hectare (× cut stems).

    Returns (new_state, cutting_outputs).
    """
    ncols = len(state.agearr)
    z = np.zeros

    wood_density_map = {1: 420.0, 2: 400.0, 3: 450.0}
    wood_density = np.array([wood_density_map[s] for s in cc.tree_species])

    harvested_volume = z(ncols)
    harvested_log_volume = z(ncols)
    harvested_pulp_volume = z(ncols)
    harvested_biomass = z(ncols)
    harvested_stems = z(ncols)
    nonwoody_lresid = z(ncols)
    n_nonwoody_lresid = z(ncols)
    p_nonwoody_lresid = z(ncols)
    k_nonwoody_lresid = z(ncols)
    woody_lresid = z(ncols)
    n_woody_lresid = z(ncols)
    p_woody_lresid = z(ncols)
    k_woody_lresid = z(ncols)

    new_remaining_share = state.remaining_share.copy()
    new_agearr = state.agearr.copy()

    if to_ba >= 1.0:
        # Thinning
        for zid, ix in cc.ixs.items():
            af = cc.allodic[zid]
            bm_z = state.biomass[ix]
            ba_per_ha = out.basalarea[ix] * out.stems[ix]
            remaining_share_z = to_ba / ba_per_ha
            cut_stems = (1.0 - remaining_share_z) * out.stems[ix]

            new_remaining_share[ix] = remaining_share_z

            harvested_volume[ix] = af.biomass_to_stand.vol(bm_z) * cut_stems
            harvested_log_volume[ix] = af.biomass_to_stand.log_vol(bm_z) * cut_stems
            harvested_pulp_volume[ix] = af.biomass_to_stand.pulp_vol(bm_z) * cut_stems
            harvested_biomass[ix] = (
                af.biomass_to_stand.log_vol(bm_z) + af.biomass_to_stand.pulp_vol(bm_z)
            ) * cut_stems * wood_density[ix]
            harvested_stems[ix] = cut_stems

            nonwoody_lresid[ix] = (out.new_lmass[ix] + af.fine_roots.fine_roots(bm_z)) * cut_stems
            n_nonwoody_lresid[ix] = (out.N_leaf[ix] + af.fine_roots.n_fine_roots(bm_z)) * cut_stems
            p_nonwoody_lresid[ix] = (out.P_leaf[ix] + af.fine_roots.p_fine_roots(bm_z)) * cut_stems
            k_nonwoody_lresid[ix] = (out.K_leaf[ix] + af.fine_roots.k_fine_roots(bm_z)) * cut_stems

            woody_lresid[ix] = af.logging_residues.woody(bm_z) * cut_stems
            n_woody_lresid[ix] = af.logging_residues.n_woody(bm_z) * cut_stems
            p_woody_lresid[ix] = af.logging_residues.p_woody(bm_z) * cut_stems
            k_woody_lresid[ix] = af.logging_residues.k_woody(bm_z) * cut_stems

        new_state = State(new_agearr, state.biomass, new_remaining_share, state.leafmass)
    else:
        # Clear-cut: remove all stems, reset age to 1, re-init
        for zid, ix in cc.ixs.items():
            af = cc.allodic[zid]
            bm_z = state.biomass[ix]
            cut_stems = out.stems[ix]
            new_remaining_share[ix] = 1.0
            new_agearr[ix] = 1.0

            harvested_volume[ix] = af.biomass_to_stand.vol(bm_z) * cut_stems
            harvested_log_volume[ix] = af.biomass_to_stand.log_vol(bm_z) * cut_stems
            harvested_pulp_volume[ix] = af.biomass_to_stand.pulp_vol(bm_z) * cut_stems
            harvested_biomass[ix] = (
                af.biomass_to_stand.log_vol(bm_z) + af.biomass_to_stand.pulp_vol(bm_z)
            ) * cut_stems * wood_density[ix]
            harvested_stems[ix] = cut_stems

            nonwoody_lresid[ix] = (out.new_lmass[ix] + af.fine_roots.fine_roots(bm_z)) * cut_stems
            n_nonwoody_lresid[ix] = (out.N_leaf[ix] + af.fine_roots.n_fine_roots(bm_z)) * cut_stems
            p_nonwoody_lresid[ix] = (out.P_leaf[ix] + af.fine_roots.p_fine_roots(bm_z)) * cut_stems
            k_nonwoody_lresid[ix] = (out.K_leaf[ix] + af.fine_roots.k_fine_roots(bm_z)) * cut_stems

            woody_lresid[ix] = af.logging_residues.woody(bm_z) * cut_stems
            n_woody_lresid[ix] = af.logging_residues.n_woody(bm_z) * cut_stems
            p_woody_lresid[ix] = af.logging_residues.p_woody(bm_z) * cut_stems
            k_woody_lresid[ix] = af.logging_residues.k_woody(bm_z) * cut_stems

        # Re-initialise biomass from age
        new_biomass = z(ncols)
        new_leafmass = z(ncols)
        for zid, ix in cc.ixs.items():
            new_biomass[ix] = cc.allodic[zid].age_based.bm(new_agearr[ix])
            new_leafmass[ix] = cc.allodic[zid].age_based.leaves(new_agearr[ix]) * nut_stat[ix]

        new_state = State(new_agearr, new_biomass, new_remaining_share, new_leafmass)

    cutting_out = CuttingOutputs(
        harvested_volume=harvested_volume,
        harvested_log_volume=harvested_log_volume,
        harvested_pulp_volume=harvested_pulp_volume,
        harvested_biomass=harvested_biomass,
        harvested_stems=harvested_stems,
        nonwoody_lresid=nonwoody_lresid,
        n_nonwoody_lresid=n_nonwoody_lresid,
        p_nonwoody_lresid=p_nonwoody_lresid,
        k_nonwoody_lresid=k_nonwoody_lresid,
        woody_lresid=woody_lresid,
        n_woody_lresid=n_woody_lresid,
        p_woody_lresid=p_woody_lresid,
        k_woody_lresid=k_woody_lresid,
    )

    return new_state, cutting_out


class Canopylayer:
    """
    UNITS: all units in /tree basis, except number of trees in the canopy layer, which is in /ha
    Canopylayer keeps track on the development of biomass components within the
    different layers of canopy. Canopy layer is of single tree species and homegeneous in age. It is
    initialized with growth and yield simulator outputfile that describes development
    of biomass components in time. Canopy layers is array shaped with dimensions
    number of columns along the strip.
    """

    def __init__(
        self,
        name,
        nscens,
        yrs,
        ncols,
        nlyrs,
        sfc,
        agearr,
        allometry_df_dict,
        species_id_dict,
        ixs,
        photopara,
        nut_stat,
    ):
        self.name = (
            name  # name of the canopy layer e.g. 'dominant', 'subdominant', etc.
        )
        self.nlyrs = nlyrs  # number of different canopy layers along the strip
        self.ixs = ixs  # indices for the location of the different canopy layers along the strip
        self.ncols = ncols  # number of columns in the strip
        self.agearr = agearr.copy()  # age of the canopy layer, yrs
        self.nscens = nscens  # number of scenarion in the simulation
        self.yrs = yrs  # number od years in the simulation
        self.remaining_share = np.ones(
            self.ncols
        )  # share of remaining stems after thinning 0...1
        self.sfc = sfc.copy()
        self.tree_species = np.zeros(
            self.ncols, dtype=np.int8
        )  # tree species 1 Scots pine, 2 Norway spruce, 3 Deciduous

        # -------- Biomass interpolation functions-------------------
        self.allodic = {}  # dictionary to contain all allometric functions
        for ncanopy in (
            nlyrs
        ):  # numner of different allometric files along the strip in this canopy layer
            if ncanopy > 0:  # zero indicates no tree in the layer
                per_layer_sfc = int(
                    np.median(self.sfc[self.ixs[ncanopy]])
                )  # site fertility class
                self.allodic[ncanopy] = build_allometry_interpolation_functions(
                    AllometryParams(
                        species=species_id_dict[ncanopy],
                        site_fertility_class=per_layer_sfc,
                    ),
                    allometry_df_dict[ncanopy],
                )
                self.tree_species[self.ixs[ncanopy]] = species_id_dict[ncanopy]
        self.initialize_domain(
            agearr, nut_stat
        )  # create variables and set initial values

        print(self.name, "initialized")

    def initialize_domain(self, agearr, nut_stat):
        self.agearr = agearr.copy()
        self.remaining_share = np.ones(
            self.ncols
        )  # share of remaining stems after thinning 0...1, in initialization should be one
        nlyrs = self.nlyrs  # number of different canopy layers along the strip
        ixs = self.ixs  # indices for the canopy layers along the strip

        ncols = self.ncols  # number of columns along the strip
        self.stems = np.zeros(ncols, dtype=float)  # stocking number of trees per ha

        self.basalarea = np.zeros(
            ncols, dtype=float
        )  # basal area in the canopy layer m2/tree
        self.biomass = np.zeros(
            ncols, dtype=float
        )  # total canopy layer biomass excluding leaves kg/tree
        self.n_demand = np.zeros(
            ncols, dtype=float
        )  # N demand exluding leaves kg/tree/yr
        self.p_demand = np.zeros(
            ncols, dtype=float
        )  # P demand exluding leaves kg/tree/yr
        self.k_demand = np.zeros(
            ncols, dtype=float
        )  # K demand exluding leaves kg/tree/yr
        self.hdom = np.zeros(
            ncols, dtype=float
        )  # domainant height m in the canopy layer
        self.Dg = np.zeros(ncols, dtype=float)  # mean diameter cm in the canopy layer
        self.leafarea = np.zeros(
            ncols, dtype=float
        )  # leaf area in the canopy layer m2 m-2 tree-1
        self.lai_above = np.zeros(
            ncols, dtype=float
        )  # one sided leaf area above this canopy layer, m2 m-2 tree-1
        self.leafmass = np.zeros(ncols, dtype=float)  # basic leaf mass kg/tree
        self.logvolume = np.zeros(ncols, dtype=float)  # volume of the saw logs m3/tree
        self.finerootlitter = np.zeros(
            ncols, dtype=float
        )  # fineroot litter in the canopy layer kg/tree/yr
        self.n_finerootlitter = np.zeros(
            ncols, dtype=float
        )  # N in fineroot litter in the canopy layer kg/treea/yr
        self.p_finerootlitter = np.zeros(
            ncols, dtype=float
        )  # P in fineroot litter in the canopy layer kg/tree/yr
        self.k_finerootlitter = np.zeros(
            ncols, dtype=float
        )  # K in fineroot litter in the canopy layer kg/tree/yr
        self.pulpvolume = np.zeros(
            ncols, dtype=float
        )  # volume of pulpwood in the canopy layer m3/tree

        self.NPP = np.zeros(
            ncols, dtype=float
        )  # net primary production kg/tree/yr dry matter
        self.NPP_pot = np.zeros(
            ncols, dtype=float
        )  # potential net primary production kg/tree/yr dry matter

        self.nonwoodylitter = np.zeros(ncols, dtype=float)  # nonwoody litter kg/tree/yr
        self.n_nonwoodylitter = np.zeros(
            ncols, dtype=float
        )  # N in nonwoody litter kg/tree/yr
        self.p_nonwoodylitter = np.zeros(
            ncols, dtype=float
        )  # P in nonwoody litter kg/tree/yr
        self.k_nonwoodylitter = np.zeros(
            ncols, dtype=float
        )  # K in nonwoody litter kg/tree/yr
        self.species = np.zeros(
            ncols, dtype=int
        )  # tree species: 1 Scots pine, 2: Norway spruce, 3: Betula pendula
        self.volume = np.zeros(
            ncols, dtype=float
        )  # volume of the growing stock m3/tree
        self.volumegrowth = np.zeros(
            ncols, dtype=float
        )  # volume growth of the growing stock m3/tree/yr
        self.woodylitter = np.zeros(ncols, dtype=float)  # woody litter kg/tree/yr
        self.n_woodylitter = np.zeros(
            ncols, dtype=float
        )  # N in woody litter kg/tree/yr
        self.p_woodylitter = np.zeros(
            ncols, dtype=float
        )  # P in woody litter kg/tree/yr
        self.k_woodylitter = np.zeros(
            ncols, dtype=float
        )  # K in woody litter kg/tree/yr
        self.yi = np.zeros(ncols, dtype=float)  # yied. here same as voolume

        self.harvested_volume = np.zeros(ncols, dtype=float)  # harvested volume m3/tree
        self.harvested_log_volume = np.zeros(
            ncols, dtype=float
        )  # harvested saw log volume m3/tree
        self.harvested_pulp_volume = np.zeros(
            ncols, dtype=float
        )  # harvsted pulp volume m3/tree
        self.harvested_biomass = np.zeros(ncols, dtype=float)  # saw biomass kg/tree
        self.harvested_stems = np.zeros(
            ncols, dtype=float
        )  # number of harvested stems/tree

        self.nonwoody_lresid = np.zeros(
            ncols, dtype=float
        )  # nonwoody logging residues kg/tree
        self.n_nonwoody_lresid = np.zeros(
            ncols, dtype=float
        )  # N in nonwoody logging residues kg/tree
        self.p_nonwoody_lresid = np.zeros(
            ncols, dtype=float
        )  # P in nonwoody logging residues kg/tree
        self.k_nonwoody_lresid = np.zeros(
            ncols, dtype=float
        )  # K in nonwoody logging residues kg/tree

        self.woody_lresid = np.zeros(
            ncols, dtype=float
        )  # woody logging residues kg/tree
        self.n_woody_lresid = np.zeros(
            ncols, dtype=float
        )  # N in woody logging residues kg/tree
        self.p_woody_lresid = np.zeros(
            ncols, dtype=float
        )  # P in woody logging residues kg/tree
        self.k_woody_lresid = np.zeros(
            ncols, dtype=float
        )  # K in woody logging residues kg/tree

        self.woody_litter_mort = np.zeros(
            ncols, dtype=float
        )  # woody litter from mortality kg/tree
        self.n_woody_litter_mort = np.zeros(
            ncols, dtype=float
        )  # N in woody litter from mortality kg/tree
        self.p_woody_litter_mort = np.zeros(
            ncols, dtype=float
        )  # P in woody litter from mortality kg/tree
        self.k_woody_litter_mort = np.zeros(
            ncols, dtype=float
        )  # K in woody litter from mortality kg/tree

        self.non_woody_litter_mort = np.zeros(
            ncols, dtype=float
        )  # non-woody litter from mortality kg/tree
        self.n_non_woody_litter_mort = np.zeros(
            ncols, dtype=float
        )  # N in non-woody litter from mortality kg/tree
        self.p_non_woody_litter_mort = np.zeros(
            ncols, dtype=float
        )  # P in non-woody litter from mortality kg/tree
        self.k_non_woody_litter_mort = np.zeros(
            ncols, dtype=float
        )  # K in non-woody litter from mortality kg/tree

        # ----------- ARRANGE Leaf dynamics variables-----------------
        self.new_lmass = np.zeros(
            ncols, dtype=float
        )  # leaf mass after dynamics computation kg/ha
        self.leaf_litter = np.zeros(ncols, dtype=float)  # leaf litter kg/tree/yr
        self.C_consumption = np.zeros(
            ncols, dtype=float
        )  # check the unit: C or mass? C consumption in leaf processes
        self.leafmax = np.zeros(ncols, dtype=float)  # upper limit for leaf mass kg/tree
        self.leafmin = np.zeros(ncols, dtype=float)  # lower limit for leaf mass kg/tree
        self.Nleafdemand = np.zeros(
            ncols, dtype=float
        )  # N demand of leaf production kg/ha/tree
        self.Nleaf_litter = np.zeros(ncols, dtype=float)  # N in litterfall kg/ha/tree
        self.N_leaf = np.zeros(ncols, dtype=float)  # N in leaves kg/tree
        self.Pleafdemand = np.zeros(
            ncols, dtype=float
        )  # P demand of leaf production kg/ha/tree
        self.Pleaf_litter = np.zeros(ncols, dtype=float)  # P in litterfall kg/ha/tree
        self.P_leaf = np.zeros(ncols, dtype=float)  # P in leaves kg/tree
        self.Kleafdemand = np.zeros(
            ncols, dtype=float
        )  # K demand in leaf production kg/ha/tree
        self.Kleaf_litter = np.zeros(ncols, dtype=float)  # K in litterfall kg/ha/tree
        self.K_leaf = np.zeros(ncols, dtype=float)  # K in leaves kg/tree
        self.basNdemand = np.zeros(
            ncols, dtype=float
        )  # basic N demand kg/tree, in table growth conditions, used in nutrient status calculation
        self.basPdemand = np.zeros(
            ncols, dtype=float
        )  # basic P demand kg/tree, in table growth conditions, used in nutrient status calculation
        self.basKdemand = np.zeros(
            ncols, dtype=float
        )  # basic K demand kg/tree, in table growth conditions, used in nutrient status calculation

        # ---------- Initial values from age -------------------------------------------------------
        for m in nlyrs:
            if m > 0:
                self.biomass[ixs[m]] = self.allodic[m].age_based.bm(
                    self.agearr[ixs[m]]
                )  # stem biomass from age kg/tree
                self.stems[ixs[m]] = (
                    self.allodic[m].biomass_to_stand.stems(self.biomass[ixs[m]])
                    * self.remaining_share[ixs[m]]
                )  # number of stems /ha from the biomass of tree

                self.basalarea[ixs[m]] = self.allodic[m].age_based.ba(
                    self.agearr[ixs[m]]
                )  # stem basal area from age m2/tree
                self.hdom[ixs[m]] = self.allodic[m].age_based.hdom(
                    self.agearr[ixs[m]]
                )  # dominant height m
                self.Dg[ixs[m]] = self.allodic[m].biomass_to_stand.dg(
                    self.biomass[ixs[m]]
                )  # mean diameter height m

                self.leafarea[ixs[m]] = (
                    self.allodic[m].biomass_to_stand.lai(self.biomass[ixs[m]])
                    * nut_stat[ixs[m]]
                )  # one sided LAI from the stem biomass M2/m2/tree
                self.leafmass[ixs[m]] = (
                    self.allodic[m].age_based.leaves(self.agearr[ixs[m]])
                    * nut_stat[ixs[m]]
                )
                self.species[ixs[m]] = self.tree_species[ixs[m]]
                # self.volume[ixs[m]] = self.allodic[m].age_based.vol(self.agearr[ixs[m]])         # stem volume m3/tree
                self.volume[ixs[m]] = self.allodic[m].biomass_to_stand.vol(
                    self.biomass[ixs[m]]
                )  # stem volume m3/tree

                self.basNdemand[ixs[m]] = self.allodic[m].nutrient_demand.n_leaf_demand(
                    self.biomass[ixs[m]]
                )
                self.basPdemand[ixs[m]] = self.allodic[m].nutrient_demand.p_leaf_demand(
                    self.biomass[ixs[m]]
                )
                self.basKdemand[ixs[m]] = self.allodic[m].nutrient_demand.k_leaf_demand(
                    self.biomass[ixs[m]]
                )

    def update(self, bm):
        # bm in kg in a mean stem
        """CHANGE all units here into /tree, Do we need remaining share?"""
        # ------------Update all variables with new biomass--------------------------------------
        ixs = self.ixs
        for m in self.nlyrs:
            if m > 0:
                # print ('**********************')
                # print (np.round(np.mean(self.allodic[m].allometry_f['bmToVol'](bm[ixs[m]])*self.stems),2))
                # print (np.round(np.mean(self.allodic[m].allometry_f['ageToVol'](self.agearr[ixs[m]])*self.stems), 2))
                # print ('vol')
                # print (self.volume)
                # print ('n stems')
                # print (self.stems)
                # print ('**********************')

                self.stems[ixs[m]] = (
                    self.allodic[m].biomass_to_stand.stems(bm[ixs[m]])
                    * self.remaining_share[ixs[m]]
                )

                self.basalarea[ixs[m]] = self.allodic[m].biomass_to_stand.ba(bm[ixs[m]])
                self.biomass[ixs[m]] = bm[ixs[m]]
                self.hdom[ixs[m]] = self.allodic[m].biomass_to_stand.hdom(bm[ixs[m]])
                self.Dg[ixs[m]] = self.allodic[m].biomass_to_stand.dg(bm[ixs[m]])

                self.leafarea[ixs[m]] = self.allodic[m].biomass_to_stand.lai(bm[ixs[m]])
                self.leafmass[ixs[m]] = self.allodic[m].biomass_to_stand.leaf_mass(
                    bm[ixs[m]]
                )
                # self.volume[ixs[m]] = self.allodic[m].biomass_to_stand.yi(bm[ixs[m]])
                self.volume[ixs[m]] = self.allodic[m].biomass_to_stand.vol(bm[ixs[m]])
                self.n_demand[ixs[m]] = self.allodic[m].nutrient_demand.n_demand(
                    bm[ixs[m]]
                )
                self.p_demand[ixs[m]] = self.allodic[m].nutrient_demand.p_demand(
                    bm[ixs[m]]
                )
                self.k_demand[ixs[m]] = self.allodic[m].nutrient_demand.k_demand(
                    bm[ixs[m]]
                )
                self.logvolume[ixs[m]] = self.allodic[m].yield_volume.vol_to_logs(
                    self.volume[ixs[m]]
                )
                self.finerootlitter[ixs[m]] = self.allodic[
                    m
                ].litter_mass.fine_root_litter(bm[ixs[m]])
                self.n_finerootlitter[ixs[m]] = self.allodic[
                    m
                ].nutrient_litter.n_fine_root_litter(bm[ixs[m]])
                self.p_finerootlitter[ixs[m]] = self.allodic[
                    m
                ].nutrient_litter.p_fine_root_litter(bm[ixs[m]])
                self.k_finerootlitter[ixs[m]] = self.allodic[
                    m
                ].nutrient_litter.k_fine_root_litter(bm[ixs[m]])

                self.nonwoodylitter[ixs[m]] = (
                    self.finerootlitter[ixs[m]] + self.leaf_litter[ixs[m]]
                )  # nonwoody litter kg/tree/yr
                self.n_nonwoodylitter[ixs[m]] = (
                    self.n_finerootlitter[ixs[m]] + self.Nleaf_litter[ixs[m]]
                )  # N in nonwoody litter kg/tree/yr
                self.p_nonwoodylitter[ixs[m]] = (
                    self.p_finerootlitter[ixs[m]] + self.Pleaf_litter[ixs[m]]
                )  # P in nonwoody litter kg/tree/yr
                self.k_nonwoodylitter[ixs[m]] = (
                    self.k_finerootlitter[ixs[m]] + self.Kleaf_litter[ixs[m]]
                )  # K in nonwoody litter kg/tree/yr

                self.pulpvolume[ixs[m]] = self.allodic[m].yield_volume.vol_to_pulp(
                    self.volume[ixs[m]]
                )
                self.woodylitter[ixs[m]] = self.allodic[m].litter_mass.woody_litter(
                    bm[ixs[m]]
                )
                self.n_woodylitter[ixs[m]] = self.allodic[
                    m
                ].nutrient_litter.n_woody_litter(bm[ixs[m]])
                self.p_woodylitter[ixs[m]] = self.allodic[
                    m
                ].nutrient_litter.p_woody_litter(bm[ixs[m]])
                self.k_woodylitter[ixs[m]] = self.allodic[
                    m
                ].nutrient_litter.k_woody_litter(bm[ixs[m]])

                self.woody_litter_mort[ixs[m]] = self.allodic[m].mortality_mass.woody(
                    bm[ixs[m]]
                )
                self.n_woody_litter_mort[ixs[m]] = self.allodic[
                    m
                ].nutrient_mortality.n_mortality_woody(bm[ixs[m]])
                self.p_woody_litter_mort[ixs[m]] = self.allodic[
                    m
                ].nutrient_mortality.p_mortality_woody(bm[ixs[m]])
                self.k_woody_litter_mort[ixs[m]] = self.allodic[
                    m
                ].nutrient_mortality.k_mortality_woody(bm[ixs[m]])

                self.non_woody_litter_mort[ixs[m]] = self.allodic[
                    m
                ].mortality_mass.fine_root(bm[ixs[m]])
                +self.allodic[m].mortality_mass.leaves(bm[ixs[m]])
                self.n_non_woody_litter_mort[ixs[m]] = self.allodic[
                    m
                ].nutrient_mortality.n_mortality_fine_root(bm[ixs[m]])
                +self.allodic[m].nutrient_mortality.n_mortality_leaves(bm[ixs[m]])
                self.p_non_woody_litter_mort[ixs[m]] = self.allodic[
                    m
                ].nutrient_mortality.p_mortality_fine_root(bm[ixs[m]])
                +self.allodic[m].nutrient_mortality.p_mortality_leaves(bm[ixs[m]])
                self.k_non_woody_litter_mort[ixs[m]] = self.allodic[
                    m
                ].nutrient_mortality.k_mortality_fine_root(bm[ixs[m]])
                +self.allodic[m].nutrient_mortality.k_mortality_leaves(bm[ixs[m]])

                self.yi[ixs[m]] = self.allodic[m].biomass_to_stand.yi(bm[ixs[m]])

                self.basNdemand[ixs[m]] = self.allodic[m].nutrient_demand.n_leaf_demand(
                    bm[ixs[m]]
                )
                self.basPdemand[ixs[m]] = self.allodic[m].nutrient_demand.p_leaf_demand(
                    bm[ixs[m]]
                )
                self.basKdemand[ixs[m]] = self.allodic[m].nutrient_demand.k_leaf_demand(
                    bm[ixs[m]]
                )
                self.agearr[ixs[m]] = self.agearr[ixs[m]] + 1
                # print ('vol')
                # print (self.volume)
                # print ('n stems')
                # print (self.stems)

    def assimilate(
        self, photopara, forc, wt, afp, previous_nut_stat, nut_stat, lai_above
    ):
        """
        Calls photosynthesis model (Mäkelä et al. 2008, standwise model) and leaf dynamics model that
        accounts for leaf mass, longevity and nutrient contents. This is a canopy model instance
        and the leaf dynamics is solved for similar columns along the strip.

        Parameters
        ----------
        forc : TYPE pandas dataframe
            DESCRIPTION. daily weather variables in year-long df
        wt : TYPE pandas dataframe
            DESCRIPTION. simulated water tables along the strip, shape: days, ncols
        afp : TYPE pandas dataframe
            DESCRIPTION.air-filled porosity in rooting zone, shape: days, ncols
        previous_nut_stat : TYPE array
            DESCRIPTION. nutrient status alonmg the strip in the previous year
        nut_stat : TYPE array
            DESCRIPTION. nutrient staus along the strip in the current year
        lai_above: TYPE: array
            DESCRIPTION, leaf area above the canopy layer incoming unit: m2 m-2
        Returns
        -------
        None.

        """

        """ Change unit of all variables to /tree """
        """ assimilation_yr function operates in /ha basis,"""
        lai_above = lai_above * 2  # lai_above is updated in stand object
        self.NPP, self.NPP_pot = assimilation_yr(
            photopara, forc, wt, afp, self.leafarea * 2 * self.stems, lai_above
        )  # double sided LAI required

        """
        self.NPP = (
            self.NPP * nut_stat / self.stems #* 1.1                            #This removed 05022026
        )  # returned back to tree basis unit
        
        self.NPP_pot = (
            self.NPP_pot * nut_stat / self.stems #* 1.1
        )  # returned back to tree basis units
        """

        # 1. Define the mask once
        condition = self.stems > 0

        # 2. Pre-calculate the numerator
        adjusted_npp = self.NPP * nut_stat
        adjusted_npp_pot = self.NPP_pot * nut_stat

        # 3. Perform the safe division: returned back to tree basis units
        self.NPP = np.divide(
            adjusted_npp,
            self.stems,
            out=np.full_like(adjusted_npp, np.nan),
            where=condition,
        )  # returned back to tree basis unit

        self.NPP_pot = np.divide(
            adjusted_npp_pot,
            self.stems,
            out=np.full_like(adjusted_npp_pot, np.nan),
            where=condition,
        )  # returned back to tree basis unit

        bm_increment = self.NPP
        bm = self.biomass

        current_leafmass = self.leafmass

        ixs = self.ixs
        for m in self.nlyrs:
            if m > 0:
                (
                    self.new_lmass[ixs[m]],
                    self.leaf_litter[ixs[m]],
                    self.C_consumption[ixs[m]],
                    self.leafmax[ixs[m]],
                    self.leafmin[ixs[m]],
                    self.Nleafdemand[ixs[m]],
                    self.Nleaf_litter[ixs[m]],
                    self.N_leaf[ixs[m]],
                    self.Pleafdemand[ixs[m]],
                    self.Pleaf_litter[ixs[m]],
                    self.P_leaf[ixs[m]],
                    self.Kleafdemand[ixs[m]],
                    self.Kleaf_litter[ixs[m]],
                    self.K_leaf[ixs[m]],
                    self.leafarea[ixs[m]],
                ) = self.leaf_dynamics(
                    bm[ixs[m]],
                    bm_increment[ixs[m]],
                    current_leafmass[ixs[m]],
                    previous_nut_stat[ixs[m]],
                    nut_stat[ixs[m]],
                    self.agearr[ixs[m]],
                    self.allodic[m],
                    self.species[ixs[m]],
                    printOpt=False,
                )

                self.finerootlitter[ixs[m]] = self.allodic[
                    m
                ].litter_mass.fine_root_litter(bm[ixs[m]])
                self.woodylitter[ixs[m]] = self.allodic[m].litter_mass.woody_litter(
                    bm[ixs[m]]
                )
        """
        if self.name=='dominant':
            print ('ooooooooooooooooooooooooo')
            print (self.name, np.mean(self.agearr))
            print (np.round(np.mean(self.NPP),2), 'npp' )
            print (np.round(np.mean(self.C_consumption),2), 'c cons')
            print (np.round(np.mean(self.finerootlitter), 2),'fr litter')
            print (np.round(np.mean(self.woodylitter),2),'woody l')
       """

        delta_bm_noleaves = (
            self.NPP - self.C_consumption - self.finerootlitter - self.woodylitter
        )
        self.leafmass = self.new_lmass
        vol_ini = self.volume.copy()
        """
        if self.name=='dominant': 
            print (np.round(np.mean(self.biomass),2), 'biomass ini' )
            print (np.round(np.mean(self.volume*self.stems),2), 'volume ini' )
            print (np.round(np.mean(self.stems),2), 'stems ini' )
            print (np.round(np.mean(self.allodic[1].allometry_f['bmToVol'](bm)*self.stems),2))
            #print (np.round(np.mean(self.allodic[1].allometry_f['ageToVol'](self.agearr)*self.stems), 2))
        """
        self.update(self.biomass + np.maximum(delta_bm_noleaves, 0.0))

        # if self.name=='dominant': print (np.round(np.mean(delta_bm_noleaves),2), 'delta no leaves' )

        self.volumegrowth = self.volume - vol_ini
        """
        if self.name=='dominant':
            print (np.round(np.mean(self.volumegrowth*self.stems),2), 'volumegrowth')
            print (np.round(np.mean(self.biomass),2), 'biomass after' )
            print (np.round(np.mean(self.volume*self.stems),2), 'volume after' )
            print (np.round(np.mean(self.stems),2), 'stems after' )
        """

    def leaf_dynamics(
        self,
        bm,
        bm_increment,
        current_leafmass,
        previous_nut_stat,
        nut_stat,
        agenow,
        allometry_funcs,
        species,
        printOpt=False,
    ):
        """Wrapper around module-level _leaf_dynamics for backward compatibility.

        Returns the original 15-element tuple (new_lmass, leaf_litter, C_consumption,
        leafmax, leafmin, Ndemand, Nleaf_litter, N_leaf, Pdemand, Pleaf_litter,
        P_leaf, Kdemand, Kleaf_litter, K_leaf, LAI).
        """
        ld = _leaf_dynamics(
            bm, bm_increment, current_leafmass,
            previous_nut_stat, nut_stat, agenow,
            allometry_funcs, species, printOpt,
        )

        leafbase1 = allometry_funcs.biomass_to_stand.leaf_mass(bm + bm_increment)
        leafmax = leafbase1 * 1.5
        leafmin = leafbase1 / 1.5

        sla = {"Pine": 6.8, "Spruce": 7.25, "Birch": 14.0}
        species_codes = {1: "Pine", 2: "Spruce", 3: "Birch"}
        spe = species_codes[species[0]]
        LAI = ld.new_lmass / 10000.0 * sla[spe]

        return (
            ld.new_lmass,
            ld.leaf_litter,
            ld.C_consumption,
            leafmax,
            leafmin,
            ld.Nleafdemand,
            ld.Nleaf_litter,
            ld.N_leaf,
            ld.Pleafdemand,
            ld.Pleaf_litter,
            ld.P_leaf,
            ld.Kleafdemand,
            ld.Kleaf_litter,
            ld.K_leaf,
            LAI,
        )

    def cutting(self, yr, nut_stat, to_ba=0.5):
        """Unit here /ha"""
        # OBS! All cutting is taken from uniformly from the canopy layer
        # You can locate cutting also to subdominant or lower suppressed canopy layer

        wood_density = {1: 420.0, 2: 400.0, 3: 450.0}
        wood_density_node = [wood_density[s] for s in self.species]

        if to_ba < 1.0:
            agearr = self.agearr
            ixs = self.ixs
            for m in self.nlyrs:
                if m > 0:
                    self.nonwoody_lresid[ixs[m]] = (
                        self.new_lmass[ixs[m]]
                        + self.allodic[m].fine_roots.fine_roots(self.biomass)
                    ) * self.stems
                    self.n_nonwoody_lresid[ixs[m]] = (
                        self.N_leaf
                        + self.allodic[m].fine_roots.n_fine_roots(self.biomass)
                    ) * self.stems
                    self.p_nonwoody_lresid[ixs[m]] = (
                        self.P_leaf
                        + self.allodic[m].fine_roots.p_fine_roots(self.biomass)
                    ) * self.stems
                    self.k_nonwoody_lresid[ixs[m]] = (
                        self.K_leaf
                        + self.allodic[m].fine_roots.k_fine_roots(self.biomass)
                    ) * self.stems

                    self.woody_lresid[ixs[m]] = (
                        self.allodic[m].logging_residues.woody(self.biomass)
                        * self.stems
                    )
                    self.n_woody_lresid[ixs[m]] = (
                        self.allodic[m].logging_residues.n_woody(self.biomass)
                        * self.stems
                    )
                    self.p_woody_lresid[ixs[m]] = (
                        self.allodic[m].logging_residues.p_woody(self.biomass)
                        * self.stems
                    )
                    self.k_woody_lresid[ixs[m]] = (
                        self.allodic[m].logging_residues.k_woody(self.biomass)
                        * self.stems
                    )

                    agearr[ixs[m]] = np.ones(self.ncols)[ixs[m]]
                    self.initialize_domain(agearr, nut_stat)
                    print("+        cutting in " + self.name + " year " + str(yr))
        else:
            for m in self.nlyrs:
                if m > 0:
                    print("******** Now cutting to: ", to_ba)
                    print(self.name)
                    print("basal area")
                    print(np.mean(self.basalarea * self.stems))
                    print("n stems")
                    print(np.mean(self.stems))
                    print("cut stems")
                    cut_stems = (
                        1.0 - to_ba / (self.basalarea * self.stems)
                    ) * self.stems  # stems harvested from all dimater classes
                    print(np.mean(cut_stems))
                    print("remaining stems")
                    remaining_stems = (
                        to_ba / (self.basalarea * self.stems)
                    ) * self.stems  # transferred to stem number
                    print(np.mean(remaining_stems))

                    print("Harvested volume ", np.mean(self.volume * cut_stems))

                    self.remaining_share = to_ba / (
                        self.allodic[m].biomass_to_stand.ba(self.biomass) * self.stems
                    )  # shate of stems remaining

                    print("remaining share")
                    print(np.mean(self.remaining_share))
                    agearr = self.agearr
                    ixs = self.ixs

                    print("nonwoody logging residues")
                    print(
                        np.mean(self.new_lmass[ixs[m]] * cut_stems[ixs[m]])
                    )  # leaf logging residues
                    print(
                        np.mean(
                            self.allodic[m].fine_roots.fine_roots(self.biomass[ixs[m]])
                            * cut_stems[ixs[m]]
                        )
                    )  # fine root logging residues

                    self.nonwoody_lresid[ixs[m]] = (
                        self.new_lmass[ixs[m]]
                        + self.allodic[m].fine_roots.fine_roots(self.biomass[ixs[m]])
                    ) * cut_stems[
                        ixs[m]
                    ]  # stemwise fineroot biomass multipled by number of cut stems

                    self.n_nonwoody_lresid[ixs[m]] = (
                        self.N_leaf[ixs[m]]
                        + self.allodic[m].fine_roots.n_fine_roots(self.biomass[ixs[m]])
                    ) * cut_stems[ixs[m]]
                    self.p_nonwoody_lresid[ixs[m]] = (
                        self.P_leaf[ixs[m]]
                        + self.allodic[m].fine_roots.p_fine_roots(self.biomass[ixs[m]])
                    ) * cut_stems[ixs[m]]
                    self.k_nonwoody_lresid[ixs[m]] = (
                        self.K_leaf[ixs[m]]
                        + self.allodic[m].fine_roots.k_fine_roots(self.biomass[ixs[m]])
                    ) * cut_stems[ixs[m]]

                    self.woody_lresid[ixs[m]] = (
                        self.allodic[m].logging_residues.woody(self.biomass[ixs[m]])
                        * cut_stems[ixs[m]]
                    )
                    self.n_woody_lresid[ixs[m]] = (
                        self.allodic[m].logging_residues.n_woody(self.biomass[ixs[m]])
                        * cut_stems[ixs[m]]
                    )
                    self.p_woody_lresid[ixs[m]] = (
                        self.allodic[m].logging_residues.p_woody(self.biomass[ixs[m]])
                        * cut_stems[ixs[m]]
                    )
                    self.k_woody_lresid[ixs[m]] = (
                        self.allodic[m].logging_residues.k_woody(self.biomass[ixs[m]])
                        * cut_stems[ixs[m]]
                    )

                    print("nonwoodylogging resids after adding")
                    print(np.mean(self.nonwoody_lresid[ixs[m]]))

                    print("woody logging residues")
                    print(np.mean(self.woody_lresid))

                    self.harvested_volume[ixs[m]] = (
                        self.allodic[m].biomass_to_stand.vol(self.biomass[ixs[m]])
                        * cut_stems[ixs[m]]
                    )  # harvested volume m3/tree
                    self.harvested_log_volume[ixs[m]] = (
                        self.allodic[m].biomass_to_stand.log_vol(self.biomass[ixs[m]])
                        * cut_stems[ixs[m]]
                    )  # harvested saw log volume m3/tree
                    self.harvested_pulp_volume[ixs[m]] = (
                        self.allodic[m].biomass_to_stand.pulp_vol(self.biomass[ixs[m]])
                        * cut_stems[ixs[m]]
                    )  # harvsted pulp volume m3/tree
                    self.harvested_biomass[ixs[m]] = (
                        (
                            self.allodic[m].biomass_to_stand.log_vol(
                                self.biomass[ixs[m]]
                            )
                            + self.allodic[m].biomass_to_stand.pulp_vol(
                                self.biomass[ixs[m]]
                            )
                        )
                        * cut_stems[ixs[m]]
                        * wood_density_node
                    )  # saw biomass kg/tree
                    self.harvested_stems[ixs[m]] = cut_stems[
                        ixs[m]
                    ]  # number of harvested stems/tree

                    print("harvested logs", np.mean(self.harvested_log_volume))
                    print("harvested pulp", np.mean(self.harvested_pulp_volume))
                    print("harvested biomass", np.mean(self.harvested_biomass))

                    self.update(self.biomass)
                    """This update to stand or to main????? """
