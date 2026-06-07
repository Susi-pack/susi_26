# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:59:52 2022

@author: alauren
"""

import dataclasses
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from supersusi.core import canopylayer



@dataclass(frozen=True)
class Params:
    dominant: canopylayer.Params
    subdominant: canopylayer.Params
    under: canopylayer.Params


@dataclass(frozen=True)
class ComputedConstants:
    dominant: canopylayer.ComputedConstants
    subdominant: canopylayer.ComputedConstants
    under: canopylayer.ComputedConstants


@dataclass(frozen=True)
class State:
    nut_stat: np.ndarray
    previous_nut_stat: np.ndarray
    dominant: canopylayer.State
    subdominant: canopylayer.State
    under: canopylayer.State


@dataclass(frozen=True)
class Outputs:
    basalarea: np.ndarray
    biomass: np.ndarray
    hdom: np.ndarray
    leafarea: np.ndarray
    leafmass: np.ndarray
    stems: np.ndarray
    volume: np.ndarray
    volumegrowth: np.ndarray
    yi: np.ndarray
    logvolume: np.ndarray
    pulpvolume: np.ndarray
    mean_diameter: np.ndarray
    biomassgrowth: np.ndarray

    NPP: np.ndarray
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
class Inputs:
    photopara: Any
    forc: pd.DataFrame = field(doc="annual weather")
    wt: pd.DataFrame = field(doc="annual water table")
    afp: pd.DataFrame = field(doc="annual air-filled porosity")
    n_supply: np.ndarray
    p_supply: np.ndarray
    k_supply: np.ndarray
    groundvegetation_outputs: Any
    previous_nut_stat: np.ndarray
    calendar_year: int
    cutting_to_ba: float | None = None


_PER_TREE_FIELDS: list[str] = [
    "basalarea", "volume", "leafarea", "leafmass", "volumegrowth",
    "logvolume", "pulpvolume", "yi",
    "NPP", "NPP_pot", "new_lmass", "leaf_litter", "C_consumption",
    "Nleafdemand", "Nleaf_litter", "N_leaf",
    "Pleafdemand", "Pleaf_litter", "P_leaf",
    "Kleafdemand", "Kleaf_litter", "K_leaf",
    "finerootlitter", "n_finerootlitter", "p_finerootlitter", "k_finerootlitter",
    "nonwoodylitter", "n_nonwoodylitter", "p_nonwoodylitter", "k_nonwoodylitter",
    "woodylitter", "n_woodylitter", "p_woodylitter", "k_woodylitter",
    "woody_litter_mort", "n_woody_litter_mort", "p_woody_litter_mort", "k_woody_litter_mort",
    "non_woody_litter_mort", "n_non_woody_litter_mort", "p_non_woody_litter_mort", "k_non_woody_litter_mort",
    "n_demand", "p_demand", "k_demand",
    "basNdemand", "basPdemand", "basKdemand",
]

_CUTTING_FIELDS: list[str] = [
    "harvested_volume", "harvested_log_volume", "harvested_pulp_volume",
    "harvested_biomass", "harvested_stems",
    "nonwoody_lresid", "n_nonwoody_lresid", "p_nonwoody_lresid", "k_nonwoody_lresid",
    "woody_lresid", "n_woody_lresid", "p_woody_lresid", "k_woody_lresid",
]


def _aggregate(
    dom_out: canopylayer.Outputs,
    sub_out: canopylayer.Outputs,
    under_out: canopylayer.Outputs,
    dom_biomass: np.ndarray,
    sub_biomass: np.ndarray,
    under_biomass: np.ndarray,
    previous_stand_biomass: np.ndarray | None = None,
) -> Outputs:
    total_stems = dom_out.stems + sub_out.stems + under_out.stems
    kw: dict[str, np.ndarray] = {}

    for fname in _PER_TREE_FIELDS:
        kw[fname] = (
            getattr(dom_out, fname) * dom_out.stems
            + getattr(sub_out, fname) * sub_out.stems
            + getattr(under_out, fname) * under_out.stems
        )

    # BUG: OOP Stand.update() line 881 has a no-op for k_non_woody_litter_mort —
    #      self.k_non_woody_litter_mort = self.k_non_woody_litter_mort (no sum added).
    #      Preserve this to match golden file.
    kw["k_non_woody_litter_mort"] = np.zeros_like(kw["k_non_woody_litter_mort"])

    kw["stems"] = total_stems
    kw["hdom"] = np.maximum(dom_out.hdom, np.maximum(sub_out.hdom, under_out.hdom))
    numerator = (
        dom_out.Dg * dom_out.stems
        + sub_out.Dg * sub_out.stems
        + under_out.Dg * under_out.stems
    )
    kw["mean_diameter"] = np.divide(
        numerator, total_stems, out=np.zeros_like(numerator), where=total_stems > 0,
    )

    biomass_val = (
        dom_biomass * dom_out.stems
        + sub_biomass * sub_out.stems
        + under_biomass * under_out.stems
    )
    kw["biomass"] = biomass_val
    if previous_stand_biomass is not None:
        kw["biomassgrowth"] = biomass_val - previous_stand_biomass
    else:
        kw["biomassgrowth"] = np.zeros_like(biomass_val)

    for fname in _CUTTING_FIELDS:
        kw[fname] = np.zeros_like(biomass_val)

    return Outputs(**kw)



def _merge_cutting_outputs(
    stand_out: Outputs,
    dom_cut: canopylayer.CuttingOutputs | None = None,
    sub_cut: canopylayer.CuttingOutputs | None = None,
    under_cut: canopylayer.CuttingOutputs | None = None,
) -> Outputs:
    kw: dict[str, np.ndarray] = {}
    for fname in _CUTTING_FIELDS:
        total = np.zeros_like(getattr(stand_out, fname))
        if dom_cut is not None:
            total += getattr(dom_cut, fname)
        if sub_cut is not None:
            total += getattr(sub_cut, fname)
        if under_cut is not None:
            total += getattr(under_cut, fname)
        kw[fname] = total
    return replace(stand_out, **kw)


def _compute_lai_above(
    dom_allom: canopylayer.Outputs,
    sub_allom: canopylayer.Outputs,
    under_allom: canopylayer.Outputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height-order LAI for shading between layers.

    Returns (lai_above_dom, lai_above_sub, lai_above_under) — each is the
    cumulative LAI (leafarea × stems) of all taller layers for that column.
    """
    heightarray = np.vstack([dom_allom.hdom, sub_allom.hdom, under_allom.hdom])
    h_order = np.argsort(heightarray * -1, axis=0)
    laiarray = np.vstack([
        dom_allom.leafarea * dom_allom.stems,
        sub_allom.leafarea * sub_allom.stems,
        under_allom.leafarea * under_allom.stems,
    ])

    n = laiarray.shape[1]
    laiout = np.zeros((3, n))
    for layer in range(3):
        laiout[layer] = laiarray[h_order[layer], np.arange(n)]

    laiabove = np.cumsum(laiout, axis=0)

    lai_above = np.zeros((3, n))
    for layer in range(3):
        order = h_order[layer]
        if layer == 0:
            lai_above[order, np.arange(n)] = 0.0
        else:
            lai_above[order, np.arange(n)] = laiabove[layer - 1]

    return lai_above[0], lai_above[1], lai_above[2]


def compute_constants(
    params: Params,
    allometry_params: Any,
) -> ComputedConstants:
    return ComputedConstants(
        dominant=canopylayer.compute_constants(
            params.dominant, allometry_params.dominant_data, allometry_params.dominant_species_id,
        ),
        subdominant=canopylayer.compute_constants(
            params.subdominant, allometry_params.subdominant_data, allometry_params.subdominant_species_id,
        ),
        under=canopylayer.compute_constants(
            params.under, allometry_params.under_data, allometry_params.under_species_id,
        ),
    )


def compute_initial_state(
    params: Params,
    cc: ComputedConstants,
    agearr: dict[str, np.ndarray],
    ncols: int,
) -> tuple[State, Outputs, canopylayer.Outputs, canopylayer.Outputs, canopylayer.Outputs]:
    nut_stat = np.ones(ncols)
    dom_state, dom_out = canopylayer.compute_initial_state(
        params.dominant, cc.dominant, agearr["dominant"], nut_stat,
    )
    sub_state, sub_out = canopylayer.compute_initial_state(
        params.subdominant, cc.subdominant, agearr["subdominant"], nut_stat,
    )
    under_state, under_out = canopylayer.compute_initial_state(
        params.under, cc.under, agearr["under"], nut_stat,
    )
    stand_out = _aggregate(
        dom_out, sub_out, under_out,
        dom_state.biomass, sub_state.biomass, under_state.biomass,
        previous_stand_biomass=None,
    )
    # BUG: OOP doesn't compute mean_diameter for year 0 (initial state write
    #      happens before Stand.update()). Zero it out to match golden.
    stand_out = replace(stand_out, mean_diameter=np.zeros(ncols))
    return State(nut_stat=nut_stat, previous_nut_stat=nut_stat.copy(), dominant=dom_state, subdominant=sub_state, under=under_state), stand_out, dom_out, sub_out, under_out


def grow_stand(
    state: State,
    cc: ComputedConstants,
    inputs: Inputs,
) -> tuple[State, Outputs, canopylayer.Outputs, canopylayer.Outputs, canopylayer.Outputs]:
    dom_allom = canopylayer.apply_allometry(
        state.dominant.biomass, state.dominant.agearr, state.dominant.remaining_share, cc.dominant,
    )
    sub_allom = canopylayer.apply_allometry(
        state.subdominant.biomass, state.subdominant.agearr, state.subdominant.remaining_share, cc.subdominant,
    )
    under_allom = canopylayer.apply_allometry(
        state.under.biomass, state.under.agearr, state.under.remaining_share, cc.under,
    )

    lai_dom, lai_sub, lai_under = _compute_lai_above(dom_allom, sub_allom, under_allom)

    dom_inputs = canopylayer.Inputs(
        photopara=inputs.photopara, forc=inputs.forc, wt=inputs.wt, afp=inputs.afp,
        previous_nut_stat=inputs.previous_nut_stat, nut_stat=state.nut_stat,
        lai_above=lai_dom,
    )
    sub_inputs = replace(dom_inputs, lai_above=lai_sub)
    under_inputs = replace(dom_inputs, lai_above=lai_under)

    dom_state, dom_out = canopylayer.grow_stand(state.dominant, cc.dominant, dom_inputs)
    sub_state, sub_out = canopylayer.grow_stand(state.subdominant, cc.subdominant, sub_inputs)
    under_state, under_out = canopylayer.grow_stand(state.under, cc.under, under_inputs)

    prev_biomass = (
        state.dominant.biomass * dom_allom.stems
        + state.subdominant.biomass * sub_allom.stems
        + state.under.biomass * under_allom.stems
    )

    stand_out = _aggregate(
        dom_out, sub_out, under_out,
        dom_state.biomass, sub_state.biomass, under_state.biomass,
        previous_stand_biomass=prev_biomass,
    )

    new_stand_state = State(state.nut_stat, state.previous_nut_stat, dom_state, sub_state, under_state)
    return new_stand_state, stand_out, dom_out, sub_out, under_out


_TAU = 3.0
_NUT_LOWER = 0.5
_NUT_UPPER = 2.0
_REINEKE_K = 4.35
_REINEKE_SLOPE = -1.605
_AREA_MOD_LOWER = 0.01
_AREA_MOD_UPPER = 1.0


def update_nutrient_status(
    state: State,
    stand_out: Outputs,
    inputs: Inputs,
) -> State:
    gv = inputs.groundvegetation_outputs
    diameter = stand_out.mean_diameter
    stems = stand_out.stems
    safe = np.maximum(diameter, 1e-30)
    area_modifier = np.clip(
        stems / ((safe / 2.54) ** _REINEKE_SLOPE * 10 ** _REINEKE_K),
        _AREA_MOD_LOWER, _AREA_MOD_UPPER,
    )
    n_ratio = (inputs.n_supply * area_modifier) / (
        stand_out.n_demand + stand_out.Nleafdemand + gv.nup + 1e-30
    )
    p_ratio = (inputs.p_supply * area_modifier) / (
        stand_out.p_demand + stand_out.Pleafdemand + gv.pup + 1e-30
    )
    k_ratio = (inputs.k_supply * area_modifier) / (
        stand_out.k_demand + stand_out.Kleafdemand + gv.kup + 1e-30
    )
    min_ratio = np.minimum(np.minimum(n_ratio, p_ratio), k_ratio)
    new_nut_stat = state.nut_stat + (min_ratio - state.nut_stat) / _TAU
    new_nut_stat = np.clip(new_nut_stat, _NUT_LOWER, _NUT_UPPER)
    return replace(state, previous_nut_stat=state.nut_stat.copy(), nut_stat=new_nut_stat)


def cut_stand(
    state: State,
    cc: ComputedConstants,
    stand_out: Outputs,
    inputs: Inputs,
) -> tuple[State, Outputs, canopylayer.CuttingOutputs]:
    dom_out = canopylayer.apply_allometry(
        state.dominant.biomass, state.dominant.agearr, state.dominant.remaining_share, cc.dominant,
    )
    dom_state, dom_cut = canopylayer.cut_stand(
        state.dominant, cc.dominant, dom_out, state.nut_stat, inputs.cutting_to_ba,
    )
    stand_out = _merge_cutting_outputs(stand_out, dom_cut=dom_cut)
    new_state = State(
        nut_stat=state.nut_stat,
        previous_nut_stat=state.previous_nut_stat,
        dominant=dom_state,
        subdominant=state.subdominant,
        under=state.under,
    )
    return new_state, stand_out, dom_cut


def assimilate_stand(
    state: State,
    cc: ComputedConstants,
    inputs: Inputs,
) -> tuple[State, Outputs, canopylayer.CuttingOutputs]:
    new_state, stand_out, *_ = grow_stand(state, cc, inputs)

    if inputs.cutting_to_ba is not None:
        new_state, stand_out, cut_out = cut_stand(new_state, cc, stand_out, inputs)
    else:
        cut_out = canopylayer.CuttingOutputs(
            *[np.zeros_like(state.nut_stat) for _ in dataclasses.fields(canopylayer.CuttingOutputs)],
        )

    new_state = update_nutrient_status(new_state, stand_out, inputs)

    return new_state, stand_out, cut_out


class Stand:
    def __init__(
        self,
        n_scenarios,
        n_yrs,
        canopylayers,
        n_cols,
        sfc,
        agearr,
        allometry_params,
        photopara,
    ):
        """
        ALL VARIABLES IN STAND OBJECT ARE IN ha AND kg -BASIS
        Creates canopy layer instances
        Stand object composes of canopy layer objects and keeps track on the
        stand-wise sums of the variables
        Stand is array-form and has dimensions of number of columns in the strip
        Input:
            nscens , int, number of scenarios in the simulation
            yrs, int, number of years in the simulation
            canopylayers, CanopyLayerAllometryPointers in spara, contains integer arrays (len(ncols)) for each canopy layer pointing to specific Motti file
            ncols, int, number of columns along the strip
            sfc, site fertility class
            agearr, dict of float arrays (len(ncols)) for stand age in the particular column and canopylayer
            allometry_parameters: AllometryParams
            photopara - photosynthesis parameters used in the assimilation model
        """
        self.n_cols = n_cols  # number of columns along the strip
        self.n_scenarios = (
            n_scenarios  # number of ditch depth scenarios in the simulation
        )
        self.n_yrs = n_yrs  # number of years in the simulation
        self.nut_stat = np.ones(
            n_cols
        )  # *0.5                                   # nutrient status, make this an argument

        ndominants = np.unique(canopylayers.dominant)
        nsubdominants = np.unique(canopylayers.subdominant)
        nunder = np.unique(canopylayers.under)

        ixdominants = {}  # location indices for dominant canopy layers, along the transect
        for m in ndominants:
            if m > 0:
                ixdominants[m] = np.where(canopylayers.dominant == m)

        ixsubdominants = {}  # location indices for subdominant canopy layers
        for m in nsubdominants:
            if m > 0:
                ixsubdominants[m] = np.where(canopylayers.subdominant == m)

        ixunder = {}  # location indices for undersmost canopy layer
        for m in nunder:
            if m > 0:
                ixunder[m] = np.where(canopylayers.under == m)

        self.dominant = canopylayer.Canopylayer(
            "dominant",
            n_scenarios,
            n_yrs,
            n_cols,
            ndominants,
            sfc,
            agearr["dominant"],
            allometry_params.dominant_data,
            allometry_params.dominant_species_id,
            ixdominants,
            photopara,
            self.nut_stat,
        )
        self.subdominant = canopylayer.Canopylayer(
            "subdominant",
            n_scenarios,
            n_yrs,
            n_cols,
            nsubdominants,
            sfc,
            agearr["subdominant"],
            allometry_params.subdominant_data,
            allometry_params.subdominant_species_id,
            ixsubdominants,
            photopara,
            self.nut_stat,
        )
        self.under = canopylayer.Canopylayer(
            "under",
            n_scenarios,
            n_yrs,
            n_cols,
            nunder,
            sfc,
            agearr["under"],
            allometry_params.under_data,
            allometry_params.under_species_id,
            ixunder,
            photopara,
            self.nut_stat,
        )
        self.clyrs = [
            self.dominant,
            self.subdominant,
            self.under,
        ]  # list of canopy layers, used later in loops

        # ---------- create stand variables------------------------------------
        self.basalarea = np.zeros(n_cols, dtype=float)  # stand basal area m2/ha
        self.biomass = np.zeros(n_cols, dtype=float)  # stand dry biomass kg/ha
        self.n_demand = np.zeros(
            n_cols, dtype=float
        )  # stand N demand excluding leaves kg/ha/yr
        self.p_demand = np.zeros(
            n_cols, dtype=float
        )  # stand N demand excluding leaves kg/ha/yr
        self.k_demand = np.zeros(
            n_cols, dtype=float
        )  # stand N demand excluding leaves kg/ha/yr
        self.hdom = np.zeros(n_cols, dtype=float)  # dominant height m
        self.leafarea = np.zeros(n_cols, dtype=float)  # one sided leaf area m2 m-2
        self.leafmass = np.zeros(n_cols, dtype=float)  # leaf dry biomass kg/ha
        self.logvolume = np.zeros(n_cols, dtype=float)  # saw log volume m3/ha
        self.mean_diameter = np.zeros(n_cols, dtype=float)  # stand mean diameter cm

        self.harvested_volume = np.zeros(n_cols, dtype=float)  # harvested volume m3/ha
        self.harvested_log_volume = np.zeros(
            n_cols, dtype=float
        )  # harvested saw log volume m3/ha
        self.harvested_pulp_volume = np.zeros(
            n_cols, dtype=float
        )  # harvsted pulp volume m3/ha
        self.harvested_biomass = np.zeros(n_cols, dtype=float)  # saw biomass kg/ha
        self.harvested_stems = np.zeros(
            n_cols, dtype=float
        )  # number of harvested stems/ha

        self.finerootlitter = np.zeros(n_cols, dtype=float)  # fine root litter kg/ha/yr
        self.n_finerootlitter = np.zeros(
            n_cols, dtype=float
        )  # N in fine root litter kg/ha/yr
        self.p_finerootlitter = np.zeros(
            n_cols, dtype=float
        )  # P in fine root litter kg/ha/yr
        self.k_finerootlitter = np.zeros(
            n_cols, dtype=float
        )  # K in fine root litter kg/ha/yr
        self.nonwoodylitter = np.zeros(n_cols, dtype=float)  # non woody litter kg/ha/yr
        self.n_nonwoodylitter = np.zeros(
            n_cols, dtype=float
        )  # N in nonwoody litter kg/ha/yr
        self.p_nonwoodylitter = np.zeros(
            n_cols, dtype=float
        )  # P in nonwoody litter kg/ha/yr
        self.k_nonwoodylitter = np.zeros(
            n_cols, dtype=float
        )  # K in nonwoody litter kg/ha/yr
        self.pulpvolume = np.zeros(n_cols, dtype=float)  # pulpwood volume m3/ha
        self.stems = np.zeros(n_cols, dtype=float)  # stocking, number of stems pcs/ha
        self.volume = np.zeros(
            n_cols, dtype=float
        )  # total volume of the growing stock m3/ha
        self.volumegrowth = np.zeros(
            n_cols, dtype=float
        )  # total volume growth of the growing stock m3/ha/yr
        self.biomassgrowth = np.zeros(
            n_cols, dtype=float
        )  # total biomass growth of stand kg/ha/yr
        self.woodylitter = np.zeros(n_cols, dtype=float)  # woody litter kg/ha/yr
        self.n_woodylitter = np.zeros(n_cols, dtype=float)  # N in woody litter kg/ha/yr
        self.p_woodylitter = np.zeros(n_cols, dtype=float)  # P in woody litter kg/ha/yr
        self.k_woodylitter = np.zeros(n_cols, dtype=float)  # K in woody litter kg/ha/yr
        self.yi = np.zeros(n_cols, dtype=float)  # yield. here same as volume

        self.nonwoody_lresid = np.zeros(
            n_cols, dtype=float
        )  # nonwoody logging residues kg/ha
        self.n_nonwoody_lresid = np.zeros(
            n_cols, dtype=float
        )  # N in nonwoody logging residues kg/ha
        self.p_nonwoody_lresid = np.zeros(
            n_cols, dtype=float
        )  # P in nonwoody logging residues kg/ha
        self.k_nonwoody_lresid = np.zeros(
            n_cols, dtype=float
        )  # K in nonwoody logging residues kg/ha

        self.woody_lresid = np.zeros(
            n_cols, dtype=float
        )  # woody logging residues kg/ha
        self.n_woody_lresid = np.zeros(
            n_cols, dtype=float
        )  # N in woody logging residues kg/ha
        self.p_woody_lresid = np.zeros(
            n_cols, dtype=float
        )  # P in woody logging residues kg/ha
        self.k_woody_lresid = np.zeros(
            n_cols, dtype=float
        )  # K in woody logging residues kg/ha

        self.woody_litter_mort = np.zeros(
            n_cols, dtype=float
        )  # woody litter from mortality kg/ha
        self.n_woody_litter_mort = np.zeros(
            n_cols, dtype=float
        )  # N in woody litter from mortality kg/ha
        self.p_woody_litter_mort = np.zeros(
            n_cols, dtype=float
        )  # P in woody litter from mortality kg/ha
        self.k_woody_litter_mort = np.zeros(
            n_cols, dtype=float
        )  # K in woody litter from mortality kg/ha

        self.non_woody_litter_mort = np.zeros(
            n_cols, dtype=float
        )  # non-woody litter from mortality kg/ha
        self.n_non_woody_litter_mort = np.zeros(
            n_cols, dtype=float
        )  # N in non-woody litter from mortality kg/ha
        self.p_non_woody_litter_mort = np.zeros(
            n_cols, dtype=float
        )  # P in non-woody litter from mortality kg/ha
        self.k_non_woody_litter_mort = np.zeros(
            n_cols, dtype=float
        )  # K in non-woody litter from mortality kg/ha

        self.basNdemand = np.zeros(
            n_cols, dtype=float
        )  # basic N demand kg/tree, in table growth conditions, used in nutrient status calculation
        self.basPdemand = np.zeros(
            n_cols, dtype=float
        )  # basic P demand kg/tree, in table growth conditions, used in nutrient status calculation
        self.basKdemand = np.zeros(
            n_cols, dtype=float
        )  # basic K demand kg/tree, in table growth conditions, used in nutrient status calculation

        self.n_leaf_demand = np.zeros(
            n_cols, dtype=float
        )  # current leaf demand for N kg/ha
        self.p_leaf_demand = np.zeros(
            n_cols, dtype=float
        )  # current leaf demand for P kg/ha
        self.k_leaf_demand = np.zeros(
            n_cols, dtype=float
        )  # current leaf demand for K kg/ha

        self.previous_nut_stat = np.ones(n_cols)  # nutrient status in previous year

        """ ATTN cl in tree basis, convert to ha basis"""
        for cl in self.clyrs:  # sum standwise initial values from the canopy layers
            self.basalarea = (
                self.basalarea + cl.basalarea * cl.stems
            )  # Here: cl.basalarea in tree basis, conversion to ha basis
            self.biomass = self.biomass + cl.biomass * cl.stems
            self.hdom = np.maximum(self.hdom, cl.hdom)
            self.leafarea = self.leafarea + cl.leafarea * cl.stems
            self.leafmass = self.leafmass + cl.leafmass * cl.stems
            self.stems = self.stems + cl.stems
            self.volume = self.volume + cl.volume * cl.stems
            self.volumegrowth = self.volumegrowth + cl.volumegrowth * cl.stems

    def reset_domain(self, agearr):
        """
        Resets stand domain, re-initializes canopy layer instances
        by setting the initial values for the state variables
        Sums all canopy layers to gain initial values for the stand
        Parameters
        ----------
        agearr : TYPE dictionary of array of floats, len(ncols)
            DESCRIPTION. dictionary of stand ages in the beginning of simulation in each canopy layer
        Returns
        -------
        None.

        """
        # reset the stand and reinitialize with initial age for a new scenario
        self.previous_nut_stat = np.ones(self.n_cols)
        self.nut_stat = np.ones(self.n_cols)  # *0.5

        self.dominant.initialize_domain(agearr["dominant"], self.nut_stat)
        self.subdominant.initialize_domain(agearr["subdominant"], self.nut_stat)
        self.under.initialize_domain(agearr["under"], self.nut_stat)
        self.reset_vars()  # sets all variables to zero

        """ ATTN cl in tree basis, convert to ha basis"""
        for cl in self.clyrs:  # sum over the canopy layers to get standwise values
            self.basalarea = self.basalarea + cl.basalarea * cl.stems
            self.biomass = self.biomass + cl.biomass * cl.stems
            self.hdom = np.maximum(self.hdom, cl.hdom)
            self.leafarea = self.leafarea + cl.leafarea * cl.stems
            self.leafmass = self.leafmass + cl.leafmass * cl.stems
            self.stems = self.stems + cl.stems
            self.volume = self.volume + cl.volume * cl.stems

    def reset_vars(self):
        """
        Stand variables are sums over canopy layer variables - therefore the sum variables
        are needed to set to zero before a new summation

        """
        self.basalarea = self.basalarea * 0.0
        self.biomass = self.biomass * 0.0
        self.hdom = self.hdom * 0.0
        self.leafarea = self.leafarea * 0.0
        self.leafmass = self.leafmass * 0.0
        self.stems = self.stems * 0.0
        self.volume = self.volume * 0.0
        self.volumegrowth = self.volumegrowth * 0.0
        self.yi = self.yi * 0.0
        self.logvolume = self.logvolume * 0.0
        self.pulpvolume = self.pulpvolume * 0.0
        self.finerootlitter = self.finerootlitter * 0.0
        self.n_finerootlitter = self.n_finerootlitter * 0.0
        self.p_finerootlitter = self.p_finerootlitter * 0.0
        self.k_finerootlitter = self.k_finerootlitter * 0.0
        self.nonwoodylitter = self.nonwoodylitter * 0.0  # woody litter kg/ha/yr
        self.n_nonwoodylitter = (
            self.n_nonwoodylitter * 0.0
        )  # N in woody litter kg/ha/yr
        self.p_nonwoodylitter = (
            self.p_nonwoodylitter * 0.0
        )  # P in woody litter kg/ha/yr
        self.k_nonwoodylitter = (
            self.k_nonwoodylitter * 0.0
        )  # K in woody litter kg/ha/yr
        self.woodylitter = self.woodylitter * 0.0
        self.n_woodylitter = self.n_woodylitter * 0.0
        self.p_woodylitter = self.p_woodylitter * 0.0
        self.k_woodylitter = self.k_woodylitter * 0.0
        self.n_demand = self.n_demand * 0.0
        self.p_demand = self.p_demand * 0.0
        self.k_demand = self.k_demand * 0.0

        self.nonwoody_lresid = self.nonwoody_lresid * 0.0
        self.n_nonwoody_lresid = self.n_nonwoody_lresid * 0.0
        self.p_nonwoody_lresid = self.p_nonwoody_lresid * 0.0
        self.k_nonwoody_lresid = self.k_nonwoody_lresid * 0.0

        self.woody_lresid = self.woody_lresid * 0.0
        self.n_woody_lresid = self.n_woody_lresid * 0.0
        self.p_woody_lresid = self.p_woody_lresid * 0.0
        self.k_woody_lresid = self.k_woody_lresid * 0.0

        self.woody_litter_mort = self.woody_litter_mort * 0.0
        self.n_woody_litter_mort = self.n_woody_litter_mort * 0.0
        self.p_woody_litter_mort = self.p_woody_litter_mort * 0.0
        self.k_woody_litter_mort = self.k_woody_litter_mort * 0.0

        self.non_woody_litter_mort = self.non_woody_litter_mort * 0.0
        self.n_non_woody_litter_mort = self.n_non_woody_litter_mort * 0.0
        self.p_non_woody_litter_mort = self.p_non_woody_litter_mort * 0.0
        self.k_non_woody_litter_mort = self.k_non_woody_litter_mort * 0.0

        self.basNdemand = (
            self.basNdemand * 0.0
        )  # basic N demand kg/tree, in table growth conditions, used in nutrient status calculation
        self.basPdemand = (
            self.basPdemand * 0.0
        )  # basic P demand kg/tree, in table growth conditions, used in nutrient status calculation
        self.basKdemand = (
            self.basKdemand * 0.0
        )  # basic K demand kg/tree, in table growth conditions, used in nutrient status calculation

        self.n_leaf_demand = self.n_leaf_demand * 0.0
        self.p_leaf_demand = self.n_leaf_demand * 0.0
        self.k_leaf_demand = self.n_leaf_demand * 0.0

        self.harvested_volume = self.harvested_volume * 0.0  # harvested volume m3/ha
        self.harvested_log_volume = (
            self.harvested_log_volume * 0.0
        )  # harvested saw log volume m3/ha
        self.harvested_pulp_volume = (
            self.harvested_pulp_volume * 0.0
        )  # harvsted pulp volume m3/ha
        self.harvested_biomass = self.harvested_biomass * 0.0  # saw biomass kg/ha
        self.harvested_stems = (
            self.harvested_stems * 0.0
        )  # number of harvested stems/ha

        self.mean_diameter = self.mean_diameter * 0.0

    def update(self):
        """
        ALL UNITS must be converted to ha BASIS
        Updates the stand variables by summing all the canopy layers
        Calls canopy layer instances
        Note: to be run after the self.layer.assimilate
        Returns
        -------
        None.

        """
        biomass_ini = self.biomass

        self.reset_vars()

        """ ATTN cl in tree basis, convert to ha basis"""
        for cl in self.clyrs:
            self.basalarea = self.basalarea + cl.basalarea * cl.stems
            self.biomass = self.biomass + cl.biomass * cl.stems
            self.hdom = np.maximum(self.hdom, cl.hdom)
            self.leafarea = self.leafarea + cl.leafarea * cl.stems
            self.leafmass = self.leafmass + cl.leafmass * cl.stems
            self.stems = self.stems + cl.stems
            self.volume = self.volume + cl.volume * cl.stems
            self.volumegrowth = self.volumegrowth + cl.volumegrowth * cl.stems
            self.yi = self.yi + cl.yi * cl.stems
            self.logvolume = self.logvolume + cl.logvolume * cl.stems
            self.pulpvolume = self.pulpvolume + cl.pulpvolume * cl.stems
            self.finerootlitter = self.finerootlitter + cl.finerootlitter * cl.stems
            self.n_finerootlitter = (
                self.n_finerootlitter + cl.n_finerootlitter * cl.stems
            )
            self.p_finerootlitter = (
                self.p_finerootlitter + cl.p_finerootlitter * cl.stems
            )
            self.k_finerootlitter = (
                self.k_finerootlitter + cl.k_finerootlitter * cl.stems
            )

            self.nonwoodylitter = (
                self.nonwoodylitter + cl.nonwoodylitter * cl.stems
            )  # woody litter kg/ha/yr
            self.n_nonwoodylitter = (
                self.n_nonwoodylitter + cl.n_nonwoodylitter * cl.stems
            )  # N in woody litter kg/ha/yr
            self.p_nonwoodylitter = (
                self.p_nonwoodylitter + cl.p_nonwoodylitter * cl.stems
            )  # P in woody litter kg/ha/yr
            self.k_nonwoodylitter = (
                self.k_nonwoodylitter + cl.k_nonwoodylitter * cl.stems
            )  # K in woody litter kg/ha/yr

            self.woodylitter = self.woodylitter + cl.woodylitter * cl.stems
            self.n_woodylitter = self.n_woodylitter + cl.n_woodylitter * cl.stems
            self.p_woodylitter = self.p_woodylitter + cl.p_woodylitter * cl.stems
            self.k_woodylitter = self.k_woodylitter + cl.k_woodylitter * cl.stems

            self.woody_litter_mort = (
                self.woody_litter_mort + cl.woody_litter_mort * cl.stems
            )
            self.n_woody_litter_mort = (
                self.n_woody_litter_mort + cl.n_woody_litter_mort * cl.stems
            )
            self.p_woody_litter_mort = (
                self.p_woody_litter_mort + cl.p_woody_litter_mort * cl.stems
            )
            self.k_woody_litter_mort = (
                self.k_woody_litter_mort + cl.k_woody_litter_mort * cl.stems
            )

            self.non_woody_litter_mort = (
                self.non_woody_litter_mort + cl.non_woody_litter_mort * cl.stems
            )
            self.n_non_woody_litter_mort = (
                self.n_non_woody_litter_mort + cl.n_non_woody_litter_mort * cl.stems
            )
            self.p_non_woody_litter_mort = (
                self.p_non_woody_litter_mort + cl.p_non_woody_litter_mort * cl.stems
            )
            self.k_non_woody_litter_mort = self.k_non_woody_litter_mort

            # self.n_demand = self.n_demand + (cl.n_demand + cl.Nleafdemand) * cl.stems
            # self.p_demand = self.p_demand + (cl.p_demand+ cl.Pleafdemand) * cl.stems
            # self.k_demand = self.k_demand + (cl.k_demand+ cl.Kleafdemand) * cl.stems

            self.n_demand = self.n_demand + cl.n_demand * cl.stems
            self.p_demand = self.p_demand + cl.p_demand * cl.stems
            self.k_demand = self.k_demand + cl.k_demand * cl.stems

            self.basNdemand = self.basNdemand + cl.basNdemand * cl.stems
            self.basPdemand = self.basPdemand + cl.basPdemand * cl.stems
            self.basKdemand = self.basKdemand + cl.basKdemand * cl.stems

            self.n_leaf_demand = self.n_leaf_demand + cl.Nleafdemand * cl.stems
            self.p_leaf_demand = self.p_leaf_demand + cl.Pleafdemand * cl.stems
            self.k_leaf_demand = self.k_leaf_demand + cl.Kleafdemand * cl.stems

        self.biomassgrowth = self.biomass - biomass_ini
        self.mean_diameter = (
            self.dominant.Dg * self.dominant.stems
            + self.subdominant.Dg * self.subdominant.stems
            + self.under.Dg * self.under.stems
        ) / (self.dominant.stems + self.subdominant.stems + self.under.stems)

    def assimilate(self, photopara, forc, wt, afp):
        """
        Runs the photosyntheis function for all canopy layers
        Calls canopy layer instances
        First it arranges the canopy layers into height order, and calculates the above leaf area

        Parameters
        ----------
        forc : TYPE   pandas dataframe
            DESCRIPTION. year-long measured daily weather variables
        wt : TYPE   pandas dataframe
            DESCRIPTION. simulated water tables, shape: days, ncols
        afp : TYPE pandas dataframe
            DESCRIPTION. air-filled porosity of rooting zone, shape days, ncols

        Returns
        -------
        None.

        """
        # specieswise specific leaf area: 1 Scots pine, 2:Norway spruce, 3: Birc h
        # assimilates each canopy layer and updates allometric variables in canopy layers
        heightarray = np.vstack(
            [self.dominant.hdom, self.subdominant.hdom, self.under.hdom]
        )  # array of canopy layer heights
        h_order = np.argsort(
            heightarray * -1, axis=0
        )  # sort along column in descending order, indices
        laiarray = np.vstack(
            [
                self.dominant.leafarea * self.dominant.stems,
                self.subdominant.leafarea * self.subdominant.stems,
                self.under.leafarea * self.under.stems,
            ]
        )  # array of leaf areas converterd to m2/m2

        laiout = np.zeros((3, self.n_cols))  # initialize temporary lai array
        lai_above = np.zeros(
            (4, self.n_cols)
        )  # initialize the ablove-lai array (used later in assimilation)

        for layer in range(3):  # loop through canopy layers
            order = h_order[
                layer
            ]  # inddices in heght array (descending order, 0 for the tallest)
            col = np.arange(0, self.n_cols, 1)  # indices along the strip
            laiout[layer, :] = laiarray[order, col]  # locate lai on the height order
        laiabove = np.cumsum(laiout, axis=0)  # cumulative lai sum above
        for layer in range(3):
            order = h_order[layer]
            col = np.arange(0, self.n_cols, 1)
            lai_above[layer + 1, :] = laiabove[order, col]

        self.dominant.assimilate(
            photopara,
            forc,
            wt,
            afp,
            self.previous_nut_stat,
            self.nut_stat,
            lai_above[0, :],
        )  # npp, leaf dynamics and updating the canopylayers
        self.subdominant.assimilate(
            photopara,
            forc,
            wt,
            afp,
            self.previous_nut_stat,
            self.nut_stat,
            lai_above[1, :],
        )  # npp, leaf dynamics and updating the canopylayers
        self.under.assimilate(
            photopara,
            forc,
            wt,
            afp,
            self.previous_nut_stat,
            self.nut_stat,
            lai_above[2, :],
        )  # npp, leaf dynamics and updating the canopylayers

        # updating call from the main program

    def update_nutrient_status(self, groundvegetation, N_supply, P_supply, K_supply):
        """
        Calculates nutrient status of the stand: supply/(stand demand + ground vegetation demand)
        Change in nutrient status is delayed using time delay difference function

        Parameters
        ----------
        groundvegetation : TYPE instance of groundvegetation class
            DESCRIPTION. includes the nutrient demand for the ground vegetation
        N_supply : TYPE array len(ncols)
            DESCRIPTION. N supply from decomposition, atmospheric deposition and fertilization kg/ha/yr
        P_supply : TYPE array len(ncols)
            DESCRIPTION. P supply from decomposition, atmospheric deposition and fertilization kg/ha/yr
        K_supply : TYPE array len(ncols)
            DESCRIPTION. K supply from decomposition, atmospheric deposition and fertilization kg/ha/yr

        Returns
        -------
        None.

        """
        self.previous_nut_stat = self.nut_stat.copy()

        nstat = np.ones((3, self.n_cols))
        # nstat[0,:] = N_supply / (self.n_demand + groundvegetation.nup)
        # nstat[1,:] = P_supply / (self.p_demand + groundvegetation.pup)
        # nstat[2,:] = K_supply / (self.k_demand + groundvegetation.kup)
        # nstat[0,:] = N_supply / (self.n_demand + self.basNdemand + groundvegetation.nup)
        # nstat[1,:] = P_supply / (self.p_demand + self.basPdemand + groundvegetation.pup)
        # nstat[2,:] = K_supply / (self.k_demand + self.basKdemand + groundvegetation.kup)

        # Small trees and/or sparse canopies cannot use nutrients from far-away locations.
        # The nutrient supply is limited according to the stand-density index (Reineke 1933).
        # When canopy is closed, the term = 1. It may be necessary to adjust the constant k by tree species.
        diameter = (
            self.dominant.Dg * self.dominant.stems
            + self.subdominant.Dg * self.subdominant.stems
            + self.under.Dg * self.under.stems
        ) / (self.dominant.stems + self.subdominant.stems + self.under.stems)
        # print ('Diameter', diameter)
        k = 4.35
        area_modifyer = np.clip(
            0.01, self.stems / ((diameter / 2.54) ** (-1.605) * 10**k), 1.0
        )
        # print (area_modifyer)
        # print (self.dominant.stems + self.subdominant.stems + self.under.stems)
        # area_modifyer = 1

        # denominator ie. demand corrected 050226
        nstat[0, :] = (N_supply * area_modifyer) / (
            self.n_demand + self.n_leaf_demand + groundvegetation.nup
        )
        nstat[1, :] = (P_supply * area_modifyer) / (
            self.p_demand + self.p_leaf_demand + groundvegetation.pup
        )
        nstat[2, :] = (K_supply * area_modifyer) / (
            self.k_demand + self.k_leaf_demand + groundvegetation.kup
        )
        minnstat = np.min(nstat, axis=0)

        tau = 3.0
        for c in range(self.n_cols):
            self.nut_stat[c] = self.nut_stat[c] + (minnstat[c] - self.nut_stat[c]) / tau
        self.nut_stat = np.clip(
            self.nut_stat, 0.5, 2.0
        )  # Too high nutstat increases transpiration too much

    def update_logging(self):
        for cl in self.clyrs:
            self.nonwoody_lresid = (
                self.nonwoody_lresid + cl.nonwoody_lresid
            )  # * cl.stems
            self.n_nonwoody_lresid = (
                self.n_nonwoody_lresid + cl.n_nonwoody_lresid
            )  # * cl.stems
            self.p_nonwoody_lresid = (
                self.p_nonwoody_lresid + cl.p_nonwoody_lresid
            )  # * cl.stems
            self.k_nonwoody_lresid = (
                self.k_nonwoody_lresid + cl.k_nonwoody_lresid
            )  # * cl.stems

            self.woody_lresid = self.woody_lresid + cl.woody_lresid  # * cl.stems
            self.n_woody_lresid = self.n_woody_lresid + cl.n_woody_lresid  # * cl.stems
            self.p_woody_lresid = self.p_woody_lresid + cl.p_woody_lresid  # * cl.stems
            self.k_woody_lresid = self.k_woody_lresid + cl.k_woody_lresid  # * cl.stems

            self.harvested_volume = (
                self.harvested_volume + cl.harvested_volume
            )  # harvested volume m3/ha
            self.harvested_log_volume = (
                self.harvested_log_volume + cl.harvested_log_volume
            )  # harvested saw log volume m3/ha
            self.harvested_pulp_volume = (
                self.harvested_pulp_volume + cl.harvested_pulp_volume
            )  # harvsted pulp volume m3/ha
            self.harvested_biomass = (
                self.harvested_biomass + cl.harvested_biomass
            )  # saw biomass kg/ha
            self.harvested_stems = (
                self.harvested_stems + cl.harvested_stems
            )  # number of harvested stems/ha

    def reset_logging(self):
        """
        Resets logging residue arrays after locating them to decomposition model

        Returns
        -------
        None.

        """

        self.nonwoody_lresid = self.nonwoody_lresid * 0.0
        self.n_nonwoody_lresid = self.n_nonwoody_lresid * 0.0
        self.p_nonwoody_lresid = self.p_nonwoody_lresid * 0.0
        self.k_nonwoody_lresid = self.k_nonwoody_lresid * 0.0

        self.woody_lresid = self.woody_lresid * 0.0
        self.n_woody_lresid = self.n_woody_lresid * 0.0
        self.p_woody_lresid = self.p_woody_lresid * 0.0
        self.k_woody_lresid = self.k_woody_lresid * 0.0

        self.harvested_volume = self.harvested_volume * 0.0  # harvested volume m3/ha
        self.harvested_log_volume = (
            self.harvested_log_volume * 0.0
        )  # harvested saw log volume m3/ha
        self.harvested_pulp_volume = (
            self.harvested_pulp_volume * 0.0
        )  # harvsted pulp volume m3/ha
        self.harvested_biomass = self.harvested_biomass * 0.0  # saw biomass kg/ha
        self.harvested_stems = (
            self.harvested_stems * 0.0
        )  # number of harvested stems/ha

        for cl in self.clyrs:
            cl.nonwoody_lresid = cl.nonwoody_lresid * 0.0
            cl.n_nonwoody_lresid = cl.n_nonwoody_lresid * 0.0
            cl.p_nonwoody_lresid = cl.p_nonwoody_lresid * 0.0
            cl.k_nonwoody_lresid = cl.k_nonwoody_lresid * 0.0

            cl.woody_lresid = cl.woody_lresid * 0.0
            cl.n_woody_lresid = cl.n_woody_lresid * 0.0
            cl.p_woody_lresid = cl.p_woody_lresid * 0.0
            cl.k_woody_lresid = cl.k_woody_lresid * 0.0

            cl.harvested_biomass = cl.harvested_biomass * 0.0
            cl.harvested_log_volume = cl.harvested_log_volume * 0.0
            cl.harvested_pulp_volume = cl.harvested_pulp_volume * 0.0
            cl.harvested_volume = cl.harvested_volume * 0.0
            cl.harvested_stems = cl.harvested_stems * 0.0
