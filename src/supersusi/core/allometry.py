# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 10:44:18 2022

@author: alauren
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class Params:
    species: int = field(
        doc="Tree species code (1=Pine, 2=Spruce, 3=Birch, >=4 maps to Birch)"
    )
    site_fertility_class: int = field(doc="Site fertility class (1-6)")


# === Sub-dataclasses, structure for AllometryFunctions ===


@dataclass(frozen=True)
class AgeBased:
    hdom: interp1d
    ba: interp1d
    vol: interp1d
    yield_: interp1d
    bm: interp1d
    bm_no_leaves: interp1d
    leaves: interp1d


@dataclass(frozen=True)
class BiomassToStand:
    leaf_mass: interp1d
    with_leaves_to_leaf_mass: interp1d
    lai: interp1d
    hdom: interp1d
    dg: interp1d
    yi: interp1d
    vol: interp1d
    log_vol: interp1d
    pulp_vol: interp1d
    ba: interp1d
    dbm: interp1d
    stems: interp1d


@dataclass(frozen=True)
class YieldVolume:
    yi_to_vol: interp1d
    yi_to_bm: interp1d
    vol_to_logs: interp1d
    vol_to_pulp: interp1d


@dataclass(frozen=True)
class FineRoots:
    fine_roots: interp1d
    n_fine_roots: interp1d
    p_fine_roots: interp1d
    k_fine_roots: interp1d


@dataclass(frozen=True)
class LitterMass:
    fine_root_litter: interp1d
    woody_litter: interp1d
    with_leaves_to_fine_root_litter: interp1d
    with_leaves_to_woody_litter: interp1d


@dataclass(frozen=True)
class MortalityMass:
    fine_root: interp1d
    woody: interp1d
    leaves: interp1d


@dataclass(frozen=True)
class NutrientDemand:
    n_demand: interp1d
    p_demand: interp1d
    k_demand: interp1d
    n_leaf_demand: interp1d
    p_leaf_demand: interp1d
    k_leaf_demand: interp1d


@dataclass(frozen=True)
class NutrientLitter:
    n_fine_root_litter: interp1d
    p_fine_root_litter: interp1d
    k_fine_root_litter: interp1d
    n_woody_litter: interp1d
    p_woody_litter: interp1d
    k_woody_litter: interp1d


@dataclass(frozen=True)
class NutrientMortality:
    n_mortality_leaves: interp1d
    p_mortality_leaves: interp1d
    k_mortality_leaves: interp1d
    n_mortality_fine_root: interp1d
    p_mortality_fine_root: interp1d
    k_mortality_fine_root: interp1d
    n_mortality_woody: interp1d
    p_mortality_woody: interp1d
    k_mortality_woody: interp1d


@dataclass(frozen=True)
class LoggingResidues:
    woody: interp1d
    n_woody: interp1d
    p_woody: interp1d
    k_woody: interp1d


# === High-level dataclass ===


@dataclass(frozen=True)
class AllometryFunctions:
    age_based: AgeBased
    biomass_to_stand: BiomassToStand
    yield_volume: YieldVolume
    fine_roots: FineRoots
    litter_mass: LitterMass
    mortality_mass: MortalityMass
    nutrient_demand: NutrientDemand
    nutrient_litter: NutrientLitter
    nutrient_mortality: NutrientMortality
    logging_residues: LoggingResidues


def build_allometry_interpolation_functions(
    params: Params, allometry_dataframe: pd.DataFrame
) -> AllometryFunctions:
    """
    Build allometry interpolation functions from Motti output data.

    Input:
        params: species code and site fertility class
        allometry_data: dataframe from allometry file page 1

    Out:
        ALL UNITS converted to /tree, except for number of stems, which is /ha
        interpolation functions:
            age in annual [yrs]
            age [yrs] to variables:
                ageToHdom, [m]
                ageToBa, [m2/tree]
                ageToVol, [m3/tree]
                ageToYield, [m3/tree]
                ageToBm [kg dry mass / tree]
            biomass [kg dry mass / tree] to variables:
                bmToLeafMass, [kg/tree]
                bmToLAI, [m2/m2/tree]
                bmToHdom, [m]
                bmToYi, [m3/tree]
                bmToBa, [m2/tree]
                bmToLitter, [kg/ha/tree]
                bmToStems [number/tree]
            volume or yield to variables:
                yiToVol [m3]
                yiToBm, [kg dry mass/tree]
                volToLogs, [m3/tree]
                volToPulp, [m3/tree]
                sp    species
        Biomass models in Motti (Repola 2008, 2009) have been develped for mineral soils. In peatlands the
        leaf mass is 35.5% lower. This bias is corrected in construction of the interpolation function
    Modifications needed:

        create new litter scheme: see Dec 21 esom model development
        biomass to nutrient interpolation functions: N, P, K
        nutrient to biomass interpolation functions
        litter: woody, nonwoody, locate to interpolation function
        locate interpolation functions to dictionaty
    """
    df = allometry_dataframe

    cnames = [
        "yr",
        "age",
        "N",
        "BA",
        "Hg",
        "Dg",
        "hdom",
        "vol",
        "logs",
        "pulp",
        "loss",
        "yield",
        "mortality",
        "stem",
        "stemloss",
        "branch_living",
        "branch_dead",
        "leaves",
        "stump",
        "roots_coarse",
        "roots_fine",
    ]
    species_codes = {1: "Pine", 2: "Spruce", 3: "Birch"}
    sp = params.species if params.species < 4 else 3
    spe = species_codes[sp]
    if sp == 2:
        leaf_scale = {
            1: 1.5,
            2: 1.5,
            3: 1.5,
            4: 1.5,
            5: 1.5,
            6: 1.5,
        }
    else:
        leaf_scale = {
            1: 1.0,
            2: 1.0,
            3: 1.355,
            4: 1.4,
            5: 1.45,
            6: 1.5,
        }

    row = np.zeros((np.shape(df)[1]), dtype=float)
    dfrow = pd.DataFrame([row])
    dfrow.columns = cnames
    df = pd.concat([dfrow, df], axis=0)
    nrows = df.shape[0]
    df["new_index"] = range(nrows)
    df = df.set_index("new_index")

    df.at[0, "N"] = df.at[1, "N"]

    df[
        [
            "stem",
            "branch_living",
            "branch_dead",
            "leaves",
            "stump",
            "roots_coarse",
            "roots_fine",
        ]
    ] = (
        df[
            [
                "stem",
                "branch_living",
                "branch_dead",
                "leaves",
                "stump",
                "roots_coarse",
                "roots_fine",
            ]
        ]
        * 1000.0
    )

    df[
        [
            "BA",
            "vol",
            "logs",
            "pulp",
            "loss",
            "yield",
            "mortality",
            "stem",
            "branch_living",
            "branch_dead",
            "leaves",
            "stump",
            "roots_coarse",
            "roots_fine",
        ]
    ] = (
        df[
            [
                "BA",
                "vol",
                "logs",
                "pulp",
                "loss",
                "yield",
                "mortality",
                "stem",
                "branch_living",
                "branch_dead",
                "leaves",
                "stump",
                "roots_coarse",
                "roots_fine",
            ]
        ].values
        / df[
            ["N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N"]
        ].values
    )

    a_arr = np.arange(0, max(df["age"].values), 1.0)

    nuts = {
        "Pine": {
            "Foliage": {"N": 12.5, "P": 1.3, "K": 4.25},
            "Stem": {"N": 1.17, "P": 0.08, "K": 0.45},
        },
        "Spruce": {
            "Foliage": {"N": 12.5, "P": 1.3, "K": 4.25},
            "Stem": {"N": 1.12, "P": 0.09, "K": 0.64},
        },
        "Birch": {
            "Foliage": {"N": 12.5, "P": 1.3, "K": 4.25},
            "Stem": {"N": 1.51, "P": 0.15, "K": 0.58},
        },
    }

    retrans = {"N": 0.69, "P": 0.73, "K": 0.8}
    sla = {
        "Pine": 6.8,
        "Spruce": 7.25,
        "Birch": 14.0,
    }

    longevityLeaves = {
        "Pine": 3.0,
        "Spruce": 5.0,
        "Birch": 1.0,
    }
    longevityFineRoots = {
        "Pine": 0.7,
        "Spruce": 1.0,
        "Birch": 1.0,
    }
    longevityBranch = {
        "Pine": 20.0,
        "Spruce": 20.0,
        "Birch": 20.0,
    }
    longevityCoarseRoots = {
        "Pine": 20.0,
        "Spruce": 20.0,
        "Birch": 20,
    }

    df["leaves"] = df["leaves"] / leaf_scale[params.site_fertility_class]
    df["leafarea"] = df["leaves"].values / 10000.0 * sla[spe]

    df["stem_mass"] = df["stem"]
    df["stem_and_stump"] = df[["stem_mass", "stump"]].sum(axis=1)
    df["bm"] = df[
        [
            "stem_mass",
            "branch_living",
            "branch_dead",
            "leaves",
            "stump",
            "roots_coarse",
            "roots_fine",
        ]
    ].sum(axis=1)
    df["bm_noleaves"] = df[
        [
            "stem_mass",
            "branch_living",
            "branch_dead",
            "stump",
            "roots_coarse",
            "roots_fine",
        ]
    ].sum(axis=1)
    df["N_leaves"] = df["leaves"] * nuts[spe]["Foliage"]["N"] / 1000.0
    df["Nbm_noleaves"] = (
        df[["stem_mass", "branch_living", "branch_dead", "stump", "roots_coarse"]].sum(
            axis=1
        )
        * nuts[spe]["Stem"]["N"]
        / 1000.0
        + df[["roots_fine"]].sum(axis=1) * nuts[spe]["Foliage"]["N"] / 1000.0
    )
    df["P_leaves"] = df["leaves"] * nuts[spe]["Foliage"]["P"] / 1000.0
    df["Pbm_noleaves"] = (
        df[["stem_mass", "branch_living", "branch_dead", "stump", "roots_coarse"]].sum(
            axis=1
        )
        * nuts[spe]["Stem"]["P"]
        / 1000.0
        + df[["roots_fine"]].sum(axis=1) * nuts[spe]["Foliage"]["P"] / 1000.0
    )
    df["K_leaves"] = df["leaves"] * nuts[spe]["Foliage"]["K"] / 1000.0
    df["Kbm_noleaves"] = (
        df[["stem_mass", "branch_living", "branch_dead", "stump", "roots_coarse"]].sum(
            axis=1
        )
        * nuts[spe]["Stem"]["K"]
        / 1000.0
        + df[["roots_fine"]].sum(axis=1) * nuts[spe]["Foliage"]["K"] / 1000.0
    )

    df["woody_logging_residues"] = df[
        ["branch_living", "branch_dead", "roots_coarse", "stump"]
    ].sum(axis=1)
    df["N_woody_logging_residues"] = (
        df["woody_logging_residues"] * nuts[spe]["Stem"]["N"] / 1000.0
    )
    df["P_woody_logging_residues"] = (
        df["woody_logging_residues"] * nuts[spe]["Stem"]["P"] / 1000.0
    )
    df["K_woody_logging_residues"] = (
        df["woody_logging_residues"] * nuts[spe]["Stem"]["K"] / 1000.0
    )

    df["N_fine_roots"] = df["roots_fine"] * nuts[spe]["Foliage"]["N"] / 1000.0
    df["P_fine_roots"] = df["roots_fine"] * nuts[spe]["Foliage"]["P"] / 1000.0
    df["K_fine_roots"] = df["roots_fine"] * nuts[spe]["Foliage"]["K"] / 1000.0

    df["N_leaf_demand"] = df["N_leaves"] / longevityLeaves[spe] * (1.0 - retrans["N"])
    df["P_leaf_demand"] = df["P_leaves"] / longevityLeaves[spe] * (1.0 - retrans["P"])
    df["K_leaf_demand"] = df["K_leaves"] / longevityLeaves[spe] * (1.0 - retrans["K"])

    # --- Age-based interpolation functions ---
    age_to_hdom = interp1d(
        df["age"].values,
        df["hdom"].values,
        fill_value=(df["hdom"].values[0], df["hdom"].values[-1]),
        bounds_error=True,
    )
    interp1d(
        df["age"].values,
        df["leafarea"].values,
        fill_value=(df["leafarea"].values[0], df["leafarea"].values[-1]),
        bounds_error=True,
    )
    age_to_yield = interp1d(
        df["age"].values,
        df["yield"].values,
        fill_value=(df["yield"].values[0], df["yield"].values[-1]),
        bounds_error=True,
    )
    age_to_vol = interp1d(
        df["age"].values,
        df["vol"].values,
        fill_value=(df["vol"].values[0], df["vol"].values[-1]),
        bounds_error=True,
    )
    age_to_ba = interp1d(
        df["age"].values,
        df["BA"].values,
        fill_value=(df["BA"].values[0], df["BA"].values[-1]),
        bounds_error=True,
    )
    age_to_bm = interp1d(
        df["age"].values,
        df["bm"].values,
        fill_value=(df["bm"].values[0], df["bm"].values[-1]),
        bounds_error=True,
    )
    age_to_bm_no_leaves = interp1d(
        df["age"].values,
        df["bm_noleaves"].values,
        fill_value=(df["bm_noleaves"].values[0], df["bm_noleaves"].values[-1]),
        bounds_error=True,
    )
    age_to_stems = interp1d(
        df["age"].values,
        df["N"].values,
        fill_value=(df["N"].values[0], df["N"].values[-1]),
        bounds_error=True,
    )
    age_to_leaves = interp1d(
        df["age"].values,
        df["leaves"].values,
        fill_value=(df["leaves"].values[0], df["leaves"].values[-1]),
        bounds_error=True,
    )
    age_to_fine_roots = interp1d(
        df["age"].values,
        df["roots_fine"].values,
        fill_value=(df["roots_fine"].values[0], df["roots_fine"].values[-1]),
        bounds_error=True,
    )
    age_to_branch_living = interp1d(
        df["age"].values,
        df["branch_living"].values,
        fill_value=(df["branch_living"].values[0], df["branch_living"].values[-1]),
        bounds_error=True,
    )
    age_to_branch_dead = interp1d(
        df["age"].values,
        df["branch_dead"].values,
        fill_value=(df["branch_dead"].values[0], df["branch_dead"].values[-1]),
        bounds_error=True,
    )
    age_to_coarse_roots = interp1d(
        df["age"].values,
        df["roots_coarse"].values,
        fill_value=(df["roots_coarse"].values[0], df["roots_coarse"].values[-1]),
        bounds_error=True,
    )
    age_to_stem_stump = interp1d(
        df["age"].values,
        df["stem_and_stump"].values,
        fill_value=(
            df["stem_and_stump"].values[0],
            df["stem_and_stump"].values[-1],
        ),
        bounds_error=True,
    )
    age_to_n_no_leaves = interp1d(
        df["age"].values,
        df["Nbm_noleaves"].values,
        fill_value=(df["Nbm_noleaves"].values[0], df["Nbm_noleaves"].values[-1]),
        bounds_error=True,
    )
    age_to_p_no_leaves = interp1d(
        df["age"].values,
        df["Pbm_noleaves"].values,
        fill_value=(df["Pbm_noleaves"].values[0], df["Pbm_noleaves"].values[-1]),
        bounds_error=True,
    )
    age_to_k_no_leaves = interp1d(
        df["age"].values,
        df["Kbm_noleaves"].values,
        fill_value=(df["Kbm_noleaves"].values[0], df["Kbm_noleaves"].values[-1]),
        bounds_error=True,
    )

    # --- Volume / yield interpolation functions ---
    vol_to_logs = interp1d(
        df["vol"].values,
        df["logs"].values,
        fill_value=(df["logs"].values[0], df["logs"].values[-1]),
        bounds_error=True,
    )
    vol_to_pulp = interp1d(
        df["vol"].values,
        df["pulp"].values,
        fill_value=(df["pulp"].values[0], df["pulp"].values[-1]),
        bounds_error=True,
    )

    yi_to_vol = interp1d(
        df["yield"].values,
        df["vol"].values,
        fill_value=(df["vol"].values[0], df["vol"].values[-1]),
        bounds_error=True,
    )
    yi_to_bm = interp1d(
        df["yield"].values,
        df["bm"].values,
        fill_value=(df["bm"].values[0], df["bm"].values[-1]),
        bounds_error=True,
    )

    bm_to_yi = interp1d(
        df["bm_noleaves"].values,
        df["yield"].values,
        fill_value=(df["yield"].values[0], df["yield"].values[-1]),
        bounds_error=True,
    )
    bm_to_vol = interp1d(
        df["bm_noleaves"].values,
        df["vol"].values,
        fill_value=(df["vol"].values[0], df["vol"].values[-1]),
        bounds_error=True,
    )

    bm_to_ba = interp1d(
        df["bm_noleaves"].values,
        df["BA"].values,
        fill_value=(df["BA"].values[0], df["BA"].values[-1]),
        bounds_error=True,
    )

    bm_to_leaf_mass = interp1d(
        df["bm_noleaves"].values,
        df["leaves"].values,
        fill_value=(df["leaves"].values[0], df["leaves"].values[-1]),
        bounds_error=True,
    )

    bm_with_leaves_to_leaf_mass = interp1d(
        df["bm"].values,
        df["leaves"].values,
        fill_value=(df["leaves"].values[0], df["leaves"].values[-1]),
        bounds_error=True,
    )
    bm_to_lai = interp1d(
        df["bm_noleaves"].values,
        df["leaves"].values * sla[spe] / 10000.0,
        fill_value=(
            df["leaves"].values[0] * sla[spe] / 10000.0,
            df["leaves"].values[-1] * sla[spe] / 10000.0,
        ),
        bounds_error=True,
    )
    bm_to_hdom = interp1d(
        df["bm_noleaves"].values,
        df["hdom"].values,
        fill_value=(df["hdom"].values[0], df["hdom"].values[-1]),
        bounds_error=True,
    )
    bm_to_stems = interp1d(
        df["bm_noleaves"].values,
        df["N"].values,
        fill_value=(df["N"].values[0], df["N"].values[-1]),
        bounds_error=True,
    )
    bm_to_dg = interp1d(
        df["bm_noleaves"].values,
        df["Dg"].values,
        fill_value=(df["Dg"].values[0], df["Dg"].values[-1]),
        bounds_error=True,
    )

    bm_to_fine_roots = interp1d(
        df["bm_noleaves"].values,
        df["roots_fine"].values,
        fill_value=(df["roots_fine"].values[0], df["roots_fine"].values[-1]),
        bounds_error=True,
    )
    bm_to_n_fine_roots = interp1d(
        df["bm_noleaves"].values,
        df["N_fine_roots"].values,
        fill_value=(df["N_fine_roots"].values[0], df["N_fine_roots"].values[-1]),
        bounds_error=True,
    )
    bm_to_p_fine_roots = interp1d(
        df["bm_noleaves"].values,
        df["P_fine_roots"].values,
        fill_value=(df["P_fine_roots"].values[0], df["P_fine_roots"].values[-1]),
        bounds_error=True,
    )
    bm_to_k_fine_roots = interp1d(
        df["bm_noleaves"].values,
        df["K_fine_roots"].values,
        fill_value=(df["K_fine_roots"].values[0], df["K_fine_roots"].values[-1]),
        bounds_error=True,
    )

    bm_to_woody_logging_residues = interp1d(
        df["bm_noleaves"].values,
        df["woody_logging_residues"].values,
        fill_value=(
            df["woody_logging_residues"].values[0],
            df["woody_logging_residues"].values[-1],
        ),
        bounds_error=True,
    )
    bm_to_n_woody_logging_residues = interp1d(
        df["bm_noleaves"].values,
        df["N_woody_logging_residues"].values,
        fill_value=(
            df["N_woody_logging_residues"].values[0],
            df["N_woody_logging_residues"].values[-1],
        ),
        bounds_error=True,
    )
    bm_to_p_woody_logging_residues = interp1d(
        df["bm_noleaves"].values,
        df["P_woody_logging_residues"].values,
        fill_value=(
            df["P_woody_logging_residues"].values[0],
            df["P_woody_logging_residues"].values[-1],
        ),
        bounds_error=True,
    )
    bm_to_k_woody_logging_residues = interp1d(
        df["bm_noleaves"].values,
        df["K_woody_logging_residues"].values,
        fill_value=(
            df["K_woody_logging_residues"].values[0],
            df["K_woody_logging_residues"].values[-1],
        ),
        bounds_error=True,
    )

    bm_to_log_volume = interp1d(
        df["bm_noleaves"].values,
        df["logs"].values,
        fill_value=(df["logs"].values[0], df["logs"].values[-1]),
        bounds_error=True,
    )
    bm_to_pulp_volume = interp1d(
        df["bm_noleaves"].values,
        df["pulp"].values,
        fill_value=(df["pulp"].values[0], df["pulp"].values[-1]),
        bounds_error=True,
    )

    bm_to_n_leaf_demand = interp1d(
        df["bm_noleaves"].values,
        df["N_leaf_demand"].values,
        fill_value=(df["N_leaf_demand"].values[0], df["N_leaf_demand"].values[-1]),
        bounds_error=True,
    )
    bm_to_p_leaf_demand = interp1d(
        df["bm_noleaves"].values,
        df["P_leaf_demand"].values,
        fill_value=(df["P_leaf_demand"].values[0], df["N_leaf_demand"].values[-1]),
        bounds_error=True,
    )
    bm_to_k_leaf_demand = interp1d(
        df["bm_noleaves"].values,
        df["K_leaf_demand"].values,
        fill_value=(df["K_leaf_demand"].values[0], df["N_leaf_demand"].values[-1]),
        bounds_error=True,
    )

    # ---------- Litter functions ----------
    fineroot_litter = (
        age_to_fine_roots(a_arr) / longevityFineRoots[spe] * np.gradient(a_arr)
    )
    woody_litter = (
        age_to_branch_living(a_arr) / longevityBranch[spe] * np.gradient(a_arr)
        + age_to_branch_dead(a_arr) / longevityBranch[spe] * np.gradient(a_arr)
        + age_to_coarse_roots(a_arr) / longevityCoarseRoots[spe] * np.gradient(a_arr)
    )

    mortality_fineroot = (
        -np.gradient(age_to_stems(a_arr))
        / age_to_stems(a_arr)
        * (age_to_fine_roots(a_arr))
    )
    mortality_leaf = (
        -np.gradient(age_to_stems(a_arr)) / age_to_stems(a_arr) * (age_to_leaves(a_arr))
    )
    mortality_woody = (
        -np.gradient(age_to_stems(a_arr))
        / age_to_stems(a_arr)
        * (
            age_to_branch_dead(a_arr)
            + age_to_branch_living(a_arr)
            + age_to_stem_stump(a_arr)
            + age_to_coarse_roots(a_arr)
        )
    )

    dbm = np.gradient(age_to_bm_no_leaves(a_arr)) + fineroot_litter + woody_litter

    bm_to_dbm = interp1d(
        age_to_bm_no_leaves(a_arr), dbm, fill_value=(dbm[0], dbm[-1]), bounds_error=True
    )
    bm_to_fineroot_litter = interp1d(
        age_to_bm_no_leaves(a_arr),
        fineroot_litter,
        fill_value=(fineroot_litter[0], fineroot_litter[-1]),
        bounds_error=True,
    )
    bm_to_woody_litter = interp1d(
        age_to_bm_no_leaves(a_arr),
        woody_litter,
        fill_value=(woody_litter[0], woody_litter[-1]),
        bounds_error=True,
    )
    bm_to_mortality_fine_root = interp1d(
        age_to_bm_no_leaves(a_arr),
        mortality_fineroot,
        fill_value=(mortality_fineroot[0], mortality_fineroot[-1]),
        bounds_error=True,
    )
    bm_to_mortality_woody = interp1d(
        age_to_bm_no_leaves(a_arr),
        mortality_woody,
        fill_value=(mortality_woody[0], mortality_woody[-1]),
        bounds_error=True,
    )
    bm_to_mortality_leaves = interp1d(
        age_to_bm_no_leaves(a_arr),
        mortality_leaf,
        fill_value=(mortality_leaf[0], mortality_leaf[-1]),
        bounds_error=True,
    )

    bm_with_leaves_to_fineroot_litter = interp1d(
        age_to_bm(a_arr),
        fineroot_litter,
        fill_value=(fineroot_litter[0], fineroot_litter[-1]),
        bounds_error=True,
    )
    bm_with_leaves_to_woody_litter = interp1d(
        age_to_bm(a_arr),
        woody_litter,
        fill_value=(woody_litter[0], woody_litter[-1]),
        bounds_error=True,
    )

    # ---------- Demand functions ----------
    n_fineroot_litter = (
        (1.0 - retrans["N"]) * nuts[spe]["Foliage"]["N"] / 1000.0 * fineroot_litter
    )
    p_fineroot_litter = (
        (1.0 - retrans["P"]) * nuts[spe]["Foliage"]["P"] / 1000.0 * fineroot_litter
    )
    k_fineroot_litter = (
        (1.0 - retrans["K"]) * nuts[spe]["Foliage"]["K"] / 1000.0 * fineroot_litter
    )

    n_woody_litter = (
        (1.0 - retrans["N"]) * nuts[spe]["Stem"]["N"] / 1000.0 * woody_litter
    )
    p_woody_litter = (
        (1.0 - retrans["P"]) * nuts[spe]["Stem"]["P"] / 1000.0 * woody_litter
    )
    k_woody_litter = (
        (1.0 - retrans["K"]) * nuts[spe]["Stem"]["K"] / 1000.0 * woody_litter
    )

    n_mortality_woody = mortality_woody * nuts[spe]["Stem"]["N"] / 1000.0
    p_mortality_woody = mortality_woody * nuts[spe]["Stem"]["P"] / 1000.0
    k_mortality_woody = mortality_woody * nuts[spe]["Stem"]["K"] / 1000.0

    n_mortality_leaves = mortality_leaf * nuts[spe]["Foliage"]["N"] / 1000.0
    p_mortality_leaves = mortality_leaf * nuts[spe]["Foliage"]["P"] / 1000.0
    k_mortality_leaves = mortality_leaf * nuts[spe]["Foliage"]["K"] / 1000.0

    n_mortality_fineroot = mortality_fineroot * nuts[spe]["Foliage"]["N"] / 1000.0
    p_mortality_fineroot = mortality_fineroot * nuts[spe]["Foliage"]["P"] / 1000.0
    k_mortality_fineroot = mortality_fineroot * nuts[spe]["Foliage"]["K"] / 1000.0

    n_demand = (
        np.gradient(age_to_n_no_leaves(a_arr))
        + (1.0 - retrans["N"]) * nuts[spe]["Foliage"]["N"] / 1000.0 * fineroot_litter
        + (1.0 - retrans["N"]) * nuts[spe]["Stem"]["N"] / 1000.0 * woody_litter
    )

    p_demand = (
        np.gradient(age_to_p_no_leaves(a_arr))
        + (1.0 - retrans["P"]) * nuts[spe]["Foliage"]["P"] / 1000.0 * fineroot_litter
        + (1.0 - retrans["P"]) * nuts[spe]["Stem"]["P"] / 1000.0 * woody_litter
    )

    k_demand = (
        np.gradient(age_to_k_no_leaves(a_arr))
        + (1.0 - retrans["K"]) * nuts[spe]["Foliage"]["K"] / 1000.0 * fineroot_litter
        + (1.0 - retrans["K"]) * nuts[spe]["Stem"]["K"] / 1000.0 * woody_litter
    )

    bm_to_n_demand = interp1d(
        age_to_bm_no_leaves(a_arr),
        n_demand,
        fill_value=(n_demand[0], n_demand[-1]),
        bounds_error=True,
    )
    bm_to_p_demand = interp1d(
        age_to_bm_no_leaves(a_arr),
        p_demand,
        fill_value=(p_demand[0], p_demand[-1]),
        bounds_error=True,
    )
    bm_to_k_demand = interp1d(
        age_to_bm_no_leaves(a_arr),
        k_demand,
        fill_value=(k_demand[0], k_demand[-1]),
        bounds_error=True,
    )

    bm_to_n_fine_root_litter = interp1d(
        age_to_bm_no_leaves(a_arr),
        n_fineroot_litter,
        fill_value=(n_fineroot_litter[0], n_fineroot_litter[-1]),
        bounds_error=True,
    )
    bm_to_p_fine_root_litter = interp1d(
        age_to_bm_no_leaves(a_arr),
        p_fineroot_litter,
        fill_value=(p_fineroot_litter[0], p_fineroot_litter[-1]),
        bounds_error=True,
    )
    bm_to_k_fine_root_litter = interp1d(
        age_to_bm_no_leaves(a_arr),
        k_fineroot_litter,
        fill_value=(k_fineroot_litter[0], k_fineroot_litter[-1]),
        bounds_error=True,
    )

    bm_to_n_woody_litter = interp1d(
        age_to_bm_no_leaves(a_arr),
        n_woody_litter,
        fill_value=(n_woody_litter[0], n_woody_litter[-1]),
        bounds_error=True,
    )
    bm_to_p_woody_litter = interp1d(
        age_to_bm_no_leaves(a_arr),
        p_woody_litter,
        fill_value=(p_woody_litter[0], p_woody_litter[-1]),
        bounds_error=True,
    )
    bm_to_k_woody_litter = interp1d(
        age_to_bm_no_leaves(a_arr),
        k_woody_litter,
        fill_value=(k_woody_litter[0], k_woody_litter[-1]),
        bounds_error=True,
    )

    bm_to_n_mortality_leaves = interp1d(
        age_to_bm_no_leaves(a_arr),
        n_mortality_leaves,
        fill_value=(n_mortality_leaves[0], n_mortality_leaves[-1]),
        bounds_error=True,
    )
    bm_to_p_mortality_leaves = interp1d(
        age_to_bm_no_leaves(a_arr),
        p_mortality_leaves,
        fill_value=(p_mortality_leaves[0], p_mortality_leaves[-1]),
        bounds_error=True,
    )
    bm_to_k_mortality_leaves = interp1d(
        age_to_bm_no_leaves(a_arr),
        k_mortality_leaves,
        fill_value=(k_mortality_leaves[0], k_mortality_leaves[-1]),
        bounds_error=True,
    )

    bm_to_n_mortality_fine_root = interp1d(
        age_to_bm_no_leaves(a_arr),
        n_mortality_fineroot,
        fill_value=(n_mortality_fineroot[0], n_mortality_fineroot[-1]),
        bounds_error=True,
    )
    bm_to_p_mortality_fine_root = interp1d(
        age_to_bm_no_leaves(a_arr),
        p_mortality_fineroot,
        fill_value=(p_mortality_fineroot[0], p_mortality_fineroot[-1]),
        bounds_error=True,
    )
    bm_to_k_mortality_fine_root = interp1d(
        age_to_bm_no_leaves(a_arr),
        k_mortality_fineroot,
        fill_value=(k_mortality_fineroot[0], k_mortality_fineroot[-1]),
        bounds_error=True,
    )

    bm_to_n_mortality_woody = interp1d(
        age_to_bm_no_leaves(a_arr),
        n_mortality_woody,
        fill_value=(n_mortality_woody[0], n_mortality_woody[-1]),
        bounds_error=True,
    )
    bm_to_p_mortality_woody = interp1d(
        age_to_bm_no_leaves(a_arr),
        p_mortality_woody,
        fill_value=(p_mortality_woody[0], p_mortality_woody[-1]),
        bounds_error=True,
    )
    bm_to_k_mortality_woody = interp1d(
        age_to_bm_no_leaves(a_arr),
        k_mortality_woody,
        fill_value=(k_mortality_woody[0], k_mortality_woody[-1]),
        bounds_error=True,
    )

    return AllometryFunctions(
        age_based=AgeBased(
            hdom=age_to_hdom,
            ba=age_to_ba,
            vol=age_to_vol,
            yield_=age_to_yield,
            bm=age_to_bm,
            bm_no_leaves=age_to_bm_no_leaves,
            leaves=age_to_leaves,
        ),
        biomass_to_stand=BiomassToStand(
            leaf_mass=bm_to_leaf_mass,
            with_leaves_to_leaf_mass=bm_with_leaves_to_leaf_mass,
            lai=bm_to_lai,
            hdom=bm_to_hdom,
            dg=bm_to_dg,
            yi=bm_to_yi,
            vol=bm_to_vol,
            log_vol=bm_to_log_volume,
            pulp_vol=bm_to_pulp_volume,
            ba=bm_to_ba,
            dbm=bm_to_dbm,
            stems=bm_to_stems,
        ),
        yield_volume=YieldVolume(
            yi_to_vol=yi_to_vol,
            yi_to_bm=yi_to_bm,
            vol_to_logs=vol_to_logs,
            vol_to_pulp=vol_to_pulp,
        ),
        fine_roots=FineRoots(
            fine_roots=bm_to_fine_roots,
            n_fine_roots=bm_to_n_fine_roots,
            p_fine_roots=bm_to_p_fine_roots,
            k_fine_roots=bm_to_k_fine_roots,
        ),
        litter_mass=LitterMass(
            fine_root_litter=bm_to_fineroot_litter,
            woody_litter=bm_to_woody_litter,
            with_leaves_to_fine_root_litter=bm_with_leaves_to_fineroot_litter,
            with_leaves_to_woody_litter=bm_with_leaves_to_woody_litter,
        ),
        mortality_mass=MortalityMass(
            fine_root=bm_to_mortality_fine_root,
            woody=bm_to_mortality_woody,
            leaves=bm_to_mortality_leaves,
        ),
        nutrient_demand=NutrientDemand(
            n_demand=bm_to_n_demand,
            p_demand=bm_to_p_demand,
            k_demand=bm_to_k_demand,
            n_leaf_demand=bm_to_n_leaf_demand,
            p_leaf_demand=bm_to_p_leaf_demand,
            k_leaf_demand=bm_to_k_leaf_demand,
        ),
        nutrient_litter=NutrientLitter(
            n_fine_root_litter=bm_to_n_fine_root_litter,
            p_fine_root_litter=bm_to_p_fine_root_litter,
            k_fine_root_litter=bm_to_k_fine_root_litter,
            n_woody_litter=bm_to_n_woody_litter,
            p_woody_litter=bm_to_p_woody_litter,
            k_woody_litter=bm_to_k_woody_litter,
        ),
        nutrient_mortality=NutrientMortality(
            n_mortality_leaves=bm_to_n_mortality_leaves,
            p_mortality_leaves=bm_to_p_mortality_leaves,
            k_mortality_leaves=bm_to_k_mortality_leaves,
            n_mortality_fine_root=bm_to_n_mortality_fine_root,
            p_mortality_fine_root=bm_to_p_mortality_fine_root,
            k_mortality_fine_root=bm_to_k_mortality_fine_root,
            n_mortality_woody=bm_to_n_mortality_woody,
            p_mortality_woody=bm_to_p_mortality_woody,
            k_mortality_woody=bm_to_k_mortality_woody,
        ),
        logging_residues=LoggingResidues(
            woody=bm_to_woody_logging_residues,
            n_woody=bm_to_n_woody_logging_residues,
            p_woody=bm_to_p_woody_logging_residues,
            k_woody=bm_to_k_woody_logging_residues,
        ),
    )
