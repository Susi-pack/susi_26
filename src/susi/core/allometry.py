# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 10:44:18 2022

@author: alauren
"""

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

T = TypeVar("T")


@dataclass(frozen=True)
class PerNutrient(Generic[T]):
    """One value per macronutrient, for the ~15 places `Allometry` fits the
    same curve shape three times over — once per nutrient. Replaces three
    independently-spelled fields (e.g. `bm_to_n_demand`/`bm_to_p_demand`/
    `bm_to_k_demand`) with one `PerNutrient[Callable]` field, so a call site
    needing all three loops over `("N", "P", "K")` instead of hand-repeating
    the same statement three times.
    """

    N: T
    P: T
    K: T


@dataclass(frozen=True)
class AllometryFunctions:
    """One fitted allometry file's interpolation functions, one field per curve.

    Built once by `Allometry.allometry_development()` and read from
    `Canopylayer` (`zone.allometry.functions.<field>(...)`). Replaces the old
    `allometry_f` string-keyed dict: a typo in a dict key used to fail at
    runtime with a `KeyError` inside a numeric loop; a typo in a field name
    now fails at authoring time as an `AttributeError` any linter catches.
    Same curves, same values — this is a structural regrouping (a mechanical
    key -> attribute rename, then folding N/P/K triplicates into
    `PerNutrient` fields), not a behavior change.
    """

    # age [yrs] -> ...
    age_to_hdom: Callable[[np.ndarray], np.ndarray]
    age_to_ba: Callable[[np.ndarray], np.ndarray]
    age_to_vol: Callable[[np.ndarray], np.ndarray]
    age_to_yield: Callable[[np.ndarray], np.ndarray]
    age_to_bm: Callable[[np.ndarray], np.ndarray]
    age_to_bm_no_leaves: Callable[[np.ndarray], np.ndarray]
    age_to_leaves: Callable[[np.ndarray], np.ndarray]

    # biomass [kg dry mass/tree] -> ...
    bm_to_leaf_mass: Callable[[np.ndarray], np.ndarray]
    bm_with_leaves_to_leaf_mass: Callable[[np.ndarray], np.ndarray]
    bm_to_lai: Callable[[np.ndarray], np.ndarray]
    bm_to_hdom: Callable[[np.ndarray], np.ndarray]
    bm_to_dg: Callable[[np.ndarray], np.ndarray]
    bm_to_yi: Callable[[np.ndarray], np.ndarray]
    bm_to_vol: Callable[[np.ndarray], np.ndarray]
    bm_to_log_vol: Callable[[np.ndarray], np.ndarray]
    bm_to_pulp_vol: Callable[[np.ndarray], np.ndarray]
    bm_to_ba: Callable[[np.ndarray], np.ndarray]
    bm_to_dbm: Callable[[np.ndarray], np.ndarray]
    bm_to_stems: Callable[[np.ndarray], np.ndarray]

    # volume/yield -> ...
    yi_to_vol: Callable[[np.ndarray], np.ndarray]
    yi_to_bm: Callable[[np.ndarray], np.ndarray]
    vol_to_logs: Callable[[np.ndarray], np.ndarray]
    vol_to_pulp: Callable[[np.ndarray], np.ndarray]

    # litter and mortality
    bm_to_fineroot_litter: Callable[[np.ndarray], np.ndarray]
    bm_to_woody_litter: Callable[[np.ndarray], np.ndarray]
    bm_to_mortality_fine_root: Callable[[np.ndarray], np.ndarray]
    bm_to_mortality_woody: Callable[[np.ndarray], np.ndarray]
    bm_to_mortality_leaves: Callable[[np.ndarray], np.ndarray]
    bm_with_leaves_to_fineroot_litter: Callable[[np.ndarray], np.ndarray]
    bm_with_leaves_to_woody_litter: Callable[[np.ndarray], np.ndarray]

    # N/P/K demand — was bm_to_n_demand/bm_to_p_demand/bm_to_k_demand
    demand: PerNutrient[Callable[[np.ndarray], np.ndarray]]

    # N/P/K fine-root litter — was bm_to_{n,p,k}_fine_root_litter
    fineroot_litter: PerNutrient[Callable[[np.ndarray], np.ndarray]]

    # N/P/K woody litter — was bm_to_{n,p,k}_woody_litter
    woody_litter: PerNutrient[Callable[[np.ndarray], np.ndarray]]

    # N/P/K leaf mortality — was bm_to_{n,p,k}_mortality_leaves
    mortality_leaves: PerNutrient[Callable[[np.ndarray], np.ndarray]]

    # N/P/K fine-root mortality — was bm_to_{n,p,k}_mortality_fine_root
    mortality_fineroot: PerNutrient[Callable[[np.ndarray], np.ndarray]]

    # N/P/K woody mortality — was bm_to_{n,p,k}_mortality_woody
    mortality_woody: PerNutrient[Callable[[np.ndarray], np.ndarray]]

    # woody logging residues, and N/P/K — was bm_to_{n,p,k}_woody_logging_residues
    bm_to_woody_logging_residues: Callable[[np.ndarray], np.ndarray]
    woody_logging_residues: PerNutrient[Callable[[np.ndarray], np.ndarray]]

    # fine roots, and N/P/K — was bm_to_{n,p,k}_fine_roots
    bm_to_fine_roots: Callable[[np.ndarray], np.ndarray]
    fine_roots: PerNutrient[Callable[[np.ndarray], np.ndarray]]

    # N/P/K leaf demand — was bm_to_{n,p,k}_leaf_demand
    leaf_demand: PerNutrient[Callable[[np.ndarray], np.ndarray]]


class Allometry:
    def __init__(self):
        pass

    def allometry_development(self, df, sp, sfc):
        """
        Input:
            df: dataframe from allometry file page 1
            sp: species number id from allometry file page 2
            sfc:
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
        sp = sp if sp < 4 else 3
        spe = species_codes[sp]
        # leaf_scale ={1: 1.1, 2: 1.2, 3: 1.355, 4:1.4, 5: 1.5, 6: 1.6 }    # scales the leaf mass down from mineral soil, key is the site fertility class
        if sp == 2:
            leaf_scale = {
                1: 1.5,
                2: 1.5,
                3: 1.5,
                4: 1.5,
                5: 1.5,
                6: 1.5,
            }  # scales the leaf mass down from mineral soil, key is the site fertility class
        else:
            leaf_scale = {
                1: 1.0,
                2: 1.0,
                3: 1.355,
                4: 1.4,
                5: 1.45,
                6: 1.5,
            }  # scales the leaf mass down from mineral soil, key is the site fertility class

        # ----modify the data frame, include age = 0-------------------------
        row = np.zeros((np.shape(df)[1]), dtype=float)
        dfrow = pd.DataFrame([row])
        dfrow.columns = cnames
        df = pd.concat([dfrow, df], axis=0)
        nrows = df.shape[0]
        df["new_index"] = range(nrows)
        df = df.set_index("new_index")

        df.at[0, "N"] = df.at[1, "N"]  # modify stem number at 0 yrs
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
        )  # unit conversion for all biomass components tn/ha -> kg/ha

        # Convert all to variables to values/stem

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

        # **********************************************************************
        # Volume and biomass variables are expressed as per mean stem: this is valid if the canopy layer diameter range is small ecough
        # attn, attn, attn, attn, attn, attn, attn, attn, attn, attn, attn,
        # **********************************************************************
        a_arr = np.arange(
            0, max(df["age"].values), 1.0
        )  # stand age array from 0 to max in Motti simulation, time step year

        # ---Nutrient concentrations in tree biomass components: Palviainen & Finer 2012 Eur J For Res 131: 945-964
        # *********** Parameters *********************************************
        # ---Concentrations in mg/g
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

        retrans = {"N": 0.69, "P": 0.73, "K": 0.8}  # Nieminen Helmisaari 1996 Tree Phys
        sla = {
            "Pine": 6.8,
            "Spruce": 7.25,
            "Birch": 14.0,
        }  # one-sided specific leaf area Härkönen et al. 2015 BER 20, 181-195

        # longevityLeaves = {'Pine':2.0, 'Spruce':4., 'Birch':1.}                    # yrs, life span of leaves and fine roots
        longevityLeaves = {
            "Pine": 3.0,
            "Spruce": 5.0,
            "Birch": 1.0,
        }  # yrs, life span of leaves and fine roots
        longevityFineRoots = {
            "Pine": 0.7,
            "Spruce": 1.0,
            "Birch": 1.0,
        }  # Yuan & Chen 2010, turnover 1.07 times per year
        # longevityBranch ={'Pine':22., 'Spruce':22., 'Birch':22.}                  # Pine Mäkinen 1999
        # longevityCoarseRoots ={'Pine':22., 'Spruce':22., 'Birch':22.}             # assumption as branches
        # longevityBranch ={'Pine':15., 'Spruce':20., 'Birch':20.}                   # Pine Mäkinen 1999
        # longevityCoarseRoots ={'Pine':15., 'Spruce':20., 'Birch':20}               # assumption as branches
        longevityBranch = {
            "Pine": 20.0,
            "Spruce": 20.0,
            "Birch": 20.0,
        }  # Pine Mäkinen 1999
        longevityCoarseRoots = {
            "Pine": 20.0,
            "Spruce": 20.0,
            "Birch": 20,
        }  # assumption as branches

        # ********** Interpolation data ****************************************
        df["leaves"] = (
            df["leaves"] / leaf_scale[sfc]
        )  # adjusting to peatland sites (Data: Hannu Hökkä 2022)
        df["leafarea"] = (
            df["leaves"].values / 10000.0 * sla[spe]
        )  # ATTN This is now leves / tree -> requires *N/1000            # leaf area index m2 m-2
        # df['leafarea'] = df['leaves'].values * df['N'] /10000. * sla[spe]         #ATTN This is now leves / tree -> requires *N/1000            # leaf area index m2 m-2
        # used in ageToLAI

        df["stem_mass"] = df[
            "stem"
        ]  # df['yield'] * rho[spe]                        # stem biomass
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
            df[
                ["stem_mass", "branch_living", "branch_dead", "stump", "roots_coarse"]
            ].sum(axis=1)
            * nuts[spe]["Stem"]["N"]
            / 1000.0
            + df[["roots_fine"]].sum(axis=1) * nuts[spe]["Foliage"]["N"] / 1000.0
        )
        df["P_leaves"] = df["leaves"] * nuts[spe]["Foliage"]["P"] / 1000.0
        df["Pbm_noleaves"] = (
            df[
                ["stem_mass", "branch_living", "branch_dead", "stump", "roots_coarse"]
            ].sum(axis=1)
            * nuts[spe]["Stem"]["P"]
            / 1000.0
            + df[["roots_fine"]].sum(axis=1) * nuts[spe]["Foliage"]["P"] / 1000.0
        )
        df["K_leaves"] = df["leaves"] * nuts[spe]["Foliage"]["K"] / 1000.0
        df["Kbm_noleaves"] = (
            df[
                ["stem_mass", "branch_living", "branch_dead", "stump", "roots_coarse"]
            ].sum(axis=1)
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

        df["N_leaf_demand"] = (
            df["N_leaves"] / longevityLeaves[spe] * (1.0 - retrans["N"])
        )
        df["P_leaf_demand"] = (
            df["P_leaves"] / longevityLeaves[spe] * (1.0 - retrans["P"])
        )
        df["K_leaf_demand"] = (
            df["K_leaves"] / longevityLeaves[spe] * (1.0 - retrans["K"])
        )

        # ********** Interpolation functions ******************************************
        ageToHdom = interp1d(
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
        ageToYield = interp1d(
            df["age"].values,
            df["yield"].values,
            fill_value=(df["yield"].values[0], df["yield"].values[-1]),
            bounds_error=True,
        )
        ageToVol = interp1d(
            df["age"].values,
            df["vol"].values,
            fill_value=(df["vol"].values[0], df["vol"].values[-1]),
            bounds_error=True,
        )
        ageToBa = interp1d(
            df["age"].values,
            df["BA"].values,
            fill_value=(df["BA"].values[0], df["BA"].values[-1]),
            bounds_error=True,
        )
        ageToBm = interp1d(
            df["age"].values,
            df["bm"].values,
            fill_value=(df["bm"].values[0], df["bm"].values[-1]),
            bounds_error=True,
        )
        ageToBmNoLeaves = interp1d(
            df["age"].values,
            df["bm_noleaves"].values,
            fill_value=(df["bm_noleaves"].values[0], df["bm_noleaves"].values[-1]),
            bounds_error=True,
        )
        ageToStems = interp1d(
            df["age"].values,
            df["N"].values,
            fill_value=(df["N"].values[0], df["N"].values[-1]),
            bounds_error=True,
        )
        ageToLeaves = interp1d(
            df["age"].values,
            df["leaves"].values,
            fill_value=(df["leaves"].values[0], df["leaves"].values[-1]),
            bounds_error=True,
        )
        ageToFineRoots = interp1d(
            df["age"].values,
            df["roots_fine"].values,
            fill_value=(df["roots_fine"].values[0], df["roots_fine"].values[-1]),
            bounds_error=True,
        )
        ageToBranchLiving = interp1d(
            df["age"].values,
            df["branch_living"].values,
            fill_value=(df["branch_living"].values[0], df["branch_living"].values[-1]),
            bounds_error=True,
        )
        ageToBranchDead = interp1d(
            df["age"].values,
            df["branch_dead"].values,
            fill_value=(df["branch_dead"].values[0], df["branch_dead"].values[-1]),
            bounds_error=True,
        )
        ageToCoarseRoots = interp1d(
            df["age"].values,
            df["roots_coarse"].values,
            fill_value=(df["roots_coarse"].values[0], df["roots_coarse"].values[-1]),
            bounds_error=True,
        )
        ageToStemStump = interp1d(
            df["age"].values,
            df["stem_and_stump"].values,
            fill_value=(
                df["stem_and_stump"].values[0],
                df["stem_and_stump"].values[-1],
            ),
            bounds_error=True,
        )
        ageToNNoLeaves = interp1d(
            df["age"].values,
            df["Nbm_noleaves"].values,
            fill_value=(df["Nbm_noleaves"].values[0], df["Nbm_noleaves"].values[-1]),
            bounds_error=True,
        )
        ageToPNoLeaves = interp1d(
            df["age"].values,
            df["Pbm_noleaves"].values,
            fill_value=(df["Pbm_noleaves"].values[0], df["Pbm_noleaves"].values[-1]),
            bounds_error=True,
        )
        ageToKNoLeaves = interp1d(
            df["age"].values,
            df["Kbm_noleaves"].values,
            fill_value=(df["Kbm_noleaves"].values[0], df["Kbm_noleaves"].values[-1]),
            bounds_error=True,
        )

        volToLogs = interp1d(
            df["vol"].values,
            df["logs"].values,
            fill_value=(df["logs"].values[0], df["logs"].values[-1]),
            bounds_error=True,
        )
        volToPulp = interp1d(
            df["vol"].values,
            df["pulp"].values,
            fill_value=(df["pulp"].values[0], df["pulp"].values[-1]),
            bounds_error=True,
        )

        yiToVol = interp1d(
            df["yield"].values,
            df["vol"].values,
            fill_value=(df["vol"].values[0], df["vol"].values[-1]),
            bounds_error=True,
        )
        yiToBm = interp1d(
            df["yield"].values,
            df["bm"].values,
            fill_value=(df["bm"].values[0], df["bm"].values[-1]),
            bounds_error=True,
        )

        bmToYi = interp1d(
            df["bm_noleaves"].values,
            df["yield"].values,
            fill_value=(df["yield"].values[0], df["yield"].values[-1]),
            bounds_error=True,
        )
        bmToVol = interp1d(
            df["bm_noleaves"].values,
            df["vol"].values,
            fill_value=(df["vol"].values[0], df["vol"].values[-1]),
            bounds_error=True,
        )

        bmToBa = interp1d(
            df["bm_noleaves"].values,
            df["BA"].values,
            fill_value=(df["BA"].values[0], df["BA"].values[-1]),
            bounds_error=True,
        )

        bmToLeafMass = interp1d(
            df["bm_noleaves"].values,
            df["leaves"].values,
            fill_value=(df["leaves"].values[0], df["leaves"].values[-1]),
            bounds_error=True,
        )

        bmWithLeavesToLeafMass = interp1d(
            df["bm"].values,
            df["leaves"].values,
            fill_value=(df["leaves"].values[0], df["leaves"].values[-1]),
            bounds_error=True,
        )
        bmToLAI = interp1d(
            df["bm_noleaves"].values,
            df["leaves"].values * sla[spe] / 10000.0,
            fill_value=(
                df["leaves"].values[0] * sla[spe] / 10000.0,
                df["leaves"].values[-1] * sla[spe] / 10000.0,
            ),
            bounds_error=True,
        )
        bmToHdom = interp1d(
            df["bm_noleaves"].values,
            df["hdom"].values,
            fill_value=(df["hdom"].values[0], df["hdom"].values[-1]),
            bounds_error=True,
        )
        bmToStems = interp1d(
            df["bm_noleaves"].values,
            df["N"].values,
            fill_value=(df["N"].values[0], df["N"].values[-1]),
            bounds_error=True,
        )
        bmToDg = interp1d(
            df["bm_noleaves"].values,
            df["Dg"].values,
            fill_value=(df["Dg"].values[0], df["Dg"].values[-1]),
            bounds_error=True,
        )

        bmToFineRoots = interp1d(
            df["bm_noleaves"].values,
            df["roots_fine"].values,
            fill_value=(df["roots_fine"].values[0], df["roots_fine"].values[-1]),
            bounds_error=True,
        )
        bmToNFineRoots = interp1d(
            df["bm_noleaves"].values,
            df["N_fine_roots"].values,
            fill_value=(df["N_fine_roots"].values[0], df["N_fine_roots"].values[-1]),
            bounds_error=True,
        )
        bmToPFineRoots = interp1d(
            df["bm_noleaves"].values,
            df["P_fine_roots"].values,
            fill_value=(df["P_fine_roots"].values[0], df["P_fine_roots"].values[-1]),
            bounds_error=True,
        )
        bmToKFineRoots = interp1d(
            df["bm_noleaves"].values,
            df["K_fine_roots"].values,
            fill_value=(df["K_fine_roots"].values[0], df["K_fine_roots"].values[-1]),
            bounds_error=True,
        )

        bmToWoodyLoggingResidues = interp1d(
            df["bm_noleaves"].values,
            df["woody_logging_residues"].values,
            fill_value=(
                df["woody_logging_residues"].values[0],
                df["woody_logging_residues"].values[-1],
            ),
            bounds_error=True,
        )
        bmToNWoodyLoggingResidues = interp1d(
            df["bm_noleaves"].values,
            df["N_woody_logging_residues"].values,
            fill_value=(
                df["N_woody_logging_residues"].values[0],
                df["N_woody_logging_residues"].values[-1],
            ),
            bounds_error=True,
        )
        bmToPWoodyLoggingResidues = interp1d(
            df["bm_noleaves"].values,
            df["P_woody_logging_residues"].values,
            fill_value=(
                df["P_woody_logging_residues"].values[0],
                df["P_woody_logging_residues"].values[-1],
            ),
            bounds_error=True,
        )
        bmToKWoodyLoggingResidues = interp1d(
            df["bm_noleaves"].values,
            df["K_woody_logging_residues"].values,
            fill_value=(
                df["K_woody_logging_residues"].values[0],
                df["K_woody_logging_residues"].values[-1],
            ),
            bounds_error=True,
        )

        bmToLogVolume = interp1d(
            df["bm_noleaves"].values,
            df["logs"].values,
            fill_value=(df["logs"].values[0], df["logs"].values[-1]),
            bounds_error=True,
        )
        bmToPulpVolume = interp1d(
            df["bm_noleaves"].values,
            df["pulp"].values,
            fill_value=(df["pulp"].values[0], df["pulp"].values[-1]),
            bounds_error=True,
        )

        bmToNLeafDemand = interp1d(
            df["bm_noleaves"].values,
            df["N_leaf_demand"].values,
            fill_value=(df["N_leaf_demand"].values[0], df["N_leaf_demand"].values[-1]),
            bounds_error=True,
        )
        bmToPLeafDemand = interp1d(
            df["bm_noleaves"].values,
            df["P_leaf_demand"].values,
            fill_value=(df["P_leaf_demand"].values[0], df["N_leaf_demand"].values[-1]),
            bounds_error=True,
        )
        bmToKLeafDemand = interp1d(
            df["bm_noleaves"].values,
            df["K_leaf_demand"].values,
            fill_value=(df["K_leaf_demand"].values[0], df["N_leaf_demand"].values[-1]),
            bounds_error=True,
        )

        # **********************************************************************
        # We need demand functions for mass, N,P,K:
        #              net change + fineroot_litter + woody_litter + add foliage net demand litter from other function
        # Nutrient contents in Litterfall:
        #              fineroot litter + woody_litter + mortality_fine_root + mortality_woody + foliage litter and net demand from another function
        #              ATTN REMOVE mortality from general litter and make an own function for that.
        #                     Litter originating from trees is used later in allocation of NPP. Mortality litter messes this up.
        # Arrange:
        #              demand functions; mass, N, P, K
        #              litter functions: mass, N, P, K
        # ***********************************************************************

        """ Litter functions """
        # ---- Litter arrays------------------
        fineroot_litter = (
            ageToFineRoots(a_arr) / longevityFineRoots[spe] * np.gradient(a_arr)
        )  # unit kg / tree / yr
        woody_litter = (
            ageToBranchLiving(a_arr) / longevityBranch[spe] * np.gradient(a_arr)
            + ageToBranchDead(a_arr) / longevityBranch[spe] * np.gradient(a_arr)
            + ageToCoarseRoots(a_arr) / longevityCoarseRoots[spe] * np.gradient(a_arr)
        )  # litterfall kg/tree in timestep

        mortality_fineroot = (
            -np.gradient(ageToStems(a_arr))
            / ageToStems(a_arr)
            * (ageToFineRoots(a_arr))
        )  # unitkg / tree / yr
        mortality_leaf = (
            -np.gradient(ageToStems(a_arr)) / ageToStems(a_arr) * (ageToLeaves(a_arr))
        )
        mortality_woody = (
            -np.gradient(ageToStems(a_arr))
            / ageToStems(a_arr)
            * (
                ageToBranchDead(a_arr)
                + ageToBranchLiving(a_arr)
                + ageToStemStump(a_arr)
                + ageToCoarseRoots(a_arr)
            )
        )

        dbm = (
            np.gradient(ageToBmNoLeaves(a_arr)) + fineroot_litter + woody_litter
        )  # biomass change without leaves kg/ha/yr

        # ---- Interpolation functions -----------------
        bmToDbm = interp1d(
            ageToBmNoLeaves(a_arr), dbm, fill_value=(dbm[0], dbm[-1]), bounds_error=True
        )  # from biomass to biomass change
        bmToFinerootLitter = interp1d(
            ageToBmNoLeaves(a_arr),
            fineroot_litter,
            fill_value=(fineroot_litter[0], fineroot_litter[-1]),
            bounds_error=True,
        )
        bmToWoodyLitter = interp1d(
            ageToBmNoLeaves(a_arr),
            woody_litter,
            fill_value=(woody_litter[0], woody_litter[-1]),
            bounds_error=True,
        )
        bmToMortalityFineRoot = interp1d(
            ageToBmNoLeaves(a_arr),
            mortality_fineroot,
            fill_value=(mortality_fineroot[0], mortality_fineroot[-1]),
            bounds_error=True,
        )
        bmToMortalityLeaves = interp1d(
            ageToBmNoLeaves(a_arr),
            mortality_leaf,
            fill_value=(mortality_leaf[0], mortality_leaf[-1]),
            bounds_error=True,
        )
        bmToMortalityWoody = interp1d(
            ageToBmNoLeaves(a_arr),
            mortality_woody,
            fill_value=(mortality_woody[0], mortality_woody[-1]),
            bounds_error=True,
        )

        bmWithLeavesToFinerootLitter = interp1d(
            ageToBm(a_arr),
            fineroot_litter,
            fill_value=(fineroot_litter[0], fineroot_litter[-1]),
            bounds_error=True,
        )
        bmWithLeavesToWoodyLitter = interp1d(
            ageToBm(a_arr),
            woody_litter,
            fill_value=(woody_litter[0], woody_litter[-1]),
            bounds_error=True,
        )

        """ Demand functions """
        # -------- arrays--------------------------
        N_fineroot_litter = (
            (1.0 - retrans["N"]) * nuts[spe]["Foliage"]["N"] / 1000.0 * fineroot_litter
        )  # + mortality_fineroot*nuts[spe]['Foliage']['N'] / 1000.
        P_fineroot_litter = (
            (1.0 - retrans["P"]) * nuts[spe]["Foliage"]["P"] / 1000.0 * fineroot_litter
        )  # + mortality_fineroot*nuts[spe]['Foliage']['P'] / 1000.
        K_fineroot_litter = (
            (1.0 - retrans["K"]) * nuts[spe]["Foliage"]["K"] / 1000.0 * fineroot_litter
        )  # + mortality_fineroot*nuts[spe]['Foliage']['K'] / 1000.

        N_woody_litter = (
            (1.0 - retrans["N"]) * nuts[spe]["Stem"]["N"] / 1000.0 * woody_litter
        )  # + mortality_woody*nuts[spe]['Stem']['N'] / 1000.
        P_woody_litter = (
            (1.0 - retrans["P"]) * nuts[spe]["Stem"]["P"] / 1000.0 * woody_litter
        )  # + mortality_woody*nuts[spe]['Stem']['P'] / 1000.
        K_woody_litter = (
            (1.0 - retrans["K"]) * nuts[spe]["Stem"]["K"] / 1000.0 * woody_litter
        )  # + mortality_woody*nuts[spe]['Stem']['K'] / 1000.

        N_mortality_woody = mortality_woody * nuts[spe]["Stem"]["N"] / 1000.0
        P_mortality_woody = mortality_woody * nuts[spe]["Stem"]["P"] / 1000.0
        K_mortality_woody = mortality_woody * nuts[spe]["Stem"]["K"] / 1000.0

        N_mortality_leaves = mortality_leaf * nuts[spe]["Foliage"]["N"] / 1000.0
        P_mortality_leaves = mortality_leaf * nuts[spe]["Foliage"]["P"] / 1000.0
        K_mortality_leaves = mortality_leaf * nuts[spe]["Foliage"]["K"] / 1000.0

        N_mortality_fineroot = mortality_fineroot * nuts[spe]["Foliage"]["N"] / 1000.0
        P_mortality_fineroot = mortality_fineroot * nuts[spe]["Foliage"]["P"] / 1000.0
        K_mortality_fineroot = mortality_fineroot * nuts[spe]["Foliage"]["K"] / 1000.0

        N_demand = (
            np.gradient(ageToNNoLeaves(a_arr))
            + (1.0 - retrans["N"])
            * nuts[spe]["Foliage"]["N"]
            / 1000.0
            * fineroot_litter
            + (1.0 - retrans["N"]) * nuts[spe]["Stem"]["N"] / 1000.0 * woody_litter
        )

        P_demand = (
            np.gradient(ageToPNoLeaves(a_arr))
            + (1.0 - retrans["P"])
            * nuts[spe]["Foliage"]["P"]
            / 1000.0
            * fineroot_litter
            + (1.0 - retrans["P"]) * nuts[spe]["Stem"]["P"] / 1000.0 * woody_litter
        )

        K_demand = (
            np.gradient(ageToKNoLeaves(a_arr))
            + (1.0 - retrans["K"])
            * nuts[spe]["Foliage"]["K"]
            / 1000.0
            * fineroot_litter
            + (1.0 - retrans["K"]) * nuts[spe]["Stem"]["K"] / 1000.0 * woody_litter
        )

        # ---- Interpolation functions -----------------
        bmToNdemand = interp1d(
            ageToBmNoLeaves(a_arr),
            N_demand,
            fill_value=(N_demand[0], N_demand[-1]),
            bounds_error=True,
        )  # from biomass to N demand No leaves here
        bmToPdemand = interp1d(
            ageToBmNoLeaves(a_arr),
            P_demand,
            fill_value=(P_demand[0], P_demand[-1]),
            bounds_error=True,
        )  # from biomass to P demand
        bmToKdemand = interp1d(
            ageToBmNoLeaves(a_arr),
            K_demand,
            fill_value=(K_demand[0], K_demand[-1]),
            bounds_error=True,
        )  # from biomass to K demand

        bmToNFineRootLitter = interp1d(
            ageToBmNoLeaves(a_arr),
            N_fineroot_litter,
            fill_value=(N_fineroot_litter[0], N_fineroot_litter[-1]),
            bounds_error=True,
        )
        bmToPFineRootLitter = interp1d(
            ageToBmNoLeaves(a_arr),
            P_fineroot_litter,
            fill_value=(P_fineroot_litter[0], P_fineroot_litter[-1]),
            bounds_error=True,
        )
        bmToKFineRootLitter = interp1d(
            ageToBmNoLeaves(a_arr),
            K_fineroot_litter,
            fill_value=(K_fineroot_litter[0], K_fineroot_litter[-1]),
            bounds_error=True,
        )

        bmToNWoodyLitter = interp1d(
            ageToBmNoLeaves(a_arr),
            N_woody_litter,
            fill_value=(N_woody_litter[0], N_woody_litter[-1]),
            bounds_error=True,
        )
        bmToPWoodyLitter = interp1d(
            ageToBmNoLeaves(a_arr),
            P_woody_litter,
            fill_value=(P_woody_litter[0], P_woody_litter[-1]),
            bounds_error=True,
        )
        bmToKWoodyLitter = interp1d(
            ageToBmNoLeaves(a_arr),
            K_woody_litter,
            fill_value=(K_woody_litter[0], K_woody_litter[-1]),
            bounds_error=True,
        )

        bmToNMortalityLeaves = interp1d(
            ageToBmNoLeaves(a_arr),
            N_mortality_leaves,
            fill_value=(N_mortality_leaves[0], N_mortality_leaves[-1]),
            bounds_error=True,
        )
        bmToPMortalityLeaves = interp1d(
            ageToBmNoLeaves(a_arr),
            P_mortality_leaves,
            fill_value=(P_mortality_leaves[0], P_mortality_leaves[-1]),
            bounds_error=True,
        )
        bmToKMortalityLeaves = interp1d(
            ageToBmNoLeaves(a_arr),
            K_mortality_leaves,
            fill_value=(K_mortality_leaves[0], K_mortality_leaves[-1]),
            bounds_error=True,
        )

        bmToNMortalityFineRoot = interp1d(
            ageToBmNoLeaves(a_arr),
            N_mortality_fineroot,
            fill_value=(N_mortality_fineroot[0], N_mortality_fineroot[-1]),
            bounds_error=True,
        )
        bmToPMortalityFineRoot = interp1d(
            ageToBmNoLeaves(a_arr),
            P_mortality_fineroot,
            fill_value=(P_mortality_fineroot[0], P_mortality_fineroot[-1]),
            bounds_error=True,
        )
        bmToKMortalityFineRoot = interp1d(
            ageToBmNoLeaves(a_arr),
            K_mortality_fineroot,
            fill_value=(K_mortality_fineroot[0], K_mortality_fineroot[-1]),
            bounds_error=True,
        )

        bmToNMortalityWoody = interp1d(
            ageToBmNoLeaves(a_arr),
            N_mortality_woody,
            fill_value=(N_mortality_woody[0], N_mortality_woody[-1]),
            bounds_error=True,
        )
        bmToPMortalityWoody = interp1d(
            ageToBmNoLeaves(a_arr),
            P_mortality_woody,
            fill_value=(P_mortality_woody[0], P_mortality_woody[-1]),
            bounds_error=True,
        )
        bmToKMortalityWoody = interp1d(
            ageToBmNoLeaves(a_arr),
            K_mortality_woody,
            fill_value=(K_mortality_woody[0], K_mortality_woody[-1]),
            bounds_error=True,
        )

        self.functions = AllometryFunctions(
            age_to_hdom=ageToHdom,
            age_to_ba=ageToBa,
            age_to_vol=ageToVol,
            age_to_yield=ageToYield,
            age_to_bm=ageToBm,
            age_to_bm_no_leaves=ageToBmNoLeaves,
            age_to_leaves=ageToLeaves,
            bm_to_leaf_mass=bmToLeafMass,
            bm_with_leaves_to_leaf_mass=bmWithLeavesToLeafMass,
            bm_to_lai=bmToLAI,
            bm_to_hdom=bmToHdom,
            bm_to_dg=bmToDg,
            bm_to_yi=bmToYi,
            bm_to_vol=bmToVol,
            bm_to_log_vol=bmToLogVolume,
            bm_to_pulp_vol=bmToPulpVolume,
            bm_to_ba=bmToBa,
            bm_to_dbm=bmToDbm,
            bm_to_stems=bmToStems,
            yi_to_vol=yiToVol,
            yi_to_bm=yiToBm,
            vol_to_logs=volToLogs,
            vol_to_pulp=volToPulp,
            bm_to_fineroot_litter=bmToFinerootLitter,
            bm_to_woody_litter=bmToWoodyLitter,
            bm_to_mortality_fine_root=bmToMortalityFineRoot,
            bm_to_mortality_woody=bmToMortalityWoody,
            bm_to_mortality_leaves=bmToMortalityLeaves,
            bm_with_leaves_to_fineroot_litter=bmWithLeavesToFinerootLitter,
            bm_with_leaves_to_woody_litter=bmWithLeavesToWoodyLitter,
            demand=PerNutrient(N=bmToNdemand, P=bmToPdemand, K=bmToKdemand),
            fineroot_litter=PerNutrient(
                N=bmToNFineRootLitter, P=bmToPFineRootLitter, K=bmToKFineRootLitter
            ),
            woody_litter=PerNutrient(
                N=bmToNWoodyLitter, P=bmToPWoodyLitter, K=bmToKWoodyLitter
            ),
            mortality_leaves=PerNutrient(
                N=bmToNMortalityLeaves,
                P=bmToPMortalityLeaves,
                K=bmToKMortalityLeaves,
            ),
            mortality_fineroot=PerNutrient(
                N=bmToNMortalityFineRoot,
                P=bmToPMortalityFineRoot,
                K=bmToKMortalityFineRoot,
            ),
            mortality_woody=PerNutrient(
                N=bmToNMortalityWoody, P=bmToPMortalityWoody, K=bmToKMortalityWoody
            ),
            bm_to_woody_logging_residues=bmToWoodyLoggingResidues,
            woody_logging_residues=PerNutrient(
                N=bmToNWoodyLoggingResidues,
                P=bmToPWoodyLoggingResidues,
                K=bmToKWoodyLoggingResidues,
            ),
            bm_to_fine_roots=bmToFineRoots,
            fine_roots=PerNutrient(
                N=bmToNFineRoots, P=bmToPFineRoots, K=bmToKFineRoots
            ),
            leaf_demand=PerNutrient(
                N=bmToNLeafDemand, P=bmToPLeafDemand, K=bmToKLeafDemand
            ),
        )
        self.sp = sp
        # self.df (the fully-transformed working DataFrame) used to be kept
        # here too, but had no reader anywhere outside this method — dropped
        # as confirmed-dead state (see issue #187).
