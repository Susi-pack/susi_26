# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:37:43 2022

@author: alauren
"""

import numpy as np
from scipy.interpolate import interp1d
from susi.core.allometry import Allometry
from susi.core.susi_utils import assimilation_yr


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
                self.sfc = int(
                    np.median(self.sfc[self.ixs[ncanopy]])
                )  # site fertility class
                # allometry instance to the dictionary
                self.allodic[ncanopy] = Allometry()
                # run the allometry; interpolation functions in the instance
                self.allodic[ncanopy].allometry_development(
                    df=allometry_df_dict[ncanopy],
                    sp=species_id_dict[ncanopy],
                    sfc=self.sfc,
                )
                self.tree_species[self.ixs[ncanopy]] = int(self.allodic[ncanopy].sp)

        # One-time, whole-layer array creation. Must run before
        # initialize_domain() (or anything else) touches self.stems,
        # self.biomass, etc. — see initial_array_allocation()'s docstring for why
        # this can never be repeated or scoped to a subset of columns.
        self.initial_array_allocation()

        # create variables and set initial values
        self.initialize_domain(agearr, nut_stat)

        print(self.name, "initialized")

    def initial_array_allocation(self) -> None:
        """Create every per-column state array as np.zeros(self.ncols), once.

        This is a ONE-TIME, WHOLE-LAYER allocation, not a "reset" — it must
        only ever be called from __init__, before self.stems/self.biomass/
        etc. exist as attributes at all. It must never be called again
        after that, and never with a target_cols subset: fancy-indexing
        into an attribute that doesn't exist yet (self.stems[target_cols])
        is exactly what used to crash here. Every later "reset to zero for
        some columns" need is handled by _reset_growth_cycle_fields (for
        growth-cycle fields) or by calling _compute_residues/_compute_harvest
        with removed_stems=0 (for harvest/residue fields) — both of those
        mutate slices of the arrays this method creates, in place.
        """
        ncols = self.ncols

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

    def _reset_growth_cycle_fields(self, target_cols) -> None:
        """Zero NPP, leaf dynamics, litter/mortality streams, log/pulp
        volume, yi, volumegrowth and N/P/K demand for target_cols.

        Unlike stand structure, none of these have an age-based formula —
        a column that hasn't been through a growth cycle yet (a brand-new
        Canopylayer, or a column that was just clear-cut back to age 1)
        unambiguously has zero NPP, zero leaf litter, etc.: there's nothing
        to "derive," only something to zero out until the next
        assimilate()/update() populates it for real.
        """
        # Fields with no age-based formula at all — they're only ever populated
        # by a real growth cycle (assimilate()/update()), never derived from
        # age the way stand structure is. So "reset" is their only operation;
        # there's no separate "recompute" counterpart the way
        # _recompute_structure_from_age is the recompute counterpart to
        # allocation. Kept as one explicit list (rather than e.g. introspecting
        # self.__dict__) so it's auditable at a glance, and so it can be the
        # single source of truth shared with tests/core/test_canopylayer.py's
        # GROUP_B_FIELDS, which pins down this exact contract.

        GROWTH_CYCLE_FIELDS = (
            "n_demand",
            "p_demand",
            "k_demand",
            "lai_above",
            "logvolume",
            "finerootlitter",
            "n_finerootlitter",
            "p_finerootlitter",
            "k_finerootlitter",
            "pulpvolume",
            "NPP",
            "NPP_pot",
            "nonwoodylitter",
            "n_nonwoodylitter",
            "p_nonwoodylitter",
            "k_nonwoodylitter",
            "volumegrowth",
            "woodylitter",
            "n_woodylitter",
            "p_woodylitter",
            "k_woodylitter",
            "yi",
            "woody_litter_mort",
            "n_woody_litter_mort",
            "p_woody_litter_mort",
            "k_woody_litter_mort",
            "non_woody_litter_mort",
            "n_non_woody_litter_mort",
            "p_non_woody_litter_mort",
            "k_non_woody_litter_mort",
            "new_lmass",
            "leaf_litter",
            "C_consumption",
            "leafmax",
            "leafmin",
            "Nleafdemand",
            "Nleaf_litter",
            "N_leaf",
            "Pleafdemand",
            "Pleaf_litter",
            "P_leaf",
            "Kleafdemand",
            "Kleaf_litter",
            "K_leaf",
        )
        for field in GROWTH_CYCLE_FIELDS:
            getattr(self, field)[target_cols] = 0.0

    def _recompute_structure_from_age(self, m, target_cols, age, nut_stat) -> None:
        """Recompute stand-structure fields — biomass, stems, basal area,
        hdom, Dg, leaf area/mass, species, volume, N/P/K leaf demand — for
        `target_cols` in allometry zone `m`, purely as a function of `age`.

        `age` is passed in explicitly (rather than this method reading
        `self.agearr[target_cols]` itself) so each call site states plainly
        what age it's recomputing at: initialize_domain passes whatever
        agearr it was given, do_clearcut passes age=1 for the columns it
        just cut.
        """
        self.biomass[target_cols] = self.allodic[m].allometry_f["ageToBm"](
            age
        )  # stem biomass from age kg/tree
        self.stems[target_cols] = (
            self.allodic[m].allometry_f["bmToStems"](self.biomass[target_cols])
            * self.remaining_share[target_cols]
        )  # number of stems /ha from the biomass of tree

        self.basalarea[target_cols] = self.allodic[m].allometry_f["ageToBa"](
            age
        )  # stem basal area from age m2/tree
        self.hdom[target_cols] = self.allodic[m].allometry_f["ageToHdom"](
            age
        )  # dominant height m
        self.Dg[target_cols] = self.allodic[m].allometry_f["bmToDg"](
            self.biomass[target_cols]
        )  # mean diameter height m

        self.leafarea[target_cols] = (
            self.allodic[m].allometry_f["bmToLAI"](self.biomass[target_cols])
            * nut_stat[target_cols]
        )  # one sided LAI from the stem biomass M2/m2/tree
        self.leafmass[target_cols] = (
            self.allodic[m].allometry_f["ageToLeaves"](age) * nut_stat[target_cols]
        )
        self.species[target_cols] = self.allodic[m].sp
        # self.volume[target_cols] = self.allodic[m].allometry_f['ageToVol'](age)         # stem volume m3/tree
        self.volume[target_cols] = self.allodic[m].allometry_f["bmToVol"](
            self.biomass[target_cols]
        )  # stem volume m3/tree

        self.basNdemand[target_cols] = self.allodic[m].allometry_f["bmToNLeafDemand"](
            self.biomass[target_cols]
        )
        self.basPdemand[target_cols] = self.allodic[m].allometry_f["bmToPLeafDemand"](
            self.biomass[target_cols]
        )
        self.basKdemand[target_cols] = self.allodic[m].allometry_f["bmToKLeafDemand"](
            self.biomass[target_cols]
        )

    def initialize_domain(self, agearr, nut_stat):
        """(Re)derive the entire layer from scratch, for every column.

        Called at construction and once per scenario (in Stand.reset_domain).
        Unlike do_clearcut(), there's no "preserve
        some columns" concern here: every column gets the same treatment.
        """
        self.agearr = agearr.copy()
        self.remaining_share = np.ones(self.ncols)  # fresh stand: nothing thinned yet
        nlyrs = self.nlyrs  # number of different canopy layers along the strip

        for m in nlyrs:
            if m > 0:
                # Target soil columns = all columns
                target_cols = self.ixs[m]

                # Must be recomputed before the harvest/residue
                # calls below: _compute_harvest looks up wood density by
                # self.species[target_cols], which this call sets.
                self._recompute_structure_from_age(
                    m,
                    target_cols=target_cols,
                    age=self.agearr[target_cols],
                    nut_stat=nut_stat,
                )
                # Growth-cycle fields (NPP, leaf dynamics, litter/mortality
                # streams, ...) have no age-formula — they're only ever
                # modified by assimilate()/update() during a real growth year
                self._reset_growth_cycle_fields(target_cols=target_cols)

                # There is no harvest event at construction/scenario-reset.
                # We here reuse the harvest/residue functions with removed_stems=0,
                # which makes every output zero instead of duplicating a
                # separate zeroing step for these fields elsewhere.
                zero_stems = np.zeros(self.ncols)
                self._compute_residues(
                    m=m, target_cols=target_cols, removed_stems=zero_stems
                )
                self._compute_harvest(
                    m=m, target_cols=target_cols, removed_stems=zero_stems
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
                    self.allodic[m].allometry_f["bmToStems"](bm[ixs[m]])
                    * self.remaining_share[ixs[m]]
                )

                self.basalarea[ixs[m]] = self.allodic[m].allometry_f["bmToBa"](
                    bm[ixs[m]]
                )
                self.biomass[ixs[m]] = bm[ixs[m]]
                self.hdom[ixs[m]] = self.allodic[m].allometry_f["bmToHdom"](bm[ixs[m]])
                self.Dg[ixs[m]] = self.allodic[m].allometry_f["bmToDg"](bm[ixs[m]])

                self.leafarea[ixs[m]] = self.allodic[m].allometry_f["bmToLAI"](
                    bm[ixs[m]]
                )
                self.leafmass[ixs[m]] = self.allodic[m].allometry_f["bmToLeafMass"](
                    bm[ixs[m]]
                )
                # self.volume[ixs[m]] = self.allodic[m].allometry_f['bmToYi'](bm[ixs[m]])
                self.volume[ixs[m]] = self.allodic[m].allometry_f["bmToVol"](bm[ixs[m]])
                self.n_demand[ixs[m]] = self.allodic[m].allometry_f["bmToNdemand"](
                    bm[ixs[m]]
                )
                self.p_demand[ixs[m]] = self.allodic[m].allometry_f["bmToPdemand"](
                    bm[ixs[m]]
                )
                self.k_demand[ixs[m]] = self.allodic[m].allometry_f["bmToKdemand"](
                    bm[ixs[m]]
                )
                self.logvolume[ixs[m]] = self.allodic[m].allometry_f["volToLogs"](
                    self.volume[ixs[m]]
                )
                self.finerootlitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToFinerootLitter"
                ](bm[ixs[m]])
                self.n_finerootlitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToNFineRootLitter"
                ](bm[ixs[m]])
                self.p_finerootlitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToPFineRootLitter"
                ](bm[ixs[m]])
                self.k_finerootlitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToKFineRootLitter"
                ](bm[ixs[m]])

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

                self.pulpvolume[ixs[m]] = self.allodic[m].allometry_f["volToPulp"](
                    self.volume[ixs[m]]
                )
                self.woodylitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToWoodyLitter"
                ](bm[ixs[m]])
                self.n_woodylitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToNWoodyLitter"
                ](bm[ixs[m]])
                self.p_woodylitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToPWoodyLitter"
                ](bm[ixs[m]])
                self.k_woodylitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToKWoodyLitter"
                ](bm[ixs[m]])

                self.woody_litter_mort[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToMortalityWoody"
                ](bm[ixs[m]])
                self.n_woody_litter_mort[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToNMortalityWoody"
                ](bm[ixs[m]])
                self.p_woody_litter_mort[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToPMortalityWoody"
                ](bm[ixs[m]])
                self.k_woody_litter_mort[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToKMortalityWoody"
                ](bm[ixs[m]])

                self.non_woody_litter_mort[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToMortalityFineRoot"
                ](bm[ixs[m]])
                +self.allodic[m].allometry_f["bmToMortalityLeaves"](bm[ixs[m]])
                self.n_non_woody_litter_mort[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToNMortalityFineRoot"
                ](bm[ixs[m]])
                +self.allodic[m].allometry_f["bmToNMortalityLeaves"](bm[ixs[m]])
                self.p_non_woody_litter_mort[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToPMortalityFineRoot"
                ](bm[ixs[m]])
                +self.allodic[m].allometry_f["bmToPMortalityLeaves"](bm[ixs[m]])
                self.k_non_woody_litter_mort[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToKMortalityFineRoot"
                ](bm[ixs[m]])
                +self.allodic[m].allometry_f["bmToKMortalityLeaves"](bm[ixs[m]])

                self.yi[ixs[m]] = self.allodic[m].allometry_f["bmToYi"](bm[ixs[m]])

                self.basNdemand[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToNLeafDemand"
                ](bm[ixs[m]])
                self.basPdemand[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToPLeafDemand"
                ](bm[ixs[m]])
                self.basKdemand[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToKLeafDemand"
                ](bm[ixs[m]])
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
                    self.allodic[m].allometry_f,
                    self.species[ixs[m]],
                    printOpt=False,
                )

                self.finerootlitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToFinerootLitter"
                ](bm[ixs[m]])
                self.woodylitter[ixs[m]] = self.allodic[m].allometry_f[
                    "bmToWoodyLitter"
                ](bm[ixs[m]])
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

        # Increment year by one
        self.agearr += 1

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
        allometry_f,
        species,
        printOpt=False,
    ):
        """
        input:
             bm, current biomass, array, kg/tree without leaves
             bm_increment, array, total NPP kg/tree
             incoming columns are of single tree species canopy layers
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
        sla = {
            "Pine": 6.8,
            "Spruce": 7.25,
            "Birch": 14.0,
        }  # specific leaf area Härkönen et al. 2015 BER 20, 181-195
        species_codes = {1: "Pine", 2: "Spruce", 3: "Birch"}
        spe = species_codes[species[0]]  # Take the first item, all are same here
        # longevityLeaves = {'Pine':[3.5, 2.5], 'Spruce':[6.0, 4.0], 'Birch':[1.0, 1.0]}                     # yrs, life span of leaves and fine roots
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
        leafbase0 = allometry_f["bmToLeafMass"](
            bm
        )  # table growth leaf mass in the beginning of timestep
        leafbase1 = allometry_f["bmToLeafMass"](
            bm + bm_increment
        )  # table growth leaf mass in the end of timestep
        leafmass = leafbase1 * nut_stat  # actual leaf mass

        # -------------------------------------------------
        leafmax = leafbase1 * 1.5
        leafmin = leafbase1 / 1.5

        net_ch = leafbase1 - leafbase0
        gr_demand = leafmass - current_leafmass

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

        LAI = new_lmass / 10000.0 * sla[spe]

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
        return (
            new_lmass,
            leaf_litter,
            C_consumption,
            leafmax,
            leafmin,
            Ndemand,
            Nleaf_litter,
            N_leaf,
            Pdemand,
            Pleaf_litter,
            P_leaf,
            Kdemand,
            Kleaf_litter,
            K_leaf,
            LAI,
        )

    def _compute_residues(self, m, target_cols, removed_stems) -> None:
        """
        Helper for do_thinning(), do_clearcut(), and initialize_domain().
        - for thinning:
            - target_cols=ixs[m] (all columns are thinned).
            - removed_stems= fraction of stems depending on objective basal area (to_ba).
        - for clear cutting:
            - target_cols=columns selected for strip cutting.
            - removed_stems= all stems
        - for initialize_domain (construction / new scenario):
            - target_cols=ixs[m] (the whole zone).
            - removed_stems=0 for every column — there's no harvest event
              here, and since every term below is a product against
              removed_stems[target_cols], this naturally yields zero for
              every residue field without any separate zeroing code.

        Modifies:
        nonwoody_lresid, n/p/k_nonwoody_lresid, woody_lresid, n/p/k_woody_lresid
        """

        # stemwise fineroot biomass multipled by number of cut stems
        self.nonwoody_lresid[target_cols] = (
            self.new_lmass[target_cols]
            + self.allodic[m].allometry_f["bmToFineRoots"](self.biomass[target_cols])
        ) * removed_stems[target_cols]

        self.n_nonwoody_lresid[target_cols] = (
            self.N_leaf[target_cols]
            + self.allodic[m].allometry_f["bmToNFineRoots"](self.biomass[target_cols])
        ) * removed_stems[target_cols]
        self.p_nonwoody_lresid[target_cols] = (
            self.P_leaf[target_cols]
            + self.allodic[m].allometry_f["bmToPFineRoots"](self.biomass[target_cols])
        ) * removed_stems[target_cols]
        self.k_nonwoody_lresid[target_cols] = (
            self.K_leaf[target_cols]
            + self.allodic[m].allometry_f["bmToKFineRoots"](self.biomass[target_cols])
        ) * removed_stems[target_cols]

        self.woody_lresid[target_cols] = (
            self.allodic[m].allometry_f["bmToWoodyLoggingResidues"](
                self.biomass[target_cols]
            )
            * removed_stems[target_cols]
        )
        self.n_woody_lresid[target_cols] = (
            self.allodic[m].allometry_f["bmToNWoodyLoggingResidues"](
                self.biomass[target_cols]
            )
            * removed_stems[target_cols]
        )
        self.p_woody_lresid[target_cols] = (
            self.allodic[m].allometry_f["bmToPWoodyLoggingResidues"](
                self.biomass[target_cols]
            )
            * removed_stems[target_cols]
        )
        self.k_woody_lresid[target_cols] = (
            self.allodic[m].allometry_f["bmToKWoodyLoggingResidues"](
                self.biomass[target_cols]
            )
            * removed_stems[target_cols]
        )

    def _compute_harvest(self, m, target_cols, removed_stems) -> None:
        """
        Helper for do_thinning(), do_clearcut(), and initialize_domain().
        - for thinning:
            - target_cols=ixs[m] (all columns are thinned).
            - removed_stems= fraction of stems depending on objective basal area (to_ba).
        - for clear cutting:
            - target_cols=columns selected for strip cutting.
            - removed_stems= all stems
        - for initialize_domain (construction / new scenario):
            - target_cols=ixs[m] (the whole zone).
            - removed_stems=0 for every column, same reasoning as
              _compute_residues above. Note self.species[target_cols] must
              already be set (by _recompute_structure_from_age) before this
              runs — the wood-density lookup below indexes by species id.

        Modifies:
        harvested_volume, harvested_log_volume, harvested_pulp_volume,
        harvested_biomass, harvested_stems
        """
        wood_density = {1: 420.0, 2: 400.0, 3: 450.0}
        wood_density_node = [wood_density[s] for s in self.species[target_cols]]

        self.harvested_volume[target_cols] = (
            self.allodic[m].allometry_f["bmToVol"](self.biomass[target_cols])
            * removed_stems[target_cols]
        )  # harvested volume m3/tree
        self.harvested_log_volume[target_cols] = (
            self.allodic[m].allometry_f["bmToLogVol"](self.biomass[target_cols])
            * removed_stems[target_cols]
        )  # harvested saw log volume m3/tree
        self.harvested_pulp_volume[target_cols] = (
            self.allodic[m].allometry_f["bmToPulpVol"](self.biomass[target_cols])
            * removed_stems[target_cols]
        )  # harvsted pulp volume m3/tree
        self.harvested_biomass[target_cols] = (
            (
                self.allodic[m].allometry_f["bmToLogVol"](self.biomass[target_cols])
                + self.allodic[m].allometry_f["bmToPulpVol"](self.biomass[target_cols])
            )
            * removed_stems[target_cols]
            * wood_density_node
        )  # saw biomass kg/tree
        self.harvested_stems[target_cols] = removed_stems[
            target_cols
        ]  # number of harvested stems/tree

    def do_thinning(self, yr: int, nut_stat: np.ndarray, to_ba: float) -> None:
        """Unit here /ha"""
        # OBS! All cutting is taken from uniformly from the canopy layer
        # You can locate cutting also to subdominant or lower suppressed canopy layer

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
                    self.allodic[m].allometry_f["bmToBa"](self.biomass) * self.stems
                )  # shate of stems remaining

                print("remaining share")
                print(np.mean(self.remaining_share))
                # agearr = self.agearr
                ixs = self.ixs

                print("nonwoody logging residues")
                print(
                    np.mean(self.new_lmass[ixs[m]] * cut_stems[ixs[m]])
                )  # leaf logging residues
                print(
                    np.mean(
                        self.allodic[m].allometry_f["bmToFineRoots"](
                            self.biomass[ixs[m]]
                        )
                        * cut_stems[ixs[m]]
                    )
                )  # fine root logging residues

                self._compute_residues(m=m, target_cols=ixs[m], removed_stems=cut_stems)

                print("nonwoodylogging resids after adding")
                print(np.mean(self.nonwoody_lresid[ixs[m]]))

                print("woody logging residues")
                print(np.mean(self.woody_lresid))

                self._compute_harvest(m=m, target_cols=ixs[m], removed_stems=cut_stems)

                print("harvested logs", np.mean(self.harvested_log_volume))
                print("harvested pulp", np.mean(self.harvested_pulp_volume))
                print("harvested biomass", np.mean(self.harvested_biomass))

                self.update(self.biomass)
                """This update to stand or to main????? """

    def do_clearcut(
        self, yr: int, nut_stat: np.ndarray, strips_to_cut: list[bool]
    ) -> None:
        """Unit here /ha"""
        # OBS! All cutting is taken from uniformly from the canopy layer
        # You can locate cutting also to subdominant or lower suppressed canopy layer

        strips_to_cut_arr = np.asarray(
            strips_to_cut, dtype=bool
        )  # shape (n_cols,), True==cut; False==don't cut.

        for m in self.nlyrs:
            if m > 0:
                zone_cols = self.ixs[m][
                    0
                ]  # all columns belonging to this allometry zone
                cut_cols = zone_cols[
                    strips_to_cut_arr[zone_cols]
                ]  # subset actually cut this call

                # Harvest/residues first: these read self.biomass/self.stems/
                # self.new_lmass/etc., which at this point still describe the
                # felled, mature stand about to be replaced below. Computing
                # them after the age reset would compute residues for a
                # sapling that hasn't grown anything yet.
                self._compute_residues(
                    m=m, target_cols=cut_cols, removed_stems=self.stems
                )
                self._compute_harvest(
                    m=m, target_cols=cut_cols, removed_stems=self.stems
                )

                # Now replace cut_cols with a fresh, unthinned, age-1 stand.
                self.agearr[cut_cols] = 1.0
                self.remaining_share[cut_cols] = (
                    1.0  # Nothing thinned for the newly grown saplings yet
                )
                self._recompute_structure_from_age(
                    m,
                    target_cols=cut_cols,
                    age=self.agearr[cut_cols],
                    nut_stat=nut_stat,
                )
                # A brand-new sapling hasn't been through a growth cycle yet
                self._reset_growth_cycle_fields(target_cols=cut_cols)

        print("+        cutting in " + self.name + " year " + str(yr))
