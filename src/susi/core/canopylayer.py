# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:37:43 2022

@author: alauren
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from susi.core.allometry import Allometry
from susi.core.susi_utils import assimilation_yr


@dataclass
class Zone:
    """One allometry zone's runtime state within a Canopylayer.

    Built once per {canopy layer, allometry zone id} pair in `Stand.__init__`.
    Then passed to `Canopylayer` as a ready-made list.
    Replaces the old `self.nlyrs`/`self.ixs`/`self.allodic` triple of parallel containers
    keyed by zone id (see issue #189): `id`/`cols`/`allometry` here are what
    those three used to hold, kept together instead of cross-referenced by a
    shared key.
    """

    id: int
    cols: np.ndarray  # plain 1D column-index array (not a np.where() tuple)
    allometry: Allometry  # fitted allometry_development() result


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
        zones: list[Zone],
        agearr,
        photopara,
        nut_stat,
    ):
        self.name = (
            name  # name of the canopy layer e.g. 'dominant', 'subdominant', etc.
        )
        self.zones = zones  # allometry zones along the strip, already fitted; see Zone
        self.ncols = ncols  # number of columns in the strip
        self.agearr = agearr.copy()  # age of the canopy layer, yrs
        self.nscens = nscens  # number of scenarion in the simulation
        self.yrs = yrs  # number od years in the simulation
        self.remaining_share = np.ones(
            self.ncols
        )  # share of remaining stems after thinning 0...1

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

    def _recompute_structure_from_age(
        self, zone: Zone, target_cols, age, nut_stat
    ) -> None:
        """Recompute stand-structure fields — biomass, stems, basal area,
        hdom, Dg, leaf area/mass, species, volume, N/P/K leaf demand — for
        `target_cols` in allometry `zone`, purely as a function of `age`.

        `age` is passed in explicitly (rather than this method reading
        `self.agearr[target_cols]` itself) so each call site states plainly
        what age it's recomputing at: initialize_domain passes whatever
        agearr it was given, do_clearcut passes age=1 for the columns it
        just cut.
        """
        self.biomass[target_cols] = zone.allometry.functions.age_to_bm(
            age
        )  # stem biomass from age kg/tree
        self.stems[target_cols] = (
            zone.allometry.functions.bm_to_stems(self.biomass[target_cols])
            * self.remaining_share[target_cols]
        )  # number of stems /ha from the biomass of tree

        self.basalarea[target_cols] = zone.allometry.functions.age_to_ba(
            age
        )  # stem basal area from age m2/tree
        self.hdom[target_cols] = zone.allometry.functions.age_to_hdom(
            age
        )  # dominant height m
        self.Dg[target_cols] = zone.allometry.functions.bm_to_dg(
            self.biomass[target_cols]
        )  # mean diameter height m

        self.leafarea[target_cols] = (
            zone.allometry.functions.bm_to_lai(self.biomass[target_cols])
            * nut_stat[target_cols]
        )  # one sided LAI from the stem biomass M2/m2/tree
        self.leafmass[target_cols] = (
            zone.allometry.functions.age_to_leaves(age) * nut_stat[target_cols]
        )
        self.species[target_cols] = zone.allometry.sp
        # self.volume[target_cols] = zone.allometry.functions.age_to_vol(age)         # stem volume m3/tree
        self.volume[target_cols] = zone.allometry.functions.bm_to_vol(
            self.biomass[target_cols]
        )  # stem volume m3/tree

        leaf_demand = zone.allometry.functions.leaf_demand
        for target, curve in (
            (self.basNdemand, leaf_demand.N),
            (self.basPdemand, leaf_demand.P),
            (self.basKdemand, leaf_demand.K),
        ):
            target[target_cols] = curve(self.biomass[target_cols])

    def initialize_domain(self, agearr, nut_stat):
        """(Re)derive the entire layer from scratch, for every column.

        Called at construction and once per scenario (in Stand.reset_domain).
        Unlike do_clearcut(), there's no "preserve
        some columns" concern here: every column gets the same treatment.
        """
        self.agearr = agearr.copy()
        self.remaining_share = np.ones(self.ncols)  # fresh stand: nothing thinned yet

        for zone in self.zones:
            # Target soil columns = all columns in this zone
            target_cols = zone.cols

            # Must be recomputed before the harvest/residue
            # calls below: _compute_harvest looks up wood density by
            # self.species[target_cols], which this call sets.
            self._recompute_structure_from_age(
                zone,
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
                zone=zone, target_cols=target_cols, removed_stems=zero_stems
            )
            self._compute_harvest(
                zone=zone, target_cols=target_cols, removed_stems=zero_stems
            )

    def update(self, bm):
        # bm in kg in a mean stem
        """CHANGE all units here into /tree, Do we need remaining share?"""
        # ------------Update all variables with new biomass--------------------------------------
        for zone in self.zones:
            cols = zone.cols
            f = zone.allometry.functions
            # print ('**********************')
            # print (np.round(np.mean(f.bm_to_vol(bm[cols])*self.stems),2))
            # print (np.round(np.mean(f.age_to_vol(self.agearr[cols])*self.stems), 2))
            # print ('vol')
            # print (self.volume)
            # print ('n stems')
            # print (self.stems)
            # print ('**********************')

            self.stems[cols] = f.bm_to_stems(bm[cols]) * self.remaining_share[cols]

            self.basalarea[cols] = f.bm_to_ba(bm[cols])
            self.biomass[cols] = bm[cols]
            self.hdom[cols] = f.bm_to_hdom(bm[cols])
            self.Dg[cols] = f.bm_to_dg(bm[cols])

            self.leafarea[cols] = f.bm_to_lai(bm[cols])
            self.leafmass[cols] = f.bm_to_leaf_mass(bm[cols])
            # self.volume[cols] = f.bm_to_yi(bm[cols])
            self.volume[cols] = f.bm_to_vol(bm[cols])
            for target, curve in (
                (self.n_demand, f.demand.N),
                (self.p_demand, f.demand.P),
                (self.k_demand, f.demand.K),
            ):
                target[cols] = curve(bm[cols])
            self.logvolume[cols] = f.vol_to_logs(self.volume[cols])
            self.finerootlitter[cols] = f.bm_to_fineroot_litter(bm[cols])
            for target, curve in (
                (self.n_finerootlitter, f.fineroot_litter.N),
                (self.p_finerootlitter, f.fineroot_litter.P),
                (self.k_finerootlitter, f.fineroot_litter.K),
            ):
                target[cols] = curve(bm[cols])

            self.nonwoodylitter[cols] = (
                self.finerootlitter[cols] + self.leaf_litter[cols]
            )  # nonwoody litter kg/tree/yr
            self.n_nonwoodylitter[cols] = (
                self.n_finerootlitter[cols] + self.Nleaf_litter[cols]
            )  # N in nonwoody litter kg/tree/yr
            self.p_nonwoodylitter[cols] = (
                self.p_finerootlitter[cols] + self.Pleaf_litter[cols]
            )  # P in nonwoody litter kg/tree/yr
            self.k_nonwoodylitter[cols] = (
                self.k_finerootlitter[cols] + self.Kleaf_litter[cols]
            )  # K in nonwoody litter kg/tree/yr

            self.pulpvolume[cols] = f.vol_to_pulp(self.volume[cols])
            self.woodylitter[cols] = f.bm_to_woody_litter(bm[cols])
            for target, curve in (
                (self.n_woodylitter, f.woody_litter.N),
                (self.p_woodylitter, f.woody_litter.P),
                (self.k_woodylitter, f.woody_litter.K),
            ):
                target[cols] = curve(bm[cols])

            self.woody_litter_mort[cols] = f.bm_to_mortality_woody(bm[cols])
            for target, curve in (
                (self.n_woody_litter_mort, f.mortality_woody.N),
                (self.p_woody_litter_mort, f.mortality_woody.P),
                (self.k_woody_litter_mort, f.mortality_woody.K),
            ):
                target[cols] = curve(bm[cols])

            self.non_woody_litter_mort[cols] = f.bm_to_mortality_fine_root(
                bm[cols]
            ) + f.bm_to_mortality_leaves(bm[cols])
            for target, curve_fineroot, curve_leaves in (
                (
                    self.n_non_woody_litter_mort,
                    f.mortality_fineroot.N,
                    f.mortality_leaves.N,
                ),
                (
                    self.p_non_woody_litter_mort,
                    f.mortality_fineroot.P,
                    f.mortality_leaves.P,
                ),
                (
                    self.k_non_woody_litter_mort,
                    f.mortality_fineroot.K,
                    f.mortality_leaves.K,
                ),
            ):
                target[cols] = curve_fineroot(bm[cols]) + curve_leaves(bm[cols])

            self.yi[cols] = f.bm_to_yi(bm[cols])

            for target, curve in (
                (self.basNdemand, f.leaf_demand.N),
                (self.basPdemand, f.leaf_demand.P),
                (self.basKdemand, f.leaf_demand.K),
            ):
                target[cols] = curve(bm[cols])
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

        for zone in self.zones:
            cols = zone.cols
            (
                self.new_lmass[cols],
                self.leaf_litter[cols],
                self.C_consumption[cols],
                self.leafmax[cols],
                self.leafmin[cols],
                self.Nleafdemand[cols],
                self.Nleaf_litter[cols],
                self.N_leaf[cols],
                self.Pleafdemand[cols],
                self.Pleaf_litter[cols],
                self.P_leaf[cols],
                self.Kleafdemand[cols],
                self.Kleaf_litter[cols],
                self.K_leaf[cols],
                self.leafarea[cols],
            ) = self.leaf_dynamics(
                bm[cols],
                bm_increment[cols],
                current_leafmass[cols],
                previous_nut_stat[cols],
                nut_stat[cols],
                self.agearr[cols],
                zone.allometry.functions,
                self.species[cols],
                printOpt=False,
            )

            self.finerootlitter[cols] = zone.allometry.functions.bm_to_fineroot_litter(
                bm[cols]
            )
            self.woodylitter[cols] = zone.allometry.functions.bm_to_woody_litter(
                bm[cols]
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
            print (np.round(np.mean(self.allodic[1].functions.bm_to_vol(bm)*self.stems),2))
            #print (np.round(np.mean(self.allodic[1].functions.age_to_vol(self.agearr)*self.stems), 2))
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
        functions,
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
        leafbase0 = functions.bm_to_leaf_mass(
            bm
        )  # table growth leaf mass in the beginning of timestep
        leafbase1 = functions.bm_to_leaf_mass(
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

    def _compute_residues(self, zone: Zone, target_cols, removed_stems) -> None:
        """
        Helper for do_thinning(), do_clearcut(), and initialize_domain().
        - for thinning:
            - target_cols=zone.cols (all columns are thinned).
            - removed_stems= fraction of stems depending on objective basal area (to_ba).
        - for clear cutting:
            - target_cols=columns selected for strip cutting.
            - removed_stems= all stems
        - for initialize_domain (construction / new scenario):
            - target_cols=zone.cols (the whole zone).
            - removed_stems=0 for every column — there's no harvest event
              here, and since every term below is a product against
              removed_stems[target_cols], this naturally yields zero for
              every residue field without any separate zeroing code.

        Modifies:
        nonwoody_lresid, n/p/k_nonwoody_lresid, woody_lresid, n/p/k_woody_lresid
        """
        f = zone.allometry.functions

        # stemwise fineroot biomass multipled by number of cut stems
        self.nonwoody_lresid[target_cols] = (
            self.new_lmass[target_cols] + f.bm_to_fine_roots(self.biomass[target_cols])
        ) * removed_stems[target_cols]

        for target, leaf, curve in (
            (self.n_nonwoody_lresid, self.N_leaf, f.fine_roots.N),
            (self.p_nonwoody_lresid, self.P_leaf, f.fine_roots.P),
            (self.k_nonwoody_lresid, self.K_leaf, f.fine_roots.K),
        ):
            target[target_cols] = (
                leaf[target_cols] + curve(self.biomass[target_cols])
            ) * removed_stems[target_cols]

        self.woody_lresid[target_cols] = (
            f.bm_to_woody_logging_residues(self.biomass[target_cols])
            * removed_stems[target_cols]
        )
        for target, curve in (
            (self.n_woody_lresid, f.woody_logging_residues.N),
            (self.p_woody_lresid, f.woody_logging_residues.P),
            (self.k_woody_lresid, f.woody_logging_residues.K),
        ):
            target[target_cols] = (
                curve(self.biomass[target_cols]) * removed_stems[target_cols]
            )

    def _compute_harvest(self, zone: Zone, target_cols, removed_stems) -> None:
        """
        Helper for do_thinning(), do_clearcut(), and initialize_domain().
        - for thinning:
            - target_cols=zone.cols (all columns are thinned).
            - removed_stems= fraction of stems depending on objective basal area (to_ba).
        - for clear cutting:
            - target_cols=columns selected for strip cutting.
            - removed_stems= all stems
        - for initialize_domain (construction / new scenario):
            - target_cols=zone.cols (the whole zone).
            - removed_stems=0 for every column, same reasoning as
              _compute_residues above. Note self.species[target_cols] must
              already be set (by _recompute_structure_from_age) before this
              runs — the wood-density lookup below indexes by species id.

        Modifies:
        harvested_volume, harvested_log_volume, harvested_pulp_volume,
        harvested_biomass, harvested_stems
        """
        f = zone.allometry.functions
        wood_density = {1: 420.0, 2: 400.0, 3: 450.0}
        wood_density_node = [wood_density[s] for s in self.species[target_cols]]

        self.harvested_volume[target_cols] = (
            f.bm_to_vol(self.biomass[target_cols]) * removed_stems[target_cols]
        )  # harvested volume m3/tree
        self.harvested_log_volume[target_cols] = (
            f.bm_to_log_vol(self.biomass[target_cols]) * removed_stems[target_cols]
        )  # harvested saw log volume m3/tree
        self.harvested_pulp_volume[target_cols] = (
            f.bm_to_pulp_vol(self.biomass[target_cols]) * removed_stems[target_cols]
        )  # harvsted pulp volume m3/tree
        self.harvested_biomass[target_cols] = (
            (
                f.bm_to_log_vol(self.biomass[target_cols])
                + f.bm_to_pulp_vol(self.biomass[target_cols])
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

        for zone in self.zones:
            cols = zone.cols
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

            # share of stems remaining
            self.remaining_share[cols] = to_ba / (
                zone.allometry.functions.bm_to_ba(self.biomass[cols]) * self.stems[cols]
            )

            print("remaining share")
            print(np.mean(self.remaining_share))
            # agearr = self.agearr

            print("nonwoody logging residues")
            print(
                np.mean(self.new_lmass[cols] * cut_stems[cols])
            )  # leaf logging residues
            print(
                np.mean(
                    zone.allometry.functions.bm_to_fine_roots(self.biomass[cols])
                    * cut_stems[cols]
                )
            )  # fine root logging residues

            self._compute_residues(zone=zone, target_cols=cols, removed_stems=cut_stems)

            print("nonwoodylogging resids after adding")
            print(np.mean(self.nonwoody_lresid[cols]))

            print("woody logging residues")
            print(np.mean(self.woody_lresid))

            self._compute_harvest(zone=zone, target_cols=cols, removed_stems=cut_stems)

            print("harvested logs", np.mean(self.harvested_log_volume))
            print("harvested pulp", np.mean(self.harvested_pulp_volume))
            print("harvested biomass", np.mean(self.harvested_biomass))

        self.update(self.biomass)
        """This update to stand or to main????? """

    def do_clearcut(
        self,
        yr: int,
        nut_stat: np.ndarray,
        strips_to_cut: list[bool],
        new_zones: list[Zone],
    ) -> None:
        """Unit here /ha

        `new_zones` are the allometry zones cut columns are switched onto
        (issue #181): already fitted from the post-clearcut allometry file(s)
        and scoped to just the columns each applies to. Every real clearcut
        must supply these -- there's no "keep the old allometry" fallback.
        """
        # OBS! All cutting is taken from uniformly from the canopy layer
        # You can locate cutting also to subdominant or lower suppressed canopy layer

        strips_to_cut_arr = np.asarray(
            strips_to_cut, dtype=bool
        )  # shape (n_cols,), True==cut; False==don't cut.

        for zone in self.zones:
            zone_cols = zone.cols  # all columns belonging to this allometry zone
            # subset actually cut this call:
            cut_cols = zone_cols[strips_to_cut_arr[zone_cols]]

            # Harvest/residues first: these read self.biomass/self.stems/
            # self.new_lmass/etc., which at this point still describe the
            # felled, mature stand about to be replaced below. Computing
            # them after the age reset would compute residues for a
            # sapling that hasn't grown anything yet. This must use the
            # zone's own (pre-cut) allometry, since it's valuing what's
            # being felled -- not what gets planted next.
            self._compute_residues(
                zone=zone, target_cols=cut_cols, removed_stems=self.stems
            )
            self._compute_harvest(
                zone=zone, target_cols=cut_cols, removed_stems=self.stems
            )

        for zone in new_zones:
            cut_cols = zone.cols  # already scoped to just the columns this zone plants

            # Replace cut_cols with a fresh, unthinned, age-1 stand under
            # the new post-clearcut allometry.
            self.agearr[cut_cols] = 1.0
            self.remaining_share[cut_cols] = (
                1.0  # Nothing thinned for the newly grown saplings yet
            )
            self._recompute_structure_from_age(
                zone,
                target_cols=cut_cols,
                age=self.agearr[cut_cols],
                nut_stat=nut_stat,
            )
            # A brand-new sapling hasn't been through a growth cycle yet
            self._reset_growth_cycle_fields(target_cols=cut_cols)

        # Every column ends up owned by exactly one zone below:
        # - cut columns move to their new post-cut zone (`new_zones`, already
        #   scoped to just those columns);
        # - uncut columns stay in their existing pre-cut zone -- we just have
        #   to drop the (now-departed) cut columns from that zone's own
        #   column list, and drop the zone entirely if nothing is left in it.
        # Without this, update()/assimilate() in later years would keep
        # growing the regrown columns under the *pre-cut* allometry.
        was_cut = strips_to_cut_arr  # length n_cols; True = this column was just cut

        surviving_zones = []
        for zone in self.zones:
            cols_still_growing_here = [col for col in zone.cols if not was_cut[col]]
            if cols_still_growing_here:
                surviving_zones.append(
                    Zone(
                        id=zone.id,
                        cols=np.array(cols_still_growing_here),
                        allometry=zone.allometry,
                    )
                )

        self.zones = surviving_zones + new_zones

        print("+        cutting in " + self.name + " year " + str(yr))
