# Tests for Canopylayer.do_clearcut's strip-selectivity (issue #171).
# Scope: verify the strips_to_cut boolean mask correctly partitions columns
# within a single allometry zone into "cut" (residues + harvest populated,
# age reset to 1, biomass recomputed, flow/derived fields reset to zero) vs
# "uncut" (completely untouched). Does NOT test post-clearcut allometry
# switch-over (new_growth_allometry isn't wired into do_clearcut yet — see
# the pending zone re-partitioning work).
#
# TestDoClearcutGroupBFields.test_group_b_fields_untouched_for_uncut_columns
# pins down the fix for a bug that used to exist here: do_clearcut used to
# call initialize_domain() unscoped over the whole layer, silently zeroing
# NPP/leaf-litter/demand/etc. for columns that were never cut. do_clearcut
# is now correctly scoped to cut_cols, and this test passes.
from pathlib import Path

import numpy as np

from susi.core.stand import Stand
from susi.io.susi_parameter_model import (
    CanopyLayerAllometry,
    CanopyLayerName,
    LocationsForPhotoParams,
    get_photo_parameters_by_location,
)

DATA_DIR = Path(__file__).parent.parent / "data"
N = 5

# cols 0, 2 cut; 1, 3, 4 left standing
STRIPS_TO_CUT = [True, False, True, False, False]
CUT_COLS = [0, 2]
UNCUT_COLS = [1, 3, 4]

# Fields initialize_domain zeroes but never recomputes from age — they're
# only ever populated by the *next* assimilate()/update() growth cycle.
# NPP, leaf dynamics, litter/mortality streams, log/pulp volume, yi,
# volumegrowth, N/P/K demand.
GROUP_B_FIELDS = (
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


def _make_stand() -> Stand:
    allometry_params = CanopyLayerAllometry(
        allometry_dir_path=DATA_DIR,
        allometry_file_registry={1: "test_allometry.xlsx"},
        pointers={
            CanopyLayerName.dominant: [1] * N,
            CanopyLayerName.subdominant: None,
            CanopyLayerName.under: None,
        },
    )
    agearr = {
        "dominant": np.full(N, 70.0),
        "subdominant": np.full(N, 70.0),
        "under": np.full(N, 70.0),
    }
    stand = Stand(
        n_scenarios=1,
        n_yrs=4,
        n_cols=N,
        sfc=np.ones(N, dtype=int) * 4,
        agearr=agearr,
        allometry_params=allometry_params,
        photopara=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data")
        ),
    )
    stand.update()
    return stand


class TestDoClearcutStripSelectivity:
    STRIPS_TO_CUT = STRIPS_TO_CUT
    CUT_COLS = CUT_COLS
    UNCUT_COLS = UNCUT_COLS

    def test_only_cut_columns_get_age_reset(self):
        stand = _make_stand()
        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=self.STRIPS_TO_CUT
        )
        assert (stand.dominant.agearr[self.CUT_COLS] == 1.0).all()
        assert (stand.dominant.agearr[self.UNCUT_COLS] == 70.0).all()

    def test_uncut_columns_are_completely_untouched(self):
        stand = _make_stand()
        biomass_before = stand.dominant.biomass.copy()
        stems_before = stand.dominant.stems.copy()

        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=self.STRIPS_TO_CUT
        )

        np.testing.assert_array_equal(
            stand.dominant.biomass[self.UNCUT_COLS], biomass_before[self.UNCUT_COLS]
        )
        np.testing.assert_array_equal(
            stand.dominant.stems[self.UNCUT_COLS], stems_before[self.UNCUT_COLS]
        )

    def test_cut_columns_biomass_recomputed_at_age_one(self):
        stand = _make_stand()
        biomass_before = stand.dominant.biomass.copy()

        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=self.STRIPS_TO_CUT
        )

        # Regrowth is a fresh age-1 stand — biomass must actually change,
        # and (until new_growth_allometry is wired in) currently comes from
        # the *same* allometry data, just evaluated at age 1.
        assert (
            stand.dominant.biomass[self.CUT_COLS] != biomass_before[self.CUT_COLS]
        ).all()

    def test_residues_populated_only_for_cut_columns(self):
        stand = _make_stand()
        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=self.STRIPS_TO_CUT
        )

        for field in (
            "nonwoody_lresid",
            "n_nonwoody_lresid",
            "p_nonwoody_lresid",
            "k_nonwoody_lresid",
            "woody_lresid",
            "n_woody_lresid",
            "p_woody_lresid",
            "k_woody_lresid",
        ):
            arr = getattr(stand.dominant, field)
            assert (arr[self.CUT_COLS] > 0).all(), (
                f"{field} should be > 0 for cut columns"
            )
            assert (arr[self.UNCUT_COLS] == 0).all(), (
                f"{field} should be 0 for uncut columns"
            )

    def test_full_strip_cut_resets_every_column(self):
        """strips_to_cut=[True]*n behaves like the old uniform clear-cut."""
        stand = _make_stand()
        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=[True] * N
        )
        assert (stand.dominant.agearr == 1.0).all()

    def test_no_strip_cut_is_a_noop(self):
        """strips_to_cut=[False]*n: nothing should change, and it must not crash."""
        stand = _make_stand()
        agearr_before = stand.dominant.agearr.copy()
        biomass_before = stand.dominant.biomass.copy()

        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=[False] * N
        )

        np.testing.assert_array_equal(stand.dominant.agearr, agearr_before)
        np.testing.assert_array_equal(stand.dominant.biomass, biomass_before)

    def test_subdominant_and_under_are_independent_of_dominant_strip_cut(self):
        """Calling do_clearcut on stand.dominant must not touch other layers
        (Stand.apply_cutting_management decides which layers get called;
        Canopylayer itself has no cross-layer awareness)."""
        stand = _make_stand()
        subdominant_before = stand.subdominant.agearr.copy()
        under_before = stand.under.agearr.copy()

        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=self.STRIPS_TO_CUT
        )

        np.testing.assert_array_equal(stand.subdominant.agearr, subdominant_before)
        np.testing.assert_array_equal(stand.under.agearr, under_before)


class TestDoClearcutHarvestFields:
    """do_clearcut now actually populates harvested_* via _compute_harvest
    (previously a TODO/known gap) — mirrors the residues contract above."""

    def test_harvest_populated_only_for_cut_columns(self):
        stand = _make_stand()
        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=STRIPS_TO_CUT
        )

        for field in (
            "harvested_volume",
            "harvested_log_volume",
            "harvested_pulp_volume",
            "harvested_biomass",
            "harvested_stems",
        ):
            arr = getattr(stand.dominant, field)
            assert (arr[CUT_COLS] > 0).all(), f"{field} should be > 0 for cut columns"
            assert (arr[UNCUT_COLS] == 0).all(), (
                f"{field} should be 0 for uncut columns"
            )


class TestDoClearcutGroupBFields:
    """A clear-cut should reset GROUP_B_FIELDS to zero for freshly-regrown
    (cut) columns — they're not recomputed from age, they're only ever
    populated by the *next* assimilate()/update() cycle, same as for a
    brand-new Canopylayer — and must leave them alone for columns that
    weren't cut.
    """

    PRESET_VALUE = 5.0

    def _stand_with_group_b_fields_set(self) -> Stand:
        stand = _make_stand()
        for field in GROUP_B_FIELDS:
            getattr(stand.dominant, field)[:] = self.PRESET_VALUE
        return stand

    def test_group_b_fields_reset_to_zero_for_cut_columns(self):
        stand = self._stand_with_group_b_fields_set()

        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=STRIPS_TO_CUT
        )

        for field in GROUP_B_FIELDS:
            arr = getattr(stand.dominant, field)
            assert (arr[CUT_COLS] == 0).all(), (
                f"{field} should be reset to 0 for cut columns (fresh age-1 "
                "regrowth hasn't grown yet — populated by next year's assimilate())"
            )

    def test_group_b_fields_untouched_for_uncut_columns(self):
        """do_clearcut used to call initialize_domain() unscoped, over the
        whole layer, so *every* clear-cut silently zeroed NPP/leaf-litter/
        demand/etc. for columns that were never cut. do_clearcut is now
        scoped to cut_cols, so uncut columns are correctly left alone.
        """
        stand = self._stand_with_group_b_fields_set()

        stand.dominant.do_clearcut(
            yr=2005, nut_stat=stand.nut_stat, strips_to_cut=STRIPS_TO_CUT
        )

        for field in GROUP_B_FIELDS:
            arr = getattr(stand.dominant, field)
            assert (arr[UNCUT_COLS] == self.PRESET_VALUE).all(), (
                f"{field} should be untouched for uncut columns, got {arr[UNCUT_COLS]}"
            )
