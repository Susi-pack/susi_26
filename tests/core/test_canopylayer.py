# Tests for Canopylayer.do_clearcut's strip-selectivity (issue #171).
# Scope: verify the strips_to_cut boolean mask correctly partitions columns
# within a single allometry zone into "cut" (residues populated, age reset
# to 1, biomass recomputed) vs "uncut" (completely untouched). Does NOT test
# post-clearcut allometry switch-over (new_growth_allometry isn't wired into
# do_clearcut yet — see the pending zone re-partitioning work) or
# harvested_volume/harvested_stems accounting (currently a known gap, marked
# TODO in do_clearcut itself).
from pathlib import Path

import numpy as np
import pytest

from susi.core.stand import Stand
from susi.io.susi_parameter_model import (
    AllometryParams,
    CanopyLayerAllometryPointers,
    LocationsForPhotoParams,
    get_photo_parameters_by_location,
)

DATA_DIR = Path(__file__).parent.parent / "data"
N = 5


def _make_stand() -> Stand:
    allometry_params = AllometryParams(
        allometry_dir_path=DATA_DIR,
        dominant={1: "test_allometry.xlsx"},
        subdominant={0: "test_allometry.xlsx"},
        under={0: "test_allometry.xlsx"},
    )
    canopylayers = CanopyLayerAllometryPointers(
        dominant=[1] * N,
        subdominant=[0] * N,
        under=[0] * N,
    )
    agearr = {
        "dominant": np.full(N, 70.0),
        "subdominant": np.full(N, 70.0),
        "under": np.full(N, 70.0),
    }
    stand = Stand(
        n_scenarios=1,
        n_yrs=4,
        canopylayers=canopylayers,
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
    # cols 0, 2 cut; 1, 3, 4 left standing
    STRIPS_TO_CUT = [True, False, True, False, False]
    CUT_COLS = [0, 2]
    UNCUT_COLS = [1, 3, 4]

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
            assert (arr[self.CUT_COLS] > 0).all(), f"{field} should be > 0 for cut columns"
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
