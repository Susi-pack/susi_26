# Tests for Canopylayer.do_clearcut (issue #171, and issue #181's allometry
# switch-over). Scope: verify the strips_to_cut boolean mask correctly
# partitions columns within a single allometry zone into "cut" (residues +
# harvest populated, age reset to 1, biomass recomputed under new_zones'
# allometry, flow/derived fields reset to zero) vs "uncut" (completely
# untouched).
#
# new_zones is a required parameter of do_clearcut (#181): every clearcut
# must say what allometry cut columns regrow under. Most tests below don't
# care about an actual allometry-file switch -- they're exercising cutting
# mechanics only -- so they use _same_allometry_new_zones() to reuse each
# pre-cut zone's own allometry, reproducing do_clearcut's pre-#181 behavior
# explicitly rather than via a hidden default. TestDoClearcutAllometrySwitchover
# is the one class that actually exercises a real allometry-file switch.
#
# TestDoClearcutGroupBFields.test_group_b_fields_untouched_for_uncut_columns
# pins down the fix for a bug that used to exist here: do_clearcut used to
# call initialize_domain() unscoped over the whole layer, silently zeroing
# NPP/leaf-litter/demand/etc. for columns that were never cut. do_clearcut
# is now correctly scoped to cut_cols, and this test passes.
from pathlib import Path
from typing import ClassVar

import numpy as np

from susi.core.canopylayer import Canopylayer, Zone
from susi.core.stand import Stand, _build_zones
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
    LocationsForPhotoParams,
    get_photo_parameters_by_location,
)

DATA_DIR = Path(__file__).parent.parent / "data"
N = 5
SFC = np.ones(N, dtype=int) * 4

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
        allometry_file_registry={
            1: AllometryFileAndSpecies(
                file_path=DATA_DIR / "test_allometry.csv", species_id=1
            )
        },
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
        sfc=SFC,
        agearr=agearr,
        allometry_params=allometry_params,
        photopara=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data")
        ),
        growth_mode="dynamic",
    )
    stand.update()
    return stand


def _same_allometry_new_zones(
    layer: Canopylayer, strips_to_cut: list[bool]
) -> list[Zone]:
    """new_zones that reuse each pre-cut zone's own allometry, scoped to
    that zone's cut columns -- for tests that only exercise do_clearcut's
    cutting mechanics, not an actual allometry-file switch. Reproduces
    do_clearcut's pre-#181 behavior explicitly."""
    strips_to_cut_arr = np.asarray(strips_to_cut, dtype=bool)
    return [
        Zone(id=zone.id, cols=cut_cols, allometry=zone.allometry)
        for zone in layer.zones
        for cut_cols in [zone.cols[strips_to_cut_arr[zone.cols]]]
        if cut_cols.size
    ]


class TestDoClearcutStripSelectivity:
    STRIPS_TO_CUT = STRIPS_TO_CUT
    CUT_COLS = CUT_COLS
    UNCUT_COLS = UNCUT_COLS

    def test_only_cut_columns_get_age_reset(self):
        stand = _make_stand()
        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=self.STRIPS_TO_CUT,
            new_zones=_same_allometry_new_zones(stand.dominant, self.STRIPS_TO_CUT),
        )
        assert (stand.dominant.agearr[self.CUT_COLS] == 1.0).all()
        assert (stand.dominant.agearr[self.UNCUT_COLS] == 70.0).all()

    def test_uncut_columns_are_completely_untouched(self):
        stand = _make_stand()
        biomass_before = stand.dominant.biomass.copy()
        stems_before = stand.dominant.stems.copy()

        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=self.STRIPS_TO_CUT,
            new_zones=_same_allometry_new_zones(stand.dominant, self.STRIPS_TO_CUT),
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
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=self.STRIPS_TO_CUT,
            new_zones=_same_allometry_new_zones(stand.dominant, self.STRIPS_TO_CUT),
        )

        # Regrowth is a fresh age-1 stand — biomass must actually change.
        # This test's new_zones deliberately reuse the *same* allometry data
        # as before the cut (see _same_allometry_new_zones) -- it's only
        # checking the age-1 reset, not an allometry-file switch (see
        # TestDoClearcutAllometrySwitchover for that).
        assert (
            stand.dominant.biomass[self.CUT_COLS] != biomass_before[self.CUT_COLS]
        ).all()

    def test_residues_populated_only_for_cut_columns(self):
        stand = _make_stand()
        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=self.STRIPS_TO_CUT,
            new_zones=_same_allometry_new_zones(stand.dominant, self.STRIPS_TO_CUT),
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
        full_cut = [True] * N
        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=full_cut,
            new_zones=_same_allometry_new_zones(stand.dominant, full_cut),
        )
        assert (stand.dominant.agearr == 1.0).all()

    def test_no_strip_cut_is_a_noop(self):
        """strips_to_cut=[False]*n: nothing should change, and it must not crash."""
        stand = _make_stand()
        agearr_before = stand.dominant.agearr.copy()
        biomass_before = stand.dominant.biomass.copy()

        no_cut = [False] * N
        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=no_cut,
            new_zones=_same_allometry_new_zones(stand.dominant, no_cut),
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
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=self.STRIPS_TO_CUT,
            new_zones=_same_allometry_new_zones(stand.dominant, self.STRIPS_TO_CUT),
        )

        np.testing.assert_array_equal(stand.subdominant.agearr, subdominant_before)
        np.testing.assert_array_equal(stand.under.agearr, under_before)


class TestDoClearcutHarvestFields:
    """do_clearcut now actually populates harvested_* via _compute_harvest
    (previously a TODO/known gap) — mirrors the residues contract above."""

    def test_harvest_populated_only_for_cut_columns(self):
        stand = _make_stand()
        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=STRIPS_TO_CUT,
            new_zones=_same_allometry_new_zones(stand.dominant, STRIPS_TO_CUT),
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
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=STRIPS_TO_CUT,
            new_zones=_same_allometry_new_zones(stand.dominant, STRIPS_TO_CUT),
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
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=STRIPS_TO_CUT,
            new_zones=_same_allometry_new_zones(stand.dominant, STRIPS_TO_CUT),
        )

        for field in GROUP_B_FIELDS:
            arr = getattr(stand.dominant, field)
            assert (arr[UNCUT_COLS] == self.PRESET_VALUE).all(), (
                f"{field} should be untouched for uncut columns, got {arr[UNCUT_COLS]}"
            )


class TestDoClearcutAllometrySwitchover:
    """Issue #181: cut columns must actually regrow under the post-clearcut
    allometry file (new_zones), not silently keep using the pre-cut zone's
    own allometry -- and must keep using it in every later growth year too,
    not just at the moment of cutting."""

    def _new_zones(self, cut_cols_global: np.ndarray) -> list[Zone]:
        """Build new_zones straight from a CanopyLayerAllometry, exactly the
        way Stand.apply_cutting_management does in production."""
        new_growth_allometry = CanopyLayerAllometry(
            allometry_file_registry={
                1: AllometryFileAndSpecies(
                    file_path=DATA_DIR / "post_clearcut_allom.csv", species_id=1
                )
            },
            pointers={
                CanopyLayerName.dominant: [1] * len(cut_cols_global),
                CanopyLayerName.subdominant: None,
                CanopyLayerName.under: None,
            },
        )
        return _build_zones(
            new_growth_allometry.pointers[CanopyLayerName.dominant],
            cut_cols_global,
            new_growth_allometry.zones_data,
            new_growth_allometry.zones_species_id,
            SFC,
        )

    def test_cut_columns_pick_up_the_new_allometrys_curves(self):
        stand = _make_stand()
        old_zone = stand.dominant.zones[0]  # pre-cut zone, test_allometry.csv
        new_zones = self._new_zones(np.array(CUT_COLS))

        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=STRIPS_TO_CUT,
            new_zones=new_zones,
        )

        # Ground truth: the *new* allometry's own age-to-biomass curve at
        # age 1 -- not the old zone's.
        expected = new_zones[0].allometry.functions.age_to_bm(np.array([1.0]))[0]
        np.testing.assert_allclose(stand.dominant.biomass[CUT_COLS], expected)

        # And it must actually differ from what the *old* allometry would
        # have given at age 1 -- otherwise this wouldn't be exercising a
        # real switch (test_allometry.csv and post_clearcut_allom.csv are
        # different growth-and-yield tables).
        old_bm_at_age_1 = old_zone.allometry.functions.age_to_bm(np.array([1.0]))[0]
        assert not np.isclose(expected, old_bm_at_age_1)

    def test_self_zones_correctly_repartitioned(self):
        stand = _make_stand()
        new_zones = self._new_zones(np.array(CUT_COLS))

        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=STRIPS_TO_CUT,
            new_zones=new_zones,
        )

        # Uncut columns still belong to a zone using the pre-cut allometry;
        # cut columns now belong to (one of) new_zones.
        cols_by_zone = {
            tuple(sorted(zone.cols.tolist())): zone for zone in stand.dominant.zones
        }
        assert tuple(sorted(UNCUT_COLS)) in cols_by_zone
        assert tuple(sorted(CUT_COLS)) in cols_by_zone
        assert cols_by_zone[tuple(sorted(CUT_COLS))].allometry is new_zones[0].allometry

    def test_new_allometry_still_used_in_a_later_growth_year(self):
        """Regression test for the bug this design avoids: without
        repartitioning self.zones, a later update() call would silently
        recompute cut columns from the *pre-cut* allometry's curves again."""
        stand = _make_stand()
        new_zones = self._new_zones(np.array(CUT_COLS))

        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=STRIPS_TO_CUT,
            new_zones=new_zones,
        )

        # Advance one growth year via the same path assimilate() uses.
        bm_after = stand.dominant.biomass + 0.5  # pretend some growth happened
        stand.dominant.update(bm_after)

        # stems is recomputed from biomass via zone.allometry.functions --
        # if the cut columns were still (silently) attached to the pre-cut
        # zone, this would use the *old* allometry's bm_to_stems curve
        # instead of the new one's.
        expected_stems = new_zones[0].allometry.functions.bm_to_stems(
            bm_after[CUT_COLS]
        )
        np.testing.assert_allclose(stand.dominant.stems[CUT_COLS], expected_stems)

    def test_strip_cut_supports_multiple_disjoint_post_cut_zones(self):
        """Two cut columns, mapped to two different post-cut zones from two
        different allometry files -- pointers are interpreted as one entry
        per cut column, in strip order."""
        stand = _make_stand()
        cut_cols_global = np.array(CUT_COLS)  # [0, 2]

        new_growth_allometry = CanopyLayerAllometry(
            allometry_file_registry={
                1: AllometryFileAndSpecies(
                    file_path=DATA_DIR / "post_clearcut_allom.csv", species_id=1
                ),
                2: AllometryFileAndSpecies(
                    file_path=DATA_DIR / "test_allometry.csv", species_id=1
                ),
            },
            pointers={
                # col 0 -> zone 1, col 2 -> zone 2 (order matches strip order
                # of True entries in STRIPS_TO_CUT, i.e. CUT_COLS itself).
                CanopyLayerName.dominant: [1, 2],
                CanopyLayerName.subdominant: None,
                CanopyLayerName.under: None,
            },
        )
        new_zones = _build_zones(
            new_growth_allometry.pointers[CanopyLayerName.dominant],
            cut_cols_global,
            new_growth_allometry.zones_data,
            new_growth_allometry.zones_species_id,
            SFC,
        )
        assert len(new_zones) == 2
        assert {z.id for z in new_zones} == {1, 2}
        assert list(new_zones[0].cols) == [CUT_COLS[0]]
        assert list(new_zones[1].cols) == [CUT_COLS[1]]

        stand.dominant.do_clearcut(
            yr=2005,
            nut_stat=stand.nut_stat,
            strips_to_cut=STRIPS_TO_CUT,
            new_zones=new_zones,
        )

        assert (stand.dominant.agearr[CUT_COLS] == 1.0).all()
        assert len(stand.dominant.zones) == 3  # 1 surviving pre-cut + 2 new
        # The two post-cut zones' distinct allometry files must show up as
        # distinct biomass for their respective columns.
        assert not np.isclose(
            stand.dominant.biomass[CUT_COLS[0]], stand.dominant.biomass[CUT_COLS[1]]
        )


class TestMultiZoneCanopylayer:
    """Issue #193.

    `Canopylayer.sfc` used to be a per-column array (`self.sfc =
    sfc.copy()`) that the per-zone loop overwrote with a bare
    `int(np.median(...))` on its first iteration. A second real zone in
    the same layer would then do `self.sfc[self.ixs[ncanopy]]` on an int
    -> TypeError. Zero existing config anywhere in the repo configured
    more than one non-zero zone id per layer, so this path was never
    exercised.

    The `Zone` consolidation (#189) removed the shared `self.sfc` field
    outright: each zone now computes a local `zone_sfc` in `Stand.__init__`
    and consumes it immediately, never writing it back anywhere shared.
    This test exercises a layer with 2 real zone ids directly against
    that current shape: it doesn't crash, and (more than that) each
    zone's allometry is actually fitted from that zone's own sfc, not
    another zone's.
    """

    N = 5
    ZONE1_COLS: ClassVar[list[int]] = [0, 1]
    ZONE2_COLS: ClassVar[list[int]] = [2, 3, 4]
    # dominant layer has 2 real zone ids (1 and 2) instead of the single
    # zone every other test/config in this repo uses.
    POINTERS: ClassVar[list[int]] = [1, 1, 2, 2, 2]
    # zone 1's columns all have sfc=1, zone 2's all have sfc=4 -- distinct
    # medians per zone, so the two zones must fit distinct allometries.
    SFC = np.array([1, 1, 4, 4, 4])

    def _make_multizone_stand(self) -> Stand:
        allometry_params = CanopyLayerAllometry(
            # Same underlying file registered under two different zone
            # ids: both zones are the same species/growth-and-yield data,
            # differing only in sfc.
            allometry_file_registry={
                1: AllometryFileAndSpecies(
                    file_path=DATA_DIR / "test_allometry.csv", species_id=1
                ),
                2: AllometryFileAndSpecies(
                    file_path=DATA_DIR / "test_allometry.csv", species_id=1
                ),
            },
            pointers={
                CanopyLayerName.dominant: self.POINTERS,
                CanopyLayerName.subdominant: None,
                CanopyLayerName.under: None,
            },
        )
        agearr = {
            "dominant": np.full(self.N, 70.0),
            "subdominant": np.full(self.N, 70.0),
            "under": np.full(self.N, 70.0),
        }
        return Stand(
            n_scenarios=1,
            n_yrs=4,
            n_cols=self.N,
            sfc=self.SFC,
            agearr=agearr,
            allometry_params=allometry_params,
            photopara=get_photo_parameters_by_location(
                location=LocationsForPhotoParams("All_data")
            ),
            growth_mode="dynamic",
        )

    def test_construction_does_not_crash_with_multiple_zones(self):
        # Regression test for the TypeError described above: a second
        # real zone in the same layer must not blow up at construction.
        self._make_multizone_stand()

    def test_each_zone_built_from_its_own_sfc(self):
        stand = self._make_multizone_stand()

        # test_allometry.csv is Pine (species id 1). Pine's leaf_scale is
        # 1.0 at sfc=1 and 1.4 at sfc=4 (allometry.py's allometry_development:
        # `df["leaves"] / leaf_scale[sfc]`), and both zones share every other
        # input (species, growth-and-yield table, age). So if each zone
        # really was fitted from its own sfc rather than the other zone's
        # (or an int-overwritten shared field), zone 1's leafmass must come
        # out ~1.4x zone 2's leafmass at the same age -- not equal, and not
        # crashed/garbage.
        leafmass = stand.dominant.leafmass
        zone1_leafmass = leafmass[self.ZONE1_COLS]
        zone2_leafmass = leafmass[self.ZONE2_COLS]

        assert (zone1_leafmass > 0).all()
        assert (zone2_leafmass > 0).all()
        # Same age within each zone -> same leafmass within that zone.
        np.testing.assert_allclose(zone1_leafmass, zone1_leafmass[0])
        np.testing.assert_allclose(zone2_leafmass, zone2_leafmass[0])

        ratio = zone1_leafmass[0] / zone2_leafmass[0]
        np.testing.assert_allclose(ratio, 1.4, rtol=1e-6)
