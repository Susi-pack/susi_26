# Tests for Stand.apply_cutting_management — the cutting-management dispatch
# (issue #171). Scope: verify routing/dispatch only (right Canopylayer method
# called with the right args, right exceptions for not-yet-implemented
# management types). Does NOT test do_thinning()/do_clearcut()'s own
# behavior (see test_canopylayer.py for that), and does NOT go through
# Susi.run() — there's no full-simulation integration test for cutting yet
# (see the TODO in susi_main.py's ContinuousCover arm; add one once that's
# implemented, mirroring supersusi's TestClearCutIntegration in
# test_susi_main_run.py).
#
# Note: apply_cutting_management is only ever called by susi_main.py's
# `if cutting_management is not None and yr == cutting_management.application_yr:`
# guard, so it's never called with cutting_management=None in practice — no
# test for that path here (see discussion in the accompanying conversation).
from pathlib import Path

import numpy as np
import pytest

from susi.core.stand import Stand
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
    ClearCut,
    ContinuousCover,
    CuttingManagementParams,
    LocationsForPhotoParams,
    Thinning,
    get_photo_parameters_by_location,
)

DATA_DIR = Path(__file__).parent.parent / "data"
N = 5
SFC = np.ones(N, dtype=int) * 4  # reused for both Stand construction and
# apply_cutting_management's `sfc` param -- must match, since a ClearCut fits
# post-cut allometry zones from the same site fertility class array.


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
    )
    stand.update()
    return stand


def _regeneration_allometry() -> CanopyLayerAllometry:
    """age must start at 1 — see ClearCut.new_allometry_includes_age_one."""
    return CanopyLayerAllometry(
        allometry_file_registry={
            1: AllometryFileAndSpecies(
                file_path=DATA_DIR / "post_clearcut_allom.csv", species_id=1
            )
        },
        pointers={
            CanopyLayerName.dominant: [1] * N,
            CanopyLayerName.subdominant: None,
            CanopyLayerName.under: None,
        },
    )


class TestApplyCuttingManagement:
    def test_thinning_dispatches_to_dominant_cutting(self, monkeypatch):
        """Thinning routes to stand.dominant.do_thinning(yr, nut_stat=stand.nut_stat, to_ba=...)
        when only the dominant layer appears in target_basal_area."""
        stand = _make_stand()
        calls = {}

        def fake_do_thinning(yr, nut_stat, to_ba):
            calls["yr"] = yr
            calls["nut_stat"] = nut_stat
            calls["to_ba"] = to_ba

        monkeypatch.setattr(stand.dominant, "do_thinning", fake_do_thinning)

        cutting_management = CuttingManagementParams(
            application_yr=2005,
            management_type=Thinning(target_basal_area={CanopyLayerName.dominant: 12}),
        )
        stand.apply_cutting_management(cutting_management, yr=2005, sfc=SFC)

        assert calls["yr"] == 2005
        assert calls["to_ba"] == 12
        assert calls["nut_stat"] is stand.nut_stat

    def test_thinning_only_touches_layers_named_in_target_basal_area(self, monkeypatch):
        """Layers omitted from target_basal_area are left untouched (#173)."""
        stand = _make_stand()
        for layer in (stand.subdominant, stand.under):
            monkeypatch.setattr(
                layer,
                "do_thinning",
                lambda *a, **k: pytest.fail(f"{layer.name} should not be cut"),
            )
        monkeypatch.setattr(stand.dominant, "do_thinning", lambda *a, **k: None)

        cutting_management = CuttingManagementParams(
            application_yr=2005,
            management_type=Thinning(target_basal_area={CanopyLayerName.dominant: 12}),
        )
        stand.apply_cutting_management(
            cutting_management, yr=2005, sfc=SFC
        )  # should not raise

    def test_thinning_dispatches_to_multiple_named_layers(self, monkeypatch):
        """#173: the user can choose which layers to thin, and each gets its
        own target basal area."""
        stand = _make_stand()
        calls = {}

        def make_fake_do_thinning(layer_name):
            def fake_do_thinning(yr, nut_stat, to_ba):
                calls[layer_name] = {"yr": yr, "nut_stat": nut_stat, "to_ba": to_ba}

            return fake_do_thinning

        monkeypatch.setattr(
            stand.dominant, "do_thinning", make_fake_do_thinning("dominant")
        )
        monkeypatch.setattr(
            stand.subdominant, "do_thinning", make_fake_do_thinning("subdominant")
        )
        monkeypatch.setattr(
            stand.under,
            "do_thinning",
            lambda *a, **k: pytest.fail("under should not be cut"),
        )

        cutting_management = CuttingManagementParams(
            application_yr=2005,
            management_type=Thinning(
                target_basal_area={
                    CanopyLayerName.dominant: 12,
                    CanopyLayerName.subdominant: 8,
                }
            ),
        )
        stand.apply_cutting_management(cutting_management, yr=2005, sfc=SFC)

        assert calls["dominant"]["to_ba"] == 12
        assert calls["subdominant"]["to_ba"] == 8
        for layer_name in ("dominant", "subdominant"):
            assert calls[layer_name]["yr"] == 2005
            assert calls[layer_name]["nut_stat"] is stand.nut_stat

    def test_clearcut_dispatches_to_all_three_layers(self, monkeypatch):
        """Unlike Thinning, ClearCut routes to do_clearcut(yr, nut_stat=stand.nut_stat,
        strips_to_cut=..., new_zones=...) on dominant, subdominant, AND under."""
        stand = _make_stand()
        calls = {}

        def make_fake_do_clearcut(layer_name):
            def fake_do_clearcut(yr, nut_stat, strips_to_cut, new_zones):
                calls[layer_name] = {
                    "yr": yr,
                    "nut_stat": nut_stat,
                    "strips_to_cut": strips_to_cut,
                    "new_zones": new_zones,
                }

            return fake_do_clearcut

        for layer in (stand.dominant, stand.subdominant, stand.under):
            monkeypatch.setattr(layer, "do_clearcut", make_fake_do_clearcut(layer.name))

        strips_to_cut = [True] * N
        cutting_management = CuttingManagementParams(
            application_yr=2005,
            management_type=ClearCut(
                new_growth_allometry=_regeneration_allometry(),
                strips_to_cut=strips_to_cut,
            ),
        )
        stand.apply_cutting_management(cutting_management, yr=2005, sfc=SFC)

        for layer_name in ("dominant", "subdominant", "under"):
            assert calls[layer_name]["yr"] == 2005
            assert calls[layer_name]["strips_to_cut"] == strips_to_cut
            assert calls[layer_name]["nut_stat"] is stand.nut_stat

        # dominant is the only real layer here (subdominant/under pointers are
        # None), so it's the only one that should get real post-cut zones --
        # the other two get new_zones=[] since _build_zones(None, ...) == [].
        assert len(calls["dominant"]["new_zones"]) == 1
        assert calls["subdominant"]["new_zones"] == []
        assert calls["under"]["new_zones"] == []

    def test_continuous_cover_cannot_even_be_constructed(self):
        """ContinuousCover raises on construction (model-level stub), so
        Stand.apply_cutting_management's `case ContinuousCover():` arm is
        currently unreachable — there is no CuttingManagementParams instance
        that could trigger it. Documenting that fact here rather than the
        (impossible) dispatch itself."""
        with pytest.raises(NotImplementedError):
            ContinuousCover()
