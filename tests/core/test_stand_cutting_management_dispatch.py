# Tests for Stand.apply_cutting_management — the cutting-management dispatch
# (issue #171). Scope: verify routing/dispatch only (right Canopylayer method
# called with the right args, right exceptions for not-yet-implemented
# management types). Does NOT test Canopylayer.cutting()'s own behavior, and
# does NOT go through Susi.run() — there's no full-simulation integration
# test for cutting yet (see the TODO in susi_main.py's ClearCut/ContinuousCover
# arms; add one once those are implemented, mirroring supersusi's
# TestClearCutIntegration in test_susi_main_run.py).
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
    AllometryParams,
    CanopyLayerAllometryPointers,
    ClearCut,
    ContinuousCover,
    CuttingManagementParams,
    LocationsForPhotoParams,
    Thinning,
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


def _regeneration_allometry() -> AllometryParams:
    """age must start at 1 — see ClearCut.new_allometry_includes_age_one."""
    return AllometryParams(
        allometry_dir_path=DATA_DIR,
        dominant={1: "post_clearcut_allom.xlsx"},
        subdominant={0: "post_clearcut_allom.xlsx"},
        under={0: "post_clearcut_allom.xlsx"},
    )


class TestApplyCuttingManagement:
    def test_thinning_dispatches_to_dominant_cutting(self, monkeypatch):
        """Thinning routes to stand.dominant.cutting(yr, nut_stat=stand.nut_stat, to_ba=...)."""
        stand = _make_stand()
        calls = {}

        def fake_cutting(yr, nut_stat, to_ba):
            calls["yr"] = yr
            calls["nut_stat"] = nut_stat
            calls["to_ba"] = to_ba

        monkeypatch.setattr(stand.dominant, "cutting", fake_cutting)

        cutting_management = CuttingManagementParams(
            application_yr=2005, management_type=Thinning(to_ba=12)
        )
        stand.apply_cutting_management(cutting_management, yr=2005)

        assert calls["yr"] == 2005
        assert calls["to_ba"] == 12
        assert calls["nut_stat"] is stand.nut_stat

    def test_thinning_does_not_touch_subdominant_or_under(self, monkeypatch):
        """Documents current (issue-flagged) behavior: only dominant is cut."""
        stand = _make_stand()
        for layer in (stand.subdominant, stand.under):
            monkeypatch.setattr(
                layer,
                "cutting",
                lambda *a, **k: pytest.fail(f"{layer.name} should not be cut"),
            )
        monkeypatch.setattr(stand.dominant, "cutting", lambda *a, **k: None)

        cutting_management = CuttingManagementParams(
            application_yr=2005, management_type=Thinning(to_ba=12)
        )
        stand.apply_cutting_management(cutting_management, yr=2005)  # should not raise

    def test_clearcut_raises_not_implemented(self):
        stand = _make_stand()
        cutting_management = CuttingManagementParams(
            application_yr=2005,
            management_type=ClearCut(
                new_growth_allometry=_regeneration_allometry(),
                strips_to_cut=[True] * N,
            ),
        )
        with pytest.raises(NotImplementedError):
            stand.apply_cutting_management(cutting_management, yr=2005)

    def test_continuous_cover_cannot_even_be_constructed(self):
        """ContinuousCover raises on construction (model-level stub), so
        Stand.apply_cutting_management's `case ContinuousCover():` arm is
        currently unreachable — there is no CuttingManagementParams instance
        that could trigger it. Documenting that fact here rather than the
        (impossible) dispatch itself."""
        with pytest.raises(NotImplementedError):
            ContinuousCover()
