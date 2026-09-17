# Model-level tests for the ditch-scenario list restriction on SiteParams (issue #30).
#
# Historically, SiteParams.ditch_depth_west/east, ditch_depth_20y_west/east and
# scenario_name were lists so that a single SUSI run could loop over several
# ditch-depth "scenarios" (see susi_main.py's scenario loop). That's kept only
# for backward compatibility with a legacy workflow; the recommended way to
# compare scenarios is to call SUSI once per scenario. By default SiteParams
# now requires exactly one element in each of those lists, and rejects
# mismatched lengths between them outright; `allow_multiple_ditch_scenarios`
# is the explicit opt-in for the legacy multi-scenario behavior.
import pytest
from pydantic import ValidationError

from susi.io.susi_parameter_model import (
    CanopyLayerName,
    PeatTemperatureParams,
    PeatTypes,
    SiteParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
)


def _make_site_params(**overrides):
    """Build a minimal-but-valid standalone SiteParams, parameterized on
    whichever fields a test cares about."""
    kwargs = dict(
        L=10.0,
        n=5,
        initial_canopylayer_age_years={
            CanopyLayerName.dominant: 70.0,
            CanopyLayerName.subdominant: 70.0,
            CanopyLayerName.under: 70.0,
        },
        sitename="test",
        sfc_specification=1,
        hdom=None,
        vol=None,
        smc="Peatland",
        nLyrs=60,
        dzLyr=0.05,
        ditch_depth_west=[-0.5],
        ditch_depth_east=[-0.5],
        ditch_depth_20y_west=[-0.5],
        ditch_depth_20y_east=[-0.5],
        scenario_name=["test"],
        drain_age=100.0,
        initial_h=-0.2,
        slope=0.0,
        peat_type=[PeatTypes.generic] * 8,
        peat_type_bottom=[PeatTypes.generic],
        anisotropy=10.0,
        vonP=True,
        vonP_top=[2, 5, 5, 5, 6, 6, 7, 7],
        vonP_bottom=8,
        bd_top=None,
        bd_bottom=0.16,
        peatN=None,
        peatP=None,
        peatK=None,
        enable_peattop=True,
        enable_peatmiddle=True,
        enable_peatbottom=True,
        rho_mor=90.0,
        h_mor=h_mor_from_drainage_and_mass_mor_Pitkanen,
        cutting_management=None,
        depoN=4.0,
        depoP=0.1,
        depoK=1.0,
        fertilization=None,
        peat_temperature=PeatTemperatureParams(),
    )
    kwargs.update(overrides)
    return SiteParams(**kwargs)  # ty: ignore[invalid-argument-type]


def test_single_element_lists_are_valid_by_default():
    """The normal, recommended shape (all lists length 1) needs no opt-in."""
    sp = _make_site_params()
    assert len(sp.ditch_depth_west) == 1
    assert sp.allow_multiple_ditch_scenarios is False


def test_mismatched_list_lengths_rejected_even_with_opt_in():
    """Lengths must agree across the five lists regardless of the opt-in flag,
    since susi_main.py's scenario loop zips them together positionally."""
    with pytest.raises(ValidationError, match="same number of elements"):
        _make_site_params(
            ditch_depth_west=[-0.5, -0.6],
            allow_multiple_ditch_scenarios=True,
        )


def test_multiple_scenarios_rejected_without_opt_in():
    """Equal-length lists longer than 1 still fail unless the escape hatch is used."""
    with pytest.raises(ValidationError, match="only accepts one by default"):
        _make_site_params(
            ditch_depth_west=[-0.5, -0.6],
            ditch_depth_east=[-0.5, -0.6],
            ditch_depth_20y_west=[-0.5, -0.6],
            ditch_depth_20y_east=[-0.5, -0.6],
            scenario_name=["a", "b"],
        )


def test_multiple_scenarios_allowed_with_opt_in():
    """The legacy multi-scenario behavior stays available, but only opt-in."""
    sp = _make_site_params(
        ditch_depth_west=[-0.5, -0.6],
        ditch_depth_east=[-0.5, -0.6],
        ditch_depth_20y_west=[-0.5, -0.6],
        ditch_depth_20y_east=[-0.5, -0.6],
        scenario_name=["a", "b"],
        allow_multiple_ditch_scenarios=True,
    )
    assert len(sp.ditch_depth_west) == 2
