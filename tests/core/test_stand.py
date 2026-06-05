from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from supersusi.core.canopylayer import Params as CLParams, State as CLState
from supersusi.core.canopylayer import CuttingOutputs as CLCutting, Outputs as CLOutputs
from supersusi.core.stand import (
    ComputedConstants, Inputs, Outputs, Params, State, _aggregate,
    _compute_lai_above, _merge_cutting_outputs,
)


class TestDataclasses:
    def test_params_is_frozen(self):
        p = Params(
            dominant=CLParams(name="dominant", ncols=5, nlyrs=np.array([0, 1]), sfc=np.ones(5)),
            subdominant=CLParams(name="subdominant", ncols=5, nlyrs=np.zeros(5, dtype=int), sfc=np.ones(5)),
            under=CLParams(name="under", ncols=5, nlyrs=np.zeros(5, dtype=int), sfc=np.ones(5)),
        )
        with pytest.raises(FrozenInstanceError):
            p.dominant = CLParams(name="x", ncols=5, nlyrs=np.ones(5), sfc=np.ones(5))

    def test_params_instantiates(self):
        p = Params(
            dominant=CLParams(name="dominant", ncols=5, nlyrs=np.ones(5, dtype=int), sfc=np.ones(5)),
            subdominant=CLParams(name="subdominant", ncols=5, nlyrs=np.zeros(5, dtype=int), sfc=np.ones(5)),
            under=CLParams(name="under", ncols=5, nlyrs=np.zeros(5, dtype=int), sfc=np.ones(5)),
        )
        assert p.dominant.name == "dominant"

    def test_state_is_frozen(self):
        cl = CLState(agearr=np.ones(5), biomass=np.ones(5), remaining_share=np.ones(5))
        s = State(nut_stat=np.ones(5), dominant=cl, subdominant=cl, under=cl)
        with pytest.raises(FrozenInstanceError):
            s.nut_stat = np.zeros(5)

    def test_state_instantiates(self):
        cl = CLState(agearr=np.ones(5), biomass=np.ones(5), remaining_share=np.ones(5))
        s = State(nut_stat=np.ones(5), dominant=cl, subdominant=cl, under=cl)
        assert s.nut_stat.shape == (5,)

    def test_computed_constants_instantiates(self):
        from supersusi.core.canopylayer import ComputedConstants as CLCC

        empty_cl = CLCC(allodic={}, ixs={}, tree_species=np.ones(5, dtype=np.int32))
        cc = ComputedConstants(dominant=empty_cl, subdominant=empty_cl, under=empty_cl)
        assert cc.dominant is not None

    def test_outputs_instantiates(self):
        z = np.zeros
        n = 3
        o = Outputs(
            basalarea=z(n), biomass=z(n), hdom=z(n), leafarea=z(n), leafmass=z(n),
            stems=z(n), volume=z(n), volumegrowth=z(n), yi=z(n), logvolume=z(n),
            pulpvolume=z(n), mean_diameter=z(n), biomassgrowth=z(n),
            NPP=z(n), NPP_pot=z(n), new_lmass=z(n), leaf_litter=z(n), C_consumption=z(n),
            Nleafdemand=z(n), Nleaf_litter=z(n), N_leaf=z(n),
            Pleafdemand=z(n), Pleaf_litter=z(n), P_leaf=z(n),
            Kleafdemand=z(n), Kleaf_litter=z(n), K_leaf=z(n),
            finerootlitter=z(n), n_finerootlitter=z(n), p_finerootlitter=z(n), k_finerootlitter=z(n),
            nonwoodylitter=z(n), n_nonwoodylitter=z(n), p_nonwoodylitter=z(n), k_nonwoodylitter=z(n),
            woodylitter=z(n), n_woodylitter=z(n), p_woodylitter=z(n), k_woodylitter=z(n),
            woody_litter_mort=z(n), n_woody_litter_mort=z(n), p_woody_litter_mort=z(n), k_woody_litter_mort=z(n),
            non_woody_litter_mort=z(n), n_non_woody_litter_mort=z(n), p_non_woody_litter_mort=z(n), k_non_woody_litter_mort=z(n),
            n_demand=z(n), p_demand=z(n), k_demand=z(n),
            basNdemand=z(n), basPdemand=z(n), basKdemand=z(n),
            harvested_volume=z(n), harvested_log_volume=z(n), harvested_pulp_volume=z(n),
            harvested_biomass=z(n), harvested_stems=z(n),
            nonwoody_lresid=z(n), n_nonwoody_lresid=z(n), p_nonwoody_lresid=z(n), k_nonwoody_lresid=z(n),
            woody_lresid=z(n), n_woody_lresid=z(n), p_woody_lresid=z(n), k_woody_lresid=z(n),
        )
        assert o.basalarea.shape == (n,)

    def test_inputs_instantiates(self):
        import types

        n = 5
        inp = Inputs(
            photopara=types.SimpleNamespace(),
            forc=type("", (), {})(),
            wt=type("", (), {})(),
            afp=type("", (), {})(),
            n_supply=np.ones(n),
            p_supply=np.ones(n),
            k_supply=np.ones(n),
            groundvegetation_outputs=None,
            previous_nut_stat=np.ones(n),
            calendar_year=2025,
            cutting_to_ba=None,
        )
        assert inp.calendar_year == 2025


def _make_cl_output(
    stems: np.ndarray,
    hdom: np.ndarray,
    dg: np.ndarray,
    multiplier: float,
    ncols: int,
) -> CLOutputs:
    kw: dict[str, np.ndarray] = {
        "stems": stems,
        "hdom": hdom,
        "Dg": dg,
        "basalarea": np.full(ncols, multiplier),
        "volume": np.full(ncols, multiplier),
        "leafarea": np.full(ncols, multiplier),
        "leafmass": np.full(ncols, multiplier),
        "volumegrowth": np.full(ncols, multiplier),
        "logvolume": np.full(ncols, multiplier),
        "pulpvolume": np.full(ncols, multiplier),
        "yi": np.full(ncols, multiplier),
        "NPP": np.full(ncols, multiplier),
        "NPP_pot": np.full(ncols, multiplier),
        "new_lmass": np.full(ncols, multiplier),
        "leaf_litter": np.full(ncols, multiplier),
        "C_consumption": np.full(ncols, multiplier),
        "Nleafdemand": np.full(ncols, multiplier),
        "Nleaf_litter": np.full(ncols, multiplier),
        "N_leaf": np.full(ncols, multiplier),
        "Pleafdemand": np.full(ncols, multiplier),
        "Pleaf_litter": np.full(ncols, multiplier),
        "P_leaf": np.full(ncols, multiplier),
        "Kleafdemand": np.full(ncols, multiplier),
        "Kleaf_litter": np.full(ncols, multiplier),
        "K_leaf": np.full(ncols, multiplier),
        "finerootlitter": np.full(ncols, multiplier),
        "n_finerootlitter": np.full(ncols, multiplier),
        "p_finerootlitter": np.full(ncols, multiplier),
        "k_finerootlitter": np.full(ncols, multiplier),
        "nonwoodylitter": np.full(ncols, multiplier),
        "n_nonwoodylitter": np.full(ncols, multiplier),
        "p_nonwoodylitter": np.full(ncols, multiplier),
        "k_nonwoodylitter": np.full(ncols, multiplier),
        "woodylitter": np.full(ncols, multiplier),
        "n_woodylitter": np.full(ncols, multiplier),
        "p_woodylitter": np.full(ncols, multiplier),
        "k_woodylitter": np.full(ncols, multiplier),
        "woody_litter_mort": np.full(ncols, multiplier),
        "n_woody_litter_mort": np.full(ncols, multiplier),
        "p_woody_litter_mort": np.full(ncols, multiplier),
        "k_woody_litter_mort": np.full(ncols, multiplier),
        "non_woody_litter_mort": np.full(ncols, multiplier),
        "n_non_woody_litter_mort": np.full(ncols, multiplier),
        "p_non_woody_litter_mort": np.full(ncols, multiplier),
        "k_non_woody_litter_mort": np.full(ncols, multiplier),
        "n_demand": np.full(ncols, multiplier),
        "p_demand": np.full(ncols, multiplier),
        "k_demand": np.full(ncols, multiplier),
        "basNdemand": np.full(ncols, multiplier),
        "basPdemand": np.full(ncols, multiplier),
        "basKdemand": np.full(ncols, multiplier),
    }
    return CLOutputs(**kw)


class TestAggregate:
    def test_aggregate_sums_per_tree_fields_and_converts_to_ha(self):
        n = 3
        dom_stems = np.array([800.0, 900.0, 1000.0])
        sub_stems = np.array([400.0, 500.0, 600.0])
        under_stems = np.array([200.0, 300.0, 400.0])
        dom_hdom = np.array([18.0, 20.0, 22.0])
        sub_hdom = np.array([12.0, 14.0, 16.0])
        under_hdom = np.array([6.0, 8.0, 10.0])
        dom_dg = np.array([20.0, 22.0, 24.0])
        sub_dg = np.array([14.0, 16.0, 18.0])
        under_dg = np.array([8.0, 10.0, 12.0])

        dom_out = _make_cl_output(dom_stems, dom_hdom, dom_dg, 1.0, n)
        sub_out = _make_cl_output(sub_stems, sub_hdom, sub_dg, 0.5, n)
        under_out = _make_cl_output(under_stems, under_hdom, under_dg, 0.25, n)

        dom_bm = np.array([50.0, 60.0, 70.0])
        sub_bm = np.array([20.0, 25.0, 30.0])
        under_bm = np.array([5.0, 8.0, 12.0])
        prev_bm = np.array([0.0, 0.0, 0.0])

        result = _aggregate(dom_out, sub_out, under_out, dom_bm, sub_bm, under_bm, prev_bm)

        total_stems = dom_stems + sub_stems + under_stems

        # stems are summed directly
        np.testing.assert_array_equal(result.stems, total_stems)

        # hdom is maximum
        expected_hdom = np.maximum(dom_hdom, np.maximum(sub_hdom, under_hdom))
        np.testing.assert_array_equal(result.hdom, expected_hdom)

        # mean_diameter is Dg weighted by stems
        expected_md = (dom_dg * dom_stems + sub_dg * sub_stems + under_dg * under_stems) / total_stems
        np.testing.assert_array_almost_equal(result.mean_diameter, expected_md)

        # biomass = state_biomass × stems summed
        expected_biomass = dom_bm * dom_stems + sub_bm * sub_stems + under_bm * under_stems
        np.testing.assert_array_equal(result.biomass, expected_biomass)

        # biomassgrowth = biomass - prev
        np.testing.assert_array_equal(result.biomassgrowth, expected_biomass)

        # per-tree fields: field × stems summed
        expected_per_tree = 1.0 * dom_stems + 0.5 * sub_stems + 0.25 * under_stems
        np.testing.assert_array_equal(result.basalarea, expected_per_tree)
        np.testing.assert_array_equal(result.NPP, expected_per_tree)
        np.testing.assert_array_equal(result.n_demand, expected_per_tree)
        np.testing.assert_array_equal(result.finerootlitter, expected_per_tree)

        # cutting fields are zero
        assert np.all(result.harvested_volume == 0)
        assert np.all(result.nonwoody_lresid == 0)
        assert np.all(result.woody_lresid == 0)

    def test_aggregate_without_previous_biomass_zeros_biomassgrowth(self):
        n = 2
        stems = np.array([500.0, 600.0])
        hdom = np.array([15.0, 18.0])
        dg = np.array([12.0, 15.0])
        out = _make_cl_output(stems, hdom, dg, 1.0, n)
        bm = np.array([30.0, 40.0])

        result = _aggregate(out, out, out, bm, bm, bm, previous_stand_biomass=None)

        expected_biomass = bm * stems * 3
        np.testing.assert_array_equal(result.biomass, expected_biomass)
        np.testing.assert_array_equal(result.biomassgrowth, np.zeros(n))

    def test_aggregate_zero_stems_division_safe(self):
        n = 3
        out = _make_cl_output(
            np.zeros(n), np.zeros(n), np.zeros(n), 0.0, n,
        )
        bm = np.zeros(n)
        result = _aggregate(out, out, out, bm, bm, bm)
        np.testing.assert_array_equal(result.stems, np.zeros(n))
        np.testing.assert_array_equal(result.mean_diameter, np.zeros(n))


def _make_cutting_output(value: float, ncols: int) -> CLCutting:
    return CLCutting(
        harvested_volume=np.full(ncols, value),
        harvested_log_volume=np.full(ncols, value),
        harvested_pulp_volume=np.full(ncols, value),
        harvested_biomass=np.full(ncols, value),
        harvested_stems=np.full(ncols, value),
        nonwoody_lresid=np.full(ncols, value),
        n_nonwoody_lresid=np.full(ncols, value),
        p_nonwoody_lresid=np.full(ncols, value),
        k_nonwoody_lresid=np.full(ncols, value),
        woody_lresid=np.full(ncols, value),
        n_woody_lresid=np.full(ncols, value),
        p_woody_lresid=np.full(ncols, value),
        k_woody_lresid=np.full(ncols, value),
    )


class TestMergeCuttingOutputs:
    def test_merges_single_layer(self):
        n = 3
        base = _make_cl_output(np.ones(n), np.ones(n), np.ones(n), 1.0, n)
        stand_out = _aggregate(base, base, base, np.ones(n), np.ones(n), np.ones(n))
        dom_cut = _make_cutting_output(10.0, n)

        result = _merge_cutting_outputs(stand_out, dom_cut=dom_cut)

        np.testing.assert_array_equal(result.harvested_volume, np.full(n, 10.0))
        np.testing.assert_array_equal(result.nonwoody_lresid, np.full(n, 10.0))
        np.testing.assert_array_equal(result.woody_lresid, np.full(n, 10.0))
        np.testing.assert_array_equal(result.stems, stand_out.stems)

    def test_merges_three_layers(self):
        n = 3
        base = _make_cl_output(np.ones(n), np.ones(n), np.ones(n), 1.0, n)
        stand_out = _aggregate(base, base, base, np.ones(n), np.ones(n), np.ones(n))
        dom_cut = _make_cutting_output(10.0, n)
        sub_cut = _make_cutting_output(5.0, n)
        under_cut = _make_cutting_output(2.0, n)

        result = _merge_cutting_outputs(stand_out, dom_cut, sub_cut, under_cut)

        np.testing.assert_array_equal(result.harvested_volume, np.full(n, 17.0))
        np.testing.assert_array_equal(result.harvested_stems, np.full(n, 17.0))
        np.testing.assert_array_equal(result.nonwoody_lresid, np.full(n, 17.0))

    def test_none_layers_contribute_zero(self):
        n = 2
        base = _make_cl_output(np.ones(n), np.ones(n), np.ones(n), 1.0, n)
        stand_out = _aggregate(base, base, base, np.ones(n), np.ones(n), np.ones(n))
        dom_cut = _make_cutting_output(7.0, n)

        result = _merge_cutting_outputs(stand_out, dom_cut=dom_cut)

        np.testing.assert_array_equal(result.harvested_volume, np.full(n, 7.0))
        assert np.all(result.nonwoody_lresid == 7.0)

    def test_no_cutting_returns_stand_out(self):
        n = 2
        base = _make_cl_output(np.ones(n), np.ones(n), np.ones(n), 1.0, n)
        stand_out = _aggregate(base, base, base, np.ones(n), np.ones(n), np.ones(n))

        result = _merge_cutting_outputs(stand_out)

        assert np.all(result.harvested_volume == 0)
        assert np.all(result.nonwoody_lresid == 0)
        np.testing.assert_array_equal(result.stems, stand_out.stems)
        np.testing.assert_array_equal(result.biomass, stand_out.biomass)


class TestComputeLaiAbove:
    def test_returns_tuples_of_arrays(self):
        n = 3
        allom = _make_cl_output(np.ones(n), np.ones(n), np.ones(n), 1.0, n)
        dom, sub, under = _compute_lai_above(allom, allom, allom)
        assert dom.shape == (n,)
        assert sub.shape == (n,)
        assert under.shape == (n,)

    def test_tallest_layer_gets_zero_lai_above(self):
        n = 3
        stems_dom = np.full(n, 1000.0)
        stems_sub = np.full(n, 500.0)
        stems_under = np.full(n, 100.0)
        dom = _make_cl_output(stems_dom, np.full(n, 20.0), np.ones(n), 1.0, n)
        sub = _make_cl_output(stems_sub, np.full(n, 15.0), np.ones(n), 1.0, n)
        under = _make_cl_output(stems_under, np.full(n, 8.0), np.ones(n), 1.0, n)
        # Override leafarea (default is multiplier=1.0)
        dom = replace(dom, leafarea=np.full(n, 5.0))
        sub = replace(sub, leafarea=np.full(n, 3.0))
        under = replace(under, leafarea=np.full(n, 1.0))

        lai_dom, lai_sub, lai_under = _compute_lai_above(dom, sub, under)

        np.testing.assert_array_equal(lai_dom, np.zeros(n))
        np.testing.assert_array_equal(lai_sub, np.full(n, 5000.0))  # 5 * 1000
        np.testing.assert_array_equal(lai_under, np.full(n, 6500.0))  # 5000 + 1500
