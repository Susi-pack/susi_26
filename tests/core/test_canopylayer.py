from dataclasses import FrozenInstanceError

import types

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import interp1d

from supersusi.core.allometry import (
    AgeBased,
    AllometryFunctions,
    BiomassToStand,
    FineRoots,
    LitterMass,
    LoggingResidues,
    MortalityMass,
    NutrientDemand,
    NutrientLitter,
    NutrientMortality,
    YieldVolume,
)
from supersusi.core.canopylayer import (
    ComputedConstants,
    CuttingOutputs,
    Inputs,
    LeafDynamicsOutputs,
    Outputs,
    Params,
    State,
    apply_allometry,
    compute_constants,
    compute_initial_state,
    cut_stand,
    grow_stand,
)


def _identity_interp():
    """Simple identity interpolation function for testing."""
    return interp1d([0, 100], [0, 100], bounds_error=False, fill_value="extrapolate")


def _make_mock_allometry():
    """Create an AllometryFunctions with minimal working interp1d for testing."""
    identity = _identity_interp()
    return AllometryFunctions(
        age_based=AgeBased(
            hdom=identity,
            ba=identity,
            vol=identity,
            yield_=identity,
            bm=identity,
            bm_no_leaves=identity,
            leaves=identity,
        ),
        biomass_to_stand=BiomassToStand(
            leaf_mass=identity,
            with_leaves_to_leaf_mass=identity,
            lai=identity,
            hdom=identity,
            dg=identity,
            yi=identity,
            vol=identity,
            log_vol=identity,
            pulp_vol=identity,
            ba=identity,
            dbm=identity,
            stems=identity,
        ),
        yield_volume=YieldVolume(
            yi_to_vol=identity,
            yi_to_bm=identity,
            vol_to_logs=identity,
            vol_to_pulp=identity,
        ),
        fine_roots=FineRoots(
            fine_roots=identity,
            n_fine_roots=identity,
            p_fine_roots=identity,
            k_fine_roots=identity,
        ),
        litter_mass=LitterMass(
            fine_root_litter=identity,
            woody_litter=identity,
            with_leaves_to_fine_root_litter=identity,
            with_leaves_to_woody_litter=identity,
        ),
        mortality_mass=MortalityMass(
            fine_root=identity,
            woody=identity,
            leaves=identity,
        ),
        nutrient_demand=NutrientDemand(
            n_demand=identity,
            p_demand=identity,
            k_demand=identity,
            n_leaf_demand=identity,
            p_leaf_demand=identity,
            k_leaf_demand=identity,
        ),
        nutrient_litter=NutrientLitter(
            n_fine_root_litter=identity,
            p_fine_root_litter=identity,
            k_fine_root_litter=identity,
            n_woody_litter=identity,
            p_woody_litter=identity,
            k_woody_litter=identity,
        ),
        nutrient_mortality=NutrientMortality(
            n_mortality_leaves=identity,
            p_mortality_leaves=identity,
            k_mortality_leaves=identity,
            n_mortality_fine_root=identity,
            p_mortality_fine_root=identity,
            k_mortality_fine_root=identity,
            n_mortality_woody=identity,
            p_mortality_woody=identity,
            k_mortality_woody=identity,
        ),
        logging_residues=LoggingResidues(
            woody=identity,
            n_woody=identity,
            p_woody=identity,
            k_woody=identity,
        ),
    )


class TestDataclasses:
    def test_params_is_frozen(self):
        p = Params(name="dominant", ncols=5, nlyrs=np.array([0, 1]), sfc=np.ones(5))
        with pytest.raises(FrozenInstanceError):
            p.name = "subdominant"  # ty: ignore[invalid-assignment]

    def test_params_instantiates(self):
        p = Params(name="dominant", ncols=5, nlyrs=np.array([0, 1]), sfc=np.ones(5))
        assert p.name == "dominant"
        assert p.ncols == 5

    def test_state_is_frozen(self):
        s = State(
            agearr=np.ones(5),
            biomass=np.ones(5),
            remaining_share=np.ones(5),
            leafmass=np.ones(5),
        )
        with pytest.raises(FrozenInstanceError):
            s.agearr = np.zeros(5)  # ty: ignore[invalid-assignment]

    def test_state_instantiates(self):
        s = State(
            agearr=np.ones(5),
            biomass=np.ones(5),
            remaining_share=np.ones(5),
            leafmass=np.ones(5),
        )
        assert s.agearr.shape == (5,)

    def test_outputs_instantiates(self):
        o = Outputs(
            stems=np.ones(5),
            basalarea=np.ones(5),
            hdom=np.ones(5),
            Dg=np.ones(5),
            volume=np.ones(5),
            leafarea=np.ones(5),
            leafmass=np.ones(5),
            volumegrowth=np.ones(5),
            logvolume=np.ones(5),
            pulpvolume=np.ones(5),
            yi=np.ones(5),
            NPP=np.ones(5),
            NPP_pot=np.ones(5),
            new_lmass=np.ones(5),
            leaf_litter=np.ones(5),
            C_consumption=np.ones(5),
            Nleafdemand=np.ones(5),
            Nleaf_litter=np.ones(5),
            N_leaf=np.ones(5),
            Pleafdemand=np.ones(5),
            Pleaf_litter=np.ones(5),
            P_leaf=np.ones(5),
            Kleafdemand=np.ones(5),
            Kleaf_litter=np.ones(5),
            K_leaf=np.ones(5),
            finerootlitter=np.ones(5),
            n_finerootlitter=np.ones(5),
            p_finerootlitter=np.ones(5),
            k_finerootlitter=np.ones(5),
            nonwoodylitter=np.ones(5),
            n_nonwoodylitter=np.ones(5),
            p_nonwoodylitter=np.ones(5),
            k_nonwoodylitter=np.ones(5),
            woodylitter=np.ones(5),
            n_woodylitter=np.ones(5),
            p_woodylitter=np.ones(5),
            k_woodylitter=np.ones(5),
            woody_litter_mort=np.ones(5),
            n_woody_litter_mort=np.ones(5),
            p_woody_litter_mort=np.ones(5),
            k_woody_litter_mort=np.ones(5),
            non_woody_litter_mort=np.ones(5),
            n_non_woody_litter_mort=np.ones(5),
            p_non_woody_litter_mort=np.ones(5),
            k_non_woody_litter_mort=np.ones(5),
            n_demand=np.ones(5),
            p_demand=np.ones(5),
            k_demand=np.ones(5),
            basNdemand=np.ones(5),
            basPdemand=np.ones(5),
            basKdemand=np.ones(5),
            leafmax=np.ones(5),
            leafmin=np.ones(5),
        )
        assert o.stems.shape == (5,)

    def test_cutting_outputs_is_frozen(self):
        co = CuttingOutputs(
            harvested_volume=np.ones(5),
            harvested_log_volume=np.ones(5),
            harvested_pulp_volume=np.ones(5),
            harvested_biomass=np.ones(5),
            harvested_stems=np.ones(5),
            nonwoody_lresid=np.ones(5),
            n_nonwoody_lresid=np.ones(5),
            p_nonwoody_lresid=np.ones(5),
            k_nonwoody_lresid=np.ones(5),
            woody_lresid=np.ones(5),
            n_woody_lresid=np.ones(5),
            p_woody_lresid=np.ones(5),
            k_woody_lresid=np.ones(5),
        )
        with pytest.raises(FrozenInstanceError):
            co.harvested_volume = np.zeros(5)  # ty: ignore[invalid-assignment]

    def test_leaf_dynamics_outputs_instantiates(self):
        ldo = LeafDynamicsOutputs(
            new_lmass=np.ones(5),
            leaf_litter=np.ones(5),
            C_consumption=np.ones(5),
            Nleafdemand=np.ones(5),
            Nleaf_litter=np.ones(5),
            N_leaf=np.ones(5),
            Pleafdemand=np.ones(5),
            Pleaf_litter=np.ones(5),
            P_leaf=np.ones(5),
            Kleafdemand=np.ones(5),
            Kleaf_litter=np.ones(5),
            K_leaf=np.ones(5),
            leafmax=np.ones(5),
            leafmin=np.ones(5),
        )
        assert ldo.new_lmass.shape == (5,)

    def test_compute_constants_returns_correct_type(self):
        ncols = 5
        params = Params(
            name="dominant",
            ncols=ncols,
            nlyrs=np.ones(ncols, dtype=int),
            sfc=np.ones(ncols, dtype=int) * 3,
        )
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
        data = np.zeros((3, len(cnames)))
        data[:, 0] = [1, 2, 3]  # yr
        data[:, 1] = [1, 2, 3]  # age
        data[:, 6] = [1.0, 2.0, 3.0]  # hdom
        df = pd.DataFrame(data, columns=pd.Index(cnames))
        cc = compute_constants(params, {1: df}, {1: 1})
        assert isinstance(cc, ComputedConstants)
        assert 1 in cc.allodic
        assert isinstance(cc.allodic[1], AllometryFunctions)

    def test_compute_initial_state_returns_state_and_outputs(self):
        ncols = 5
        params = Params(
            name="dominant",
            ncols=ncols,
            nlyrs=np.ones(ncols, dtype=int),
            sfc=np.ones(ncols, dtype=int) * 3,
        )
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
        data = np.zeros((3, len(cnames)))
        data[:, 0] = [1, 2, 3]
        data[:, 1] = [1, 2, 3]
        data[:, 6] = [1.0, 2.0, 3.0]
        df = pd.DataFrame(data, columns=pd.Index(cnames))
        cc = compute_constants(params, {1: df}, {1: 1})
        agearr = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        nut_stat = np.ones(ncols)
        state, outputs = compute_initial_state(params, cc, agearr, nut_stat)
        assert isinstance(state, State)
        assert isinstance(outputs, Outputs)
        assert state.agearr.shape == (ncols,)
        assert state.biomass.shape == (ncols,)
        assert state.remaining_share.shape == (ncols,)
        np.testing.assert_array_equal(state.remaining_share, np.ones(ncols))
        np.testing.assert_array_equal(state.agearr, agearr)
        assert np.all(state.biomass >= 0)
        assert outputs.stems.shape == (ncols,)

    def test_grow_stand_returns_state_and_outputs(self):
        ncols = 10
        mock_af = _make_mock_allometry()
        cc = ComputedConstants(
            allodic={1: mock_af},
            ixs={1: np.arange(ncols)},
            tree_species=np.ones(ncols, dtype=np.int32),
        )
        state = State(
            agearr=np.linspace(1.0, 3.0, ncols),
            biomass=np.full(ncols, 20.0),
            remaining_share=np.ones(ncols),
            leafmass=np.full(ncols, 1.0),
        )

        days = 3
        photopara = types.SimpleNamespace(
            beta=1.0,
            gamma=0.5,
            kappa=-0.5,
            tau=10.0,
            X0=5.0,
            Smax=20.0,
            alfa=0.5,
            nu=2.0,
        )
        forc = pd.DataFrame(
            {
                "Rg": np.full(days, 100.0),
                "vpd": np.full(days, 0.5),
                "T": np.full(days, 15.0),
            }
        )
        wt = pd.DataFrame(np.full((days, ncols), -0.3))
        afp = pd.DataFrame(np.ones((days, ncols)))

        inputs = Inputs(
            photopara=photopara,
            forc=forc,
            wt=wt,
            afp=afp,
            previous_nut_stat=np.ones(ncols),
            nut_stat=np.ones(ncols),
            lai_above=np.zeros(ncols),
        )

        new_state, out = grow_stand(state, cc, inputs)
        assert isinstance(new_state, State)
        assert isinstance(out, Outputs)
        assert new_state.agearr.shape == (ncols,)
        assert new_state.biomass.shape == (ncols,)
        assert out.stems.shape == (ncols,)
        np.testing.assert_array_equal(new_state.agearr, state.agearr + 1)
        assert np.all(out.NPP >= 0)
        assert np.all(out.leaf_litter >= 0)
        assert np.all(out.finerootlitter >= 0)
        assert np.all(out.nonwoodylitter >= 0)
        assert np.all(out.volumegrowth >= 0)

    def test_cut_stand_thinning_returns_cutting_outputs(self):
        ncols = 10
        mock_af = _make_mock_allometry()
        cc = ComputedConstants(
            allodic={1: mock_af},
            ixs={1: np.arange(ncols)},
            tree_species=np.ones(ncols, dtype=np.int32),
        )
        state = State(
            agearr=np.full(ncols, 20.0),
            biomass=np.full(ncols, 20.0),
            remaining_share=np.ones(ncols),
            leafmass=np.full(ncols, 1.0),
        )
        out = apply_allometry(state.biomass, state.agearr, state.remaining_share, cc)
        nuts = np.ones(ncols)

        new_state, cut = cut_stand(state, cc, out, nuts, to_ba=12.0)
        assert isinstance(new_state, State)
        assert isinstance(cut, CuttingOutputs)
        assert new_state.biomass.shape == (ncols,)
        assert cut.harvested_volume.shape == (ncols,)
        assert np.all(cut.harvested_volume >= 0)
        assert np.all(cut.harvested_stems >= 0)
        assert np.all(cut.harvested_stems <= out.stems)
        assert np.all(cut.nonwoody_lresid >= 0)
        assert np.all(cut.woody_lresid >= 0)


class TestApplyAllometry:
    def test_returns_outputs_with_correct_shape(self):
        ncols = 5
        bm = np.arange(10.0, 10.0 + ncols)
        age = np.full(ncols, 10.0)
        remaining = np.ones(ncols)
        mock_af = _make_mock_allometry()
        cc = ComputedConstants(
            allodic={1: mock_af},
            ixs={1: np.arange(ncols)},
            tree_species=np.ones(ncols, dtype=np.int32),
        )
        out = apply_allometry(bm, age, remaining, cc)
        assert isinstance(out, Outputs)
        for field_name in Outputs.__dataclass_fields__:
            val = getattr(out, field_name)
            assert isinstance(val, np.ndarray), f"{field_name} is not an ndarray"
            assert val.shape == (ncols,), f"{field_name} shape mismatch"

    def test_growth_fields_are_zero(self):
        ncols = 5
        bm = np.ones(ncols) * 20.0
        age = np.full(ncols, 10.0)
        remaining = np.ones(ncols)
        mock_af = _make_mock_allometry()
        cc = ComputedConstants(
            allodic={1: mock_af},
            ixs={1: np.arange(ncols)},
            tree_species=np.ones(ncols, dtype=np.int32),
        )
        out = apply_allometry(bm, age, remaining, cc)
        assert np.all(out.NPP == 0.0)
        assert np.all(out.NPP_pot == 0.0)
        assert np.all(out.leaf_litter == 0.0)
        assert np.all(out.C_consumption == 0.0)
        assert np.all(out.new_lmass == 0.0)
        assert np.all(out.Nleafdemand == 0.0)
        assert np.all(out.Pleafdemand == 0.0)
        assert np.all(out.Kleafdemand == 0.0)
        assert np.all(out.Nleaf_litter == 0.0)
        assert np.all(out.Pleaf_litter == 0.0)
        assert np.all(out.Kleaf_litter == 0.0)
        assert np.all(out.N_leaf == 0.0)
        assert np.all(out.P_leaf == 0.0)
        assert np.all(out.K_leaf == 0.0)

    def test_multiple_zones(self):
        ncols = 10
        bm = np.arange(10.0, 10.0 + ncols)
        age = np.full(ncols, 10.0)
        remaining = np.ones(ncols)
        mock_1 = _make_mock_allometry()
        mock_2 = _make_mock_allometry()
        cc = ComputedConstants(
            allodic={1: mock_1, 2: mock_2},
            ixs={1: np.arange(0, 5), 2: np.arange(5, 10)},
            tree_species=np.concatenate([np.ones(5), np.full(5, 2)]).astype(np.int32),
        )
        out = apply_allometry(bm, age, remaining, cc)
        assert out.stems.shape == (ncols,)

    def test_nonwoodylitter_equals_finerootlitter(self):
        ncols = 5
        bm = np.ones(ncols) * 20.0
        age = np.full(ncols, 10.0)
        remaining = np.ones(ncols)
        mock_af = _make_mock_allometry()
        cc = ComputedConstants(
            allodic={1: mock_af},
            ixs={1: np.arange(ncols)},
            tree_species=np.ones(ncols, dtype=np.int32),
        )
        out = apply_allometry(bm, age, remaining, cc)
        np.testing.assert_array_equal(out.nonwoodylitter, out.finerootlitter)
        np.testing.assert_array_equal(out.n_nonwoodylitter, out.n_finerootlitter)
        np.testing.assert_array_equal(out.p_nonwoodylitter, out.p_finerootlitter)
        np.testing.assert_array_equal(out.k_nonwoodylitter, out.k_finerootlitter)
