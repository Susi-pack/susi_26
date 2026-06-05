from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from supersusi.core.canopylayer import (
    Params,
    ComputedConstants,
    State,
    Outputs,
    CuttingOutputs,
    LeafDynamicsOutputs,
    Inputs,
)


class TestDataclasses:
    def test_params_is_frozen(self):
        p = Params(name="dominant", ncols=5, nlyrs=np.array([0, 1]), sfc=np.ones(5))
        with pytest.raises(FrozenInstanceError):
            p.name = "subdominant"

    def test_params_instantiates(self):
        p = Params(name="dominant", ncols=5, nlyrs=np.array([0, 1]), sfc=np.ones(5))
        assert p.name == "dominant"
        assert p.ncols == 5

    def test_state_is_frozen(self):
        s = State(
            agearr=np.ones(5), biomass=np.ones(5), remaining_share=np.ones(5)
        )
        with pytest.raises(FrozenInstanceError):
            s.agearr = np.zeros(5)

    def test_state_instantiates(self):
        s = State(
            agearr=np.ones(5), biomass=np.ones(5), remaining_share=np.ones(5)
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
            co.harvested_volume = np.zeros(5)

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
        )
        assert ldo.new_lmass.shape == (5,)

    def test_inputs_instantiates(self):
        inp = Inputs(
            photopara=None,
            forc="dummy",
            wt="dummy",
            afp="dummy",
            previous_nut_stat=np.ones(5),
            nut_stat=np.ones(5),
            lai_above=np.ones(5),
        )
        assert inp.lai_above.shape == (5,)

    def test_computed_constants_is_frozen(self):
        cc = ComputedConstants(
            allodic={},
            ixs={},
            tree_species=np.ones(5, dtype=np.int32),
        )
        with pytest.raises(FrozenInstanceError):
            cc.tree_species = np.zeros(5, dtype=np.int32)
