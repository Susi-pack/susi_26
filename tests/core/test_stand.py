from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from supersusi.core.canopylayer import Params as CLParams, State as CLState
from supersusi.core.stand import ComputedConstants, Inputs, Outputs, Params, State


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
