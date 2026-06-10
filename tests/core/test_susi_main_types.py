import numpy as np
from dataclasses import replace

from supersusi.core.susi_main import (
    DailyState, AnnualState, AllState,
    DailyForcing, DailyOutputs,
    AnnualForcing, AnnualOutputs,
    SimulationOutput,
    ModuleComputedConstants,
)


def test_dataclasses_are_frozen():
    for cls in [DailyState, AnnualState, AllState, DailyForcing,
                DailyOutputs, AnnualForcing, AnnualOutputs, SimulationOutput,
                ModuleComputedConstants]:
        assert cls.__dataclass_params__.frozen


def test_daily_state_construction():
    state = DailyState(
        canopy=_mock_canopy_state(),
        moss=_mock_moss_state(),
        strip=_mock_strip_state(),
        peat_T=_mock_peat_T_state(),
    )
    assert state is not None


def test_annual_state_construction():
    state = AnnualState(
        stand=_mock_stand_state(),
        stand_outputs=_mock_stand_outputs(),
        gv=_mock_gv_state(),
        esom_mass=_mock_esom_state(),
        esom_N=_mock_esom_state(),
        esom_P=_mock_esom_state(),
        esom_K=_mock_esom_state(),
    )
    assert state is not None


def test_all_state_construction():
    ds = DailyState(
        canopy=_mock_canopy_state(), moss=_mock_moss_state(),
        strip=_mock_strip_state(), peat_T=_mock_peat_T_state(),
    )
    ann = AnnualState(
        stand=_mock_stand_state(), stand_outputs=_mock_stand_outputs(),
        gv=_mock_gv_state(), esom_mass=_mock_esom_state(),
        esom_N=_mock_esom_state(), esom_P=_mock_esom_state(),
        esom_K=_mock_esom_state(),
    )
    state = AllState(daily=ds, annual=ann)
    assert state is not None


def test_daily_forcing_construction():
    f = DailyForcing(
        T=5.0, Prec=0.001, Rg=100.0, Par=50.0, VPD=0.5,
        h0ts_west=-0.3, h0ts_east=-0.3,
    )
    assert abs(f.T - 5.0) < 1e-12


def test_daily_outputs_construction():
    n = 5
    out = DailyOutputs(
        wtd=np.zeros(n), afp=np.zeros(n),
        T_soil_hydro=np.zeros(3), delta=np.zeros(n),
        total_runoff=np.asarray(0.0), surface_runoff=np.zeros(n),
        interc=np.zeros(n), evap=np.zeros(n), et=np.zeros(n),
        transpi=np.zeros(n), efloor=np.zeros(n), swe=np.zeros(n),
        H=np.zeros(n), runoffwest=0.0, roffeast=0.0,
    )
    assert out.wtd.shape == (n,)


def test_annual_forcing_construction():
    ndays = 365
    n = 5
    forc = AnnualForcing(
        daily_T=np.zeros(ndays), daily_Rg=np.zeros(ndays),
        daily_VPD=np.zeros(ndays), daily_Prec=np.zeros(ndays),
        daily_Par=np.zeros(ndays),
        h0ts_west=np.zeros(ndays), h0ts_east=np.zeros(ndays),
        valid_days=ndays, calendar_year=2004,
        temp_sum=np.zeros(n), do_cutting=False, cutting_to_ba=0.0,
    )
    assert forc.valid_days == 365


def test_simulation_output_construction():
    n = 5
    ds = DailyState(canopy=_mock_canopy_state(), moss=_mock_moss_state(),
                    strip=_mock_strip_state(), peat_T=_mock_peat_T_state())
    ann = AnnualState(stand=_mock_stand_state(),
                      stand_outputs=_mock_stand_outputs(),
                      gv=_mock_gv_state(), esom_mass=_mock_esom_state(),
                      esom_N=_mock_esom_state(), esom_P=_mock_esom_state(),
                      esom_K=_mock_esom_state())
    final = AllState(daily=ds, annual=ann)

    do = DailyOutputs(wtd=np.zeros(n), afp=np.zeros(n),
                      T_soil_hydro=np.zeros(3), delta=np.zeros(n),
                      total_runoff=np.zeros(1), surface_runoff=np.zeros(n),
                      interc=np.zeros(n), evap=np.zeros(n), et=np.zeros(n),
                      transpi=np.zeros(n), efloor=np.zeros(n), swe=np.zeros(n),
                      H=np.zeros(n), runoffwest=0.0, roffeast=0.0)
    ao = AnnualOutputs(
        daily=do, stand=_mock_stand_outputs(),
        stand_dom=_mock_canopylayer_outputs(),
        stand_sub=_mock_canopylayer_outputs(),
        stand_under=_mock_canopylayer_outputs(),
        gv=_mock_gv_outputs(),
        esom_mass=_mock_esom_yr_outputs(),
        esom_N=_mock_esom_yr_outputs(),
        esom_P=_mock_esom_yr_outputs(),
        esom_K=_mock_esom_yr_outputs(),
        methane=_mock_methane_outputs(),
        fertilization=_mock_fert_outputs(),
        Rhet=np.zeros(n), soil_co2_balance=np.zeros(n),
        doc_export=_mock_doc_export(),
        strip_diag=_mock_strip_diag(),
    )
    sim = SimulationOutput(annual=[ao], final_state=final)
    assert len(sim.annual) == 1


def test_replace_on_state():
    ds = DailyState(canopy=_mock_canopy_state(), moss=_mock_moss_state(),
                    strip=_mock_strip_state(), peat_T=_mock_peat_T_state())
    ds2 = replace(ds, strip=_mock_strip_state())
    assert ds2.strip is not ds.strip
    assert ds2.canopy is ds.canopy


# ── Helpers ──────────────────────────────────────────────────────────

def _mock_canopy_state():
    from supersusi.core.canopygrid import State
    return State(W=np.zeros(5), SWE=np.zeros(5), SWEi=np.zeros(5),
                 SWEl=np.zeros(5), X=np.zeros(5), amax=np.ones(5))


def _mock_moss_state():
    from supersusi.core.mosslayer import State
    return State(Wsto_top=np.zeros(5), h_pond=np.zeros(5), Ree=np.ones(5))


def _mock_strip_state():
    from supersusi.core.strip import State
    return State(H=np.zeros(5))


def _mock_peat_T_state():
    from supersusi.core.temperature import State
    return State(T_soil=np.zeros(31))


def _mock_stand_state():
    from supersusi.core.stand import State
    from supersusi.core.canopylayer import State as CLState
    return State(nut_stat=np.ones(5), previous_nut_stat=np.ones(5),
                 dominant=CLState(agearr=np.zeros(5), biomass=np.zeros(5),
                                  remaining_share=np.ones(5), leafmass=np.zeros(5)),
                 subdominant=CLState(agearr=np.zeros(5), biomass=np.zeros(5),
                                     remaining_share=np.ones(5), leafmass=np.zeros(5)),
                 under=CLState(agearr=np.zeros(5), biomass=np.zeros(5),
                               remaining_share=np.ones(5), leafmass=np.zeros(5)))


def _mock_stand_outputs():
    from supersusi.core.stand import Outputs
    z = np.zeros(5)
    return Outputs(basalarea=z.copy(), biomass=z.copy(), hdom=z.copy(),
                   leafarea=z.copy(), leafmass=z.copy(), stems=z.copy(),
                   volume=z.copy(), volumegrowth=z.copy(), yi=z.copy(),
                   logvolume=z.copy(), pulpvolume=z.copy(),
                   mean_diameter=z.copy(), biomassgrowth=z.copy(),
                   NPP=z.copy(), NPP_pot=z.copy(), new_lmass=z.copy(),
                   leaf_litter=z.copy(), C_consumption=z.copy(),
                   Nleafdemand=z.copy(), Nleaf_litter=z.copy(),
                   N_leaf=z.copy(), Pleafdemand=z.copy(),
                   Pleaf_litter=z.copy(), P_leaf=z.copy(),
                   Kleafdemand=z.copy(), Kleaf_litter=z.copy(),
                   K_leaf=z.copy(), finerootlitter=z.copy(),
                   n_finerootlitter=z.copy(), p_finerootlitter=z.copy(),
                   k_finerootlitter=z.copy(), nonwoodylitter=z.copy(),
                   n_nonwoodylitter=z.copy(), p_nonwoodylitter=z.copy(),
                   k_nonwoodylitter=z.copy(), woodylitter=z.copy(),
                   n_woodylitter=z.copy(), p_woodylitter=z.copy(),
                   k_woodylitter=z.copy(),
                   woody_litter_mort=z.copy(), n_woody_litter_mort=z.copy(),
                   p_woody_litter_mort=z.copy(), k_woody_litter_mort=z.copy(),
                   non_woody_litter_mort=z.copy(),
                   n_non_woody_litter_mort=z.copy(),
                   p_non_woody_litter_mort=z.copy(),
                   k_non_woody_litter_mort=z.copy(),
                   n_demand=z.copy(), p_demand=z.copy(), k_demand=z.copy(),
                   basNdemand=z.copy(), basPdemand=z.copy(),
                   basKdemand=z.copy(),
                   harvested_volume=z.copy(), harvested_log_volume=z.copy(),
                   harvested_pulp_volume=z.copy(), harvested_biomass=z.copy(),
                   harvested_stems=z.copy(), nonwoody_lresid=z.copy(),
                   n_nonwoody_lresid=z.copy(), p_nonwoody_lresid=z.copy(),
                   k_nonwoody_lresid=z.copy(), woody_lresid=z.copy(),
                   n_woody_lresid=z.copy(), p_woody_lresid=z.copy(),
                   k_woody_lresid=z.copy())


def _mock_gv_state():
    from supersusi.core.gvegetation import State
    return State(gv_tot=np.zeros(5), n_gv=np.zeros(5),
                 p_gv=np.zeros(5), k_gv=np.zeros(5))


def _mock_gv_outputs():
    from supersusi.core.gvegetation import Outputs
    z = np.zeros(5)
    return Outputs(gv_change=z.copy(), gv_field=z.copy(), gv_bot=z.copy(),
                   gv_leafmass=z.copy(), ds_litterfall=z.copy(),
                   h_litterfall=z.copy(), s_litterfall=z.copy(),
                   n_litter_nw=z.copy(), p_litter_nw=z.copy(),
                   k_litter_nw=z.copy(), n_litter_w=z.copy(),
                   p_litter_w=z.copy(), k_litter_w=z.copy(),
                   nup=z.copy(), pup=z.copy(), kup=z.copy(),
                   nonwoodylitter=z.copy(), woodylitter=z.copy())


def _mock_esom_state():
    from supersusi.core.esom import State
    return State(M=np.zeros((1, 5, 11)), i=0,
                 previous_mass=np.zeros(5), pH=np.zeros((1, 5)))


def _mock_esom_yr_outputs():
    from supersusi.core.esom import YearOutputs
    z = np.zeros(5)
    return YearOutputs(out=z.copy(), out_root_lyr=z.copy(),
                       out_below_root_lyr=z.copy(),
                       nonwoodylitter=z.copy(), woodylitter=z.copy(),
                       daily_cumulative_out=np.zeros((5, 365)))


def _mock_methane_outputs():
    from supersusi.core.methane import Outputs
    return Outputs(ch4=np.zeros(5), ch4_as_co2eq=np.zeros(5))


def _mock_fert_outputs():
    from supersusi.core.fertilization_types import Outputs
    return Outputs(pH_increment=0.0, nutrient_release={"N": np.zeros(5),
                                                        "P": np.zeros(5),
                                                        "K": np.zeros(5)})


def _mock_doc_export():
    from supersusi.core.esom import DOCExportOutputs
    z = np.zeros(5)
    return DOCExportOutputs(hmw=z.copy(), lmw=z.copy(),
                            hmwtoditch=z.copy(), lmwtoditch=z.copy(),
                            hmw_to_west=0.0, hmw_to_east=0.0,
                            lmw_to_west=0.0, lmw_to_east=0.0)


def _mock_strip_diag():
    from supersusi.core.strip import ResidenceTimeOutput
    return ResidenceTimeOutput(n=5, residence_time=np.zeros(5),
                                ixwest=(), ixeast=())


def _mock_canopylayer_outputs():
    from supersusi.core.canopylayer import Outputs
    z = np.zeros(5)
    return Outputs(stems=z.copy(), basalarea=z.copy(), hdom=z.copy(),
                   Dg=z.copy(), volume=z.copy(), leafarea=z.copy(),
                   leafmass=z.copy(), volumegrowth=z.copy(),
                   logvolume=z.copy(), pulpvolume=z.copy(), yi=z.copy(),
                   NPP=z.copy(), NPP_pot=z.copy(), new_lmass=z.copy(),
                   leaf_litter=z.copy(), C_consumption=z.copy(),
                   Nleafdemand=z.copy(), Nleaf_litter=z.copy(),
                   N_leaf=z.copy(), Pleafdemand=z.copy(),
                   Pleaf_litter=z.copy(), P_leaf=z.copy(),
                   Kleafdemand=z.copy(), Kleaf_litter=z.copy(),
                   K_leaf=z.copy(), finerootlitter=z.copy(),
                   n_finerootlitter=z.copy(), p_finerootlitter=z.copy(),
                   k_finerootlitter=z.copy(), nonwoodylitter=z.copy(),
                   n_nonwoodylitter=z.copy(), p_nonwoodylitter=z.copy(),
                   k_nonwoodylitter=z.copy(), woodylitter=z.copy(),
                   n_woodylitter=z.copy(), p_woodylitter=z.copy(),
                   k_woodylitter=z.copy(),
                   woody_litter_mort=z.copy(), n_woody_litter_mort=z.copy(),
                   p_woody_litter_mort=z.copy(), k_woody_litter_mort=z.copy(),
                   non_woody_litter_mort=z.copy(),
                   n_non_woody_litter_mort=z.copy(),
                   p_non_woody_litter_mort=z.copy(),
                   k_non_woody_litter_mort=z.copy(),
                   n_demand=z.copy(), p_demand=z.copy(), k_demand=z.copy(),
                   basNdemand=z.copy(), basPdemand=z.copy(),
                   basKdemand=z.copy(),
                   leafmax=z.copy(), leafmin=z.copy())
