# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:38:10 2018

@author: lauren
"""

from supersusi.io.execution_config import SimulationParams

from typing import Any, cast
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import datetime

from supersusi.io.metadata_model import SimulationMetaData
from supersusi.io.susi_parameter_model import (
    SusiParams,
    AshFertilizationParameters,
    StandardNPKFertilizationParameters,
)
from supersusi.core.strip import drain_depth_development

from supersusi.core import fertilization_types
import supersusi.core.canopylayer as canopylayer
from supersusi.core.susi_utils import (
    get_temp_sum,
    heterotrophic_respiration_yr,
    ojanen_2019,
    rew_drylimit,
)
import supersusi.io.susi_io as susi_io
from supersusi.io.outputs import Outputs
import supersusi.io.utils as io_utils
from supersusi.io.forcing_weather import read_FMI_weather, WeatherForcings

from supersusi.core import (
    stand,
    esom,
    strip,
    methane,
    temperature,
    fertilization,
    mosslayer,
    gvegetation,
    canopygrid,
)
from supersusi.core.fertilization_models import ash, npk, no_fertilization


@dataclass(frozen=True)
class ModuleParams:
    stand: stand.Params
    esom: esom.Params
    canopygrid: canopygrid.Params
    mosslayer: mosslayer.Params
    methane: methane.Params
    temperature: temperature.Params
    fertilization: fertilization.Params
    strip: strip.Params
    gvegetation: gvegetation.Params


@dataclass(frozen=True)
class ModuleComputedConstants:
    stand: stand.ComputedConstants
    esom: esom.ComputedConstants
    canopygrid: canopygrid.ComputedConstants
    mosslayer: mosslayer.ComputedConstants
    methane: methane.ComputedConstants
    temperature: temperature.ComputedConstants
    fertilization: fertilization.ComputedConstants
    strip: strip.ComputedConstants
    gvegetation: gvegetation.ComputedConstants

    # Extra constants threaded through from SusiParams (not yet split)
    age: np.ndarray              # (n,) tree age per column
    depoN: float                 # atmospheric N deposition
    depoP: float                 # atmospheric P deposition
    depoK: float                 # atmospheric K deposition
    spara: Any                   # Pydantic SusiParams.site_parameters (temporary bridge)
    photo_parameters: Any        # photo_parameters from SusiParams (temporary bridge)


@dataclass(frozen=True)
class DailyState:
    canopy: canopygrid.State
    moss: mosslayer.State
    strip: strip.State
    peat_T: temperature.State


@dataclass(frozen=True)
class AnnualState:
    stand: stand.State
    stand_outputs: stand.Outputs
    gv: gvegetation.State
    esom_mass: esom.State
    esom_N: esom.State
    esom_P: esom.State
    esom_K: esom.State


@dataclass(frozen=True)
class AllState:
    daily: DailyState
    annual: AnnualState


@dataclass(frozen=True)
class DailyForcing:
    T: float
    Prec: float
    Rg: float
    Par: float
    VPD: float
    h0ts_west: float
    h0ts_east: float


@dataclass(frozen=True)
class DailyOutputs:
    wtd: np.ndarray
    afp: np.ndarray
    T_soil_hydro: np.ndarray
    delta: np.ndarray
    total_runoff: np.ndarray
    surface_runoff: np.ndarray
    interc: np.ndarray
    evap: np.ndarray
    et: np.ndarray
    transpi: np.ndarray
    efloor: np.ndarray
    swe: np.ndarray
    H: np.ndarray
    runoffwest: np.ndarray
    roffeast: np.ndarray


@dataclass(frozen=True)
class AnnualForcing:
    daily_T: np.ndarray
    daily_Rg: np.ndarray
    daily_VPD: np.ndarray
    daily_Prec: np.ndarray
    daily_Par: np.ndarray
    h0ts_west: np.ndarray
    h0ts_east: np.ndarray
    valid_days: int
    calendar_year: int
    temp_sum: np.ndarray
    do_cutting: bool
    cutting_to_ba: float


@dataclass(frozen=True)
class AnnualOutputs:
    daily: DailyOutputs
    stand: stand.Outputs
    stand_dom: canopylayer.Outputs
    stand_sub: canopylayer.Outputs
    stand_under: canopylayer.Outputs
    gv: gvegetation.Outputs
    esom_mass: esom.YearOutputs
    esom_N: esom.YearOutputs
    esom_P: esom.YearOutputs
    esom_K: esom.YearOutputs
    methane: methane.Outputs
    fertilization: fertilization_types.Outputs
    Rhet: np.ndarray
    soil_co2_balance: np.ndarray
    doc_export: esom.DOCExportOutputs
    strip_diag: strip.ResidenceTimeOutput


@dataclass(frozen=True)
class SimulationOutput:
    annual: list[AnnualOutputs]
    final_state: AllState


def _create_output_folder(simulation_metadata: SimulationMetaData) -> None:
    assert simulation_metadata.experiment_folder_path is not None
    assert not simulation_metadata.experiment_folder_path.is_dir()
    assert not simulation_metadata.experiment_folder_path.exists()
    io_utils.create_folder(path=simulation_metadata.experiment_folder_path)
    return None


def write_params_and_metadata(
    simulation_metadata: SimulationMetaData, susi_params: SusiParams
) -> None:
    simulation_metadata.record_end_timestamp()
    simulation_metadata.dump_json_to_file()
    susi_params.dump_json_to_file(
        filepath=simulation_metadata.parameter_output_filepath
    )
    return None


def _compute_constants(
    module_params: ModuleParams,
    susi_params: SusiParams,
    weather_data: pd.DataFrame,
) -> ModuleComputedConstants:
    lat = weather_data["lat"].iloc[0]
    lon = weather_data["lon"].iloc[0]

    stand_cc = stand.compute_constants(
        module_params.stand,
        susi_params.allometry_parameters,
    )
    return ModuleComputedConstants(
        stand=stand_cc,
        esom=esom.compute_all_constants(module_params.esom),
        canopygrid=canopygrid.ComputedConstants(),
        mosslayer=mosslayer.compute_constants(module_params.mosslayer),
        methane=methane.ComputedConstants(),
        temperature=temperature.compute_constants(
            params=module_params.temperature,
            T_air_mean=weather_data["T"].mean(),
        ),
        fertilization=fertilization.compute_constants(module_params.fertilization),
        strip=strip.compute_constants(module_params.strip),
        gvegetation=gvegetation.compute_constants(
            params=module_params.gvegetation,
            lon=lon,
            lat=lat,
            dominant_tree_species=stand_cc.dominant.tree_species,
        ),
        age=susi_params.site_parameters.age["dominant"],
        depoN=susi_params.site_parameters.depoN,
        depoP=susi_params.site_parameters.depoP,
        depoK=susi_params.site_parameters.depoK,
        spara=susi_params.site_parameters,
        photo_parameters=susi_params.photo_parameters,
    )


def _build_params(susi_params: SusiParams) -> ModuleParams:
    """
    Takes params coming from the user-facing Pydantic model,
    converts them into classes to pass to the modules.
    """

    match susi_params.site_parameters.fertilization:
        case AshFertilizationParameters():
            fertilization_params = ash.Params(
                n_cols=susi_params.site_parameters.n,
                simulation_end_year=susi_params.simulation_config.end_date.year,
                fpara=susi_params.site_parameters.fertilization,
            )
        case StandardNPKFertilizationParameters():
            fertilization_params = npk.Params(
                n_cols=susi_params.site_parameters.n,
                fpara=susi_params.site_parameters.fertilization,
            )
        case None:
            fertilization_params = no_fertilization.Params(
                n_cols=susi_params.site_parameters.n
            )

    org = susi_params.organic_layer_parameters
    n = susi_params.site_parameters.n

    c = susi_params.canopy_parameters

    sp = susi_params.site_parameters
    return ModuleParams(
        esom=esom.build_params(
            n=sp.n,
            nLyrs=sp.nLyrs,
            dzLyr=sp.dzLyr,
            vonP=sp.vonP,
            vonP_top=sp.vonP_top,
            vonP_bottom=sp.vonP_bottom,
            bd_top=cast(list[float], sp.bd_top) if sp.bd_top is not None else None,
            bd_bottom=sp.bd_bottom,
            sfc=sp.sfc,
            h_mor=cast(float, sp.h_mor),
            rho_mor=sp.rho_mor,
            enable_peattop=sp.enable_peattop,
            enable_peatmiddle=sp.enable_peatmiddle,
            enable_peatbottom=sp.enable_peatbottom,
            peatN=sp.peatN,
            peatP=sp.peatP,
            peatK=sp.peatK,
        ),
        stand=stand.Params(
            dominant=canopylayer.Params(
                name="dominant",
                ncols=sp.n,
                nlyrs=np.array(sp.canopylayers.dominant, dtype=int),
                sfc=sp.sfc.copy(),
            ),
            subdominant=canopylayer.Params(
                name="subdominant",
                ncols=sp.n,
                nlyrs=np.array(sp.canopylayers.subdominant, dtype=int),
                sfc=sp.sfc.copy(),
            ),
            under=canopylayer.Params(
                name="under",
                ncols=sp.n,
                nlyrs=np.array(sp.canopylayers.under, dtype=int),
                sfc=sp.sfc.copy(),
            ),
        ),
        canopygrid=canopygrid.Params(
            dt=c.dt,
            cf=np.ones(n) * c.state.cf,
            lai_decid_max=np.ones(n) * c.state.lai_decid_max,
            wmax=c.interception.wmax,
            wmaxsnow=c.interception.wmaxsnow,
            kmelt=c.snow.kmelt,
            kfreeze=c.snow.kfreeze,
            r=c.snow.r,
            amax_init=c.physpara.amax_init,
            g1_conif=c.physpara.g1_conif,
            g1_decid=c.physpara.g1_decid,
            kp=c.physpara.kp,
            q50=c.physpara.q50,
            gsoil=c.physpara.gsoil,
            zmeas=c.flow.zmeas,
            zground=c.flow.zground,
            zo_ground=c.flow.zo_ground,
            smax=c.phenology.smax,
            tau=c.phenology.tau,
            xo=c.phenology.xo,
            fmin=c.phenology.fmin,
            P=101300.0,
            U=2.0,
            CO2=380.0,
            initial_W=np.ones(n) * c.state.w,
            initial_SWE=np.ones(n) * c.state.swe,
        ),
        mosslayer=mosslayer.Params(
            org_depth=np.ones(n) * org.org_depth,
            org_poros=np.ones(n) * org.org_poros,
            org_fc=np.ones(n) * org.org_fc,
            org_rw=np.ones(n) * org.org_rw,
            pond_storage_max=np.ones(n) * org.pond_storage_max,
            org_sat=np.ones(n) * org.org_sat,
            pond_storage_initial=np.ones(n) * org.pond_storage,
        ),
        methane=methane.Params(),
        temperature=temperature.Params(
            n_layers_hydro=susi_params.site_parameters.nLyrs,
            dz=susi_params.site_parameters.dzLyr,
            timestep=susi_params.site_parameters.peat_temperature.timestep,
            n_subtimesteps=susi_params.site_parameters.peat_temperature.n_subtimesteps,
            D=susi_params.site_parameters.peat_temperature.D,
            heat_of_vaporization=susi_params.site_parameters.peat_temperature.heat_of_vaporization,
        ),
        fertilization=fertilization_params,
        strip=strip.Params(
            nLyrs=sp.nLyrs,
            dzLyr=sp.dzLyr,
            vonP=sp.vonP,
            vonP_top=np.array(sp.vonP_top),
            vonP_bottom=sp.vonP_bottom,
            peat_type=[p.value for p in sp.peat_type],
            peat_type_bottom=[p.value for p in sp.peat_type_bottom],
            bd_top=np.array(sp.bd_top) if sp.bd_top is not None else None,
            bd_bottom=sp.bd_bottom,
            anisotropy=sp.anisotropy,
            L=sp.L,
            n=sp.n,
            slope=sp.slope,
            initial_h=sp.initial_h,
        ),
        gvegetation=gvegetation.Params(
            num_nodes=n,
            site_fertility_class=susi_params.site_parameters.sfc,
        ),
    )


def run_legacy(simulation_params: SimulationParams):
    susi_params = simulation_params.susi_params
    simulation_metadata = simulation_params.metadata

    # Create output folder for the simulation results.
    _create_output_folder(simulation_metadata)

    # Read weather data
    weather_data = read_FMI_weather(
        ID=0,
        start_date=susi_params.simulation_config.start_date,
        end_date=susi_params.simulation_config.end_date,
        sourcefile=susi_params.weather_parameters.FMI_weather_filepath,
    )

    # Assemble per-module parameters
    module_params = _build_params(susi_params)

    # Compute other constants
    computed_constants = _compute_constants(
        module_params,
        susi_params,
        weather_data,
    )

    # Numerical scratch buffers for hydrology PDEs
    strip_numerical_buffer = strip.make_numerical_buffer(module_params.strip)

    # simulation time in days
    n_simulation_days = (
        susi_params.simulation_config.end_date
        - susi_params.simulation_config.start_date
    ).days + 1

    # simulation time in years
    n_simulation_years = (
        susi_params.simulation_config.end_date.year
        - susi_params.simulation_config.start_date.year
        + 1
    )
    temperature_sun_days_degree = get_temp_sum(weather_data)

    out = Outputs(
        n_scenarios=len(susi_params.site_parameters.ditch_depth_east),
        n_cols=susi_params.site_parameters.n,
        n_days=n_simulation_days,
        n_years=n_simulation_years,
        n_layers=susi_params.site_parameters.nLyrs,
        fname=simulation_metadata.netcdf_output_filepath,
    )

    # Initialize output netcdf variable
    out.initialize(strip_constants=computed_constants.strip)

    out.write_paras(
        sfc=susi_params.site_parameters.sfc,
        dominant_sp=computed_constants.stand.dominant.tree_species,
        subdominant_sp=computed_constants.stand.subdominant.tree_species,
        under_sp=computed_constants.stand.under.tree_species,
    )
    # describe site parameters for user
    susi_io.print_site_description(susi_params.site_parameters)

    # ********* Above ground hydrology initialization ***************

    stand_state, _, *_ = stand.compute_initial_state(
        module_params.stand,
        computed_constants.stand,
        susi_params.site_parameters.age,
        susi_params.site_parameters.n,
    )
    canopy_state = canopygrid.compute_initial_state(module_params.canopygrid)
    canopy_state = canopygrid.update_amax(stand_state.nut_stat, canopy_state)

    moss_state = mosslayer.compute_initial_state(
        module_params.mosslayer, computed_constants.mosslayer
    )
    peat_T_state = temperature.compute_initial_state(
        computed_constants=computed_constants.temperature
    )

    # ******** Soil and strip parameterization *************************

    # Evapotranspiration, mm/day
    ets = np.zeros((n_simulation_days, susi_params.site_parameters.n))

    # ********initialize result arrays***************************
    # number of ditch depth scenarios (used in comparison of management)
    rounds = len(susi_params.site_parameters.ditch_depth_east)

    n = susi_params.site_parameters.n
    stpout = {}
    stpout["dwts"] = np.zeros(
        (rounds, n_simulation_days, n), dtype=float
    )  # water table depths, m
    stpout["afps"] = np.zeros(
        (rounds, n_simulation_days, n), dtype=float
    )  # air-filled porosity (m3 m-3)
    stpout["deltas"] = np.zeros((rounds, n_simulation_days, n), dtype=float)
    stpout["hts"] = np.zeros(
        (rounds, n_simulation_days, n), dtype=float
    )  # water table heights, m
    stpout["runoff"] = np.zeros(
        (rounds, n_simulation_days), dtype=float
    )  # daily total runoff, m
    stpout["runoffwest"] = np.zeros(
        (rounds, n_simulation_days), dtype=float
    )  # daily runoff from west ditch, m
    stpout["runoffeast"] = np.zeros(
        (rounds, n_simulation_days), dtype=float
    )  # daily runoff from east ditch, m
    stpout["surfacerunoff"] = np.zeros(
        (rounds, n_simulation_days, n), dtype=float
    )  # daily surface runoff, m

    peat_temperatures = np.zeros(
        (rounds, n_simulation_days, susi_params.site_parameters.nLyrs)
    )  # daily peat temperature profiles

    intercs = np.zeros((rounds, n_simulation_days, susi_params.site_parameters.n))
    evaps = np.zeros_like(intercs)
    ETs = np.zeros_like(intercs)
    transpis = np.zeros_like(intercs)
    efloors = np.zeros_like(intercs)
    swes = np.zeros_like(intercs)

    # ***********Scenario loop ********************************************************

    for n_ditch_scen, dr in enumerate(
        zip(
            susi_params.site_parameters.ditch_depth_west,
            susi_params.site_parameters.ditch_depth_20y_west,
            susi_params.site_parameters.ditch_depth_east,
            susi_params.site_parameters.ditch_depth_20y_east,
        )
    ):
        dwt = susi_params.site_parameters.initial_h * np.ones(
            susi_params.site_parameters.n
        )  # set the initial WT for the scenario
        hdr_west, hdr20y_west, hdr_east, hdr20y_east = (
            dr  # drain depth [m] in the beginning and after 20 yrs
        )
        h0ts_west = drain_depth_development(
            n_simulation_days, hdr_west, hdr20y_west
        )  # compute daily values for drain bottom boundary condition
        h0ts_east = drain_depth_development(
            n_simulation_days, hdr_east, hdr20y_east
        )  # compute daily values for drain bottom boundary condition

        # ---- Initialize integrative output arrays (outputs in nodewise sums) -------------------------------

        stand_state, stand_out, dom_out_init, sub_out_init, under_out_init = (
            stand.compute_initial_state(
                module_params.stand,
                computed_constants.stand,
                susi_params.site_parameters.age,
                susi_params.site_parameters.n,
            )
        )

        out.write_scen(n_ditch_scen, hdr_west, hdr_east)

        out.write_stand(
            n_ditch_scen,
            0,
            stand_out,
            stand_state,
            previous_nut_stat=stand_state.previous_nut_stat,
        )
        out.write_canopy_layer(
            n_ditch_scen, 0, "dominant", stand_state.dominant, dom_out_init
        )
        out.write_canopy_layer(
            n_ditch_scen, 0, "subdominant", stand_state.subdominant, sub_out_init
        )
        out.write_canopy_layer(
            n_ditch_scen, 0, "under", stand_state.under, under_out_init
        )

        gv_state = gvegetation.compute_initial_state(
            module_params.gvegetation, computed_constants.gvegetation
        )
        gv_state, gv_outputs = gvegetation.run_timestep(
            module_params.gvegetation,
            computed_constants.gvegetation,
            gvegetation.assemble_inputs(
                ts=temperature_sun_days_degree,
                vol=stand_out.volume,
                stems=stand_out.stems,
                ba=stand_out.basalarea,
                age=susi_params.site_parameters.age["dominant"],
            ),
            gv_state,
        )
        out.write_groundvegetation(n_ditch_scen, 0, gv_state, gv_outputs)

        state_mass = esom.compute_initial_state(
            module_params.esom.mass, computed_constants.esom.mass
        )
        state_N = esom.compute_initial_state(
            module_params.esom.n, computed_constants.esom.n
        )
        state_P = esom.compute_initial_state(
            module_params.esom.p, computed_constants.esom.p
        )
        state_K = esom.compute_initial_state(
            module_params.esom.k, computed_constants.esom.k
        )

        out.write_esom(n_ditch_scen, 0, "Mass", state_mass, inivals=True)
        out.write_esom(n_ditch_scen, 0, "N", state_N, inivals=True)
        out.write_esom(n_ditch_scen, 0, "P", state_P, inivals=True)
        out.write_esom(n_ditch_scen, 0, "K", state_K, inivals=True)

        strip_state = strip.compute_initial_state(
            module_params.strip, computed_constants.strip
        )

        d = 0  # day index
        start = 0  # day counter in annual loop
        # *************** Annual loop *****************************************************************
        # simulation_year starts at 1
        # calendar_year starts at the year of the starting date, e.g., 2004.
        for simulation_year, calendar_year in enumerate(
            range(
                susi_params.simulation_config.start_date.year,
                susi_params.simulation_config.end_date.year + 1,
            ),
            start=1,
        ):
            days = (
                datetime.datetime(calendar_year, 12, 31)
                - datetime.datetime(calendar_year, 1, 1)
            ).days + 1

            canopy_state = canopygrid.update_amax(stand_state.nut_stat, canopy_state)

            # **********  Daily loop ************************************************************
            for dd in range(days):  # day loop
                # -------Canopy hydrology--------------------------
                reww = rew_drylimit(
                    dwt
                )  # for each column: moisture limitation from ground water level (Feddes-function)
                forcings = WeatherForcings(
                    T=weather_data.iloc[d, 4],
                    Prec=weather_data.iloc[d, 7],
                    Rg=weather_data.iloc[d, 8],
                    Par=weather_data.iloc[d, 10],
                    VPD=weather_data.iloc[d, 13],
                )

                inputs = canopygrid.assemble_inputs(
                    forcings,
                    hc=stand_out.hdom,
                    LAIconif=stand_out.leafarea,
                    Rew=reww,
                    beta=moss_state.Ree,
                )
                canopy_state, canopy_out = canopygrid.run_timestep(
                    module_params.canopygrid, inputs, canopy_state
                )
                potinf = canopy_out.potinf
                interc = canopy_out.interc
                evap = canopy_out.evap
                ET = canopy_out.et
                transpi = canopy_out.transpi
                efloor = canopy_out.efloor
                SWE = canopy_out.swe

                intercs[n_ditch_scen, d, :] = interc
                evaps[n_ditch_scen, d, :] = evap
                ETs[n_ditch_scen, d, :] = ET
                transpis[n_ditch_scen, d, :] = transpi
                efloors[n_ditch_scen, d, :] = efloor

                moss_state, moss_interception_outputs = mosslayer.run_interception(
                    computed_constants.mosslayer,
                    mosslayer.assemble_interception_inputs(potinf=potinf, evap=efloor),
                    moss_state,
                )
                # TODO: This should go to some module!
                potinf, efloor = (
                    moss_interception_outputs.potinf,
                    moss_interception_outputs.evap,
                )
                stpout["deltas"][n_ditch_scen, d, :] = (
                    potinf - transpi
                )  # water flux thru soil surface
                ets[d] = efloor + transpi + interc  # evapotranspiration components

                if d % 365 == 0:
                    print(
                        "  - day #",
                        d,
                        " hdom ",
                        np.round(np.mean(stand_out.hdom), 2),
                        " m, ",
                        "LAI ",
                        np.round(np.mean(stand_out.leafarea), 2),
                        " m2 m-2",
                    )

                # --------Soil hydrology-----------------
                exfil_out = strip.compute_exfil(
                    strip_state,
                    computed_constants.strip,
                    h0ts_west[d],
                    h0ts_east[d],
                    stpout["deltas"][n_ditch_scen, d, :],
                )  # how much water soil cannot store
                moss_state, moss_rf_out = mosslayer.run_returnflow(
                    module_params.mosslayer,
                    computed_constants.mosslayer,
                    mosslayer.assemble_returnflow_inputs(
                        rflow=exfil_out.exfil,
                        interception_mbe=moss_interception_outputs.mbe,
                    ),
                    moss_state,
                )
                strip_state, ts_out = strip.run_timestep(
                    module_params.strip,
                    computed_constants.strip,
                    strip_state,
                    strip.assemble_timestep_inputs(
                        h0ts_west=h0ts_west[d],
                        h0ts_east=h0ts_east[d],
                        exfil_out=exfil_out,
                        moss_rf_out=moss_rf_out,
                    ),
                    buffer=strip_numerical_buffer,
                )  # strip/peat hydrology
                stpout["dwts"][n_ditch_scen, d, :] = (
                    strip_state.H - computed_constants.strip.ele
                )
                stpout["hts"][n_ditch_scen, d, :] = strip_state.H
                stpout["afps"][n_ditch_scen, d, :] = ts_out.afp
                stpout["runoff"][n_ditch_scen, d] = ts_out.roff + np.mean(
                    moss_rf_out.surface_runoff
                )
                stpout["runoffwest"][n_ditch_scen, d] = ts_out.roffwest
                stpout["runoffeast"][n_ditch_scen, d] = ts_out.roffeast
                stpout["surfacerunoff"][n_ditch_scen, d, :] = moss_rf_out.surface_runoff

                peat_T_state, _ = temperature.run_timestep(
                    params=module_params.temperature,
                    computed_constants=computed_constants.temperature,
                    inputs=temperature.assemble_inputs(
                        T_air=forcings.T, swe=SWE, efloor=efloor
                    ),
                    state=peat_T_state,
                )
                peat_temperatures[n_ditch_scen, d, :] = peat_T_state.T_soil[
                    : module_params.temperature.n_layers_hydro
                ]

                swes[n_ditch_scen, d] = np.mean(SWE)  # snow water equivalent
                d += 1
            # ******* End of daily loop*****************************

            # ----- Hydrology and temperature-related variables to time-indexed dataframes -----------------
            sday = datetime.datetime(calendar_year, 1, 1)  # start day of the year
            df_peat_temperatures = pd.DataFrame(
                peat_temperatures[n_ditch_scen, start : start + days, :],
                index=pd.date_range(sday, periods=days),
            )  # all peat temperatures
            dfwt = pd.DataFrame(
                stpout["dwts"][n_ditch_scen, start : start + days, :],
                index=pd.date_range(sday, periods=days),
            )  # daily water table data frame
            dfafp = pd.DataFrame(
                stpout["afps"][n_ditch_scen, start : start + days, :],
                index=pd.date_range(sday, periods=days),
            )  # air filled porosity

            out.write_cpy(
                n_ditch_scen,
                start,
                days,
                simulation_year,
                intercs[n_ditch_scen, start : start + days, :],
                evaps[n_ditch_scen, start : start + days, :],
                ETs[n_ditch_scen, start : start + days, :],
                transpis[n_ditch_scen, start : start + days, :],
                efloors[n_ditch_scen, start : start + days, :],
                swes[n_ditch_scen, start : start + days, :],
            )

            out.write_temperature(
                n_ditch_scen,
                start,
                days,
                peat_temperatures[n_ditch_scen, start : start + days, :],
            )

            mean_dwt = dfwt.mean(axis=0).values
            strip_diag = strip.compute_residence_time(
                module_params.strip, computed_constants.strip, mean_dwt
            )
            out.write_strip(
                n_ditch_scen,
                start,
                days,
                calendar_year,
                simulation_year,
                dfwt,
                stpout,
                susi_params.output_parameters,
                strip_diag,
            )

            # **************  Biogeochemistry ***********************************
            v = stand_out.volume
            _, _co2, Rhet = heterotrophic_respiration_yr(
                df_peat_temperatures,
                calendar_year,
                dfwt,
                v,
                susi_params.site_parameters,
            )  # Rhet is total annual heterotrophic respiration in kg/ha/yr CO2, per computation node
            soil_co2_balance = ojanen_2019(
                susi_params.site_parameters, calendar_year, dfwt
            )
            out.write_ojanen(n_ditch_scen, simulation_year, Rhet, soil_co2_balance)

            gv_state, gv_outputs = gvegetation.run_timestep(
                module_params.gvegetation,
                computed_constants.gvegetation,
                gvegetation.assemble_inputs(
                    ts=temperature_sun_days_degree,
                    vol=stand_out.volume,
                    stems=stand_out.stems,
                    ba=stand_out.basalarea,
                    age=susi_params.site_parameters.age["dominant"],
                ),
                gv_state,
            )

            _stand_inputs = stand.Inputs(
                photopara=susi_params.photo_parameters,
                forc=weather_data.loc[str(calendar_year)],
                wt=dfwt.loc[str(calendar_year)],
                afp=dfafp.loc[str(calendar_year)],
                n_supply=np.zeros(susi_params.site_parameters.n),
                p_supply=np.zeros(susi_params.site_parameters.n),
                k_supply=np.zeros(susi_params.site_parameters.n),
                groundvegetation_outputs=gv_outputs,
                previous_nut_stat=stand_state.previous_nut_stat,
                calendar_year=calendar_year,
            )
            stand_state, stand_out, dom_out, sub_out, under_out = stand.grow_stand(
                stand_state, computed_constants.stand, _stand_inputs
            )

            # --------- Locate cuttings here--------------------
            print("calculating year " + str(calendar_year))
            if calendar_year == susi_params.site_parameters.cutting_yr:
                print("xxxxxxxxxxxx   VOL before cutting xxxxxxxxxxxxxxxx")
                print(str(np.round(np.mean(stand_out.volume), 1)))
                print(
                    "cutting now "
                    + str(calendar_year)
                    + " from basal area "
                    + str(np.round(np.mean(stand_out.basalarea), 1))
                    + " to "
                    + str(susi_params.site_parameters.cutting_to_ba)
                )

                _cut_inputs = replace(
                    _stand_inputs,
                    cutting_to_ba=susi_params.site_parameters.cutting_to_ba,
                )
                stand_state, stand_out, _cutting_out = stand.cut_stand(
                    stand_state, computed_constants.stand, stand_out, _cut_inputs
                )
            else:
                _cutting_out = canopylayer.CuttingOutputs(
                    harvested_volume=np.zeros(susi_params.site_parameters.n),
                    harvested_log_volume=np.zeros(susi_params.site_parameters.n),
                    harvested_pulp_volume=np.zeros(susi_params.site_parameters.n),
                    harvested_biomass=np.zeros(susi_params.site_parameters.n),
                    harvested_stems=np.zeros(susi_params.site_parameters.n),
                    nonwoody_lresid=np.zeros(susi_params.site_parameters.n),
                    n_nonwoody_lresid=np.zeros(susi_params.site_parameters.n),
                    p_nonwoody_lresid=np.zeros(susi_params.site_parameters.n),
                    k_nonwoody_lresid=np.zeros(susi_params.site_parameters.n),
                    woody_lresid=np.zeros(susi_params.site_parameters.n),
                    n_woody_lresid=np.zeros(susi_params.site_parameters.n),
                    p_woody_lresid=np.zeros(susi_params.site_parameters.n),
                    k_woody_lresid=np.zeros(susi_params.site_parameters.n),
                )

            #      """ ATTN distinct stem mortaility from the logging -> fate of stems different!!!"""
            # ---------- Organic matter decomposition and nutrient release-----------------

            # ---------------- Fertilization --------------------------------

            _, fertilization_outputs = fertilization.run_timestep(
                params=module_params.fertilization,
                computed_constants=computed_constants.fertilization,
                inputs=fertilization.assemble_inputs(
                    params=module_params.fertilization, calendar_year=calendar_year
                ),
            )
            pH_inc = fertilization_outputs.pH_increment
            state_mass = esom.State(
                M=state_mass.M,
                i=state_mass.i,
                previous_mass=state_mass.previous_mass,
                pH=esom.update_soil_pH(state_mass.pH, module_params.esom.mass, pH_inc),
            )
            state_N = esom.State(
                M=state_N.M,
                i=state_N.i,
                previous_mass=state_N.previous_mass,
                pH=esom.update_soil_pH(state_N.pH, module_params.esom.n, pH_inc),
            )
            state_P = esom.State(
                M=state_P.M,
                i=state_P.i,
                previous_mass=state_P.previous_mass,
                pH=esom.update_soil_pH(state_P.pH, module_params.esom.p, pH_inc),
            )
            state_K = esom.State(
                M=state_K.M,
                i=state_K.i,
                previous_mass=state_K.previous_mass,
                pH=esom.update_soil_pH(state_K.pH, module_params.esom.k, pH_inc),
            )

            out.write_fertilization(
                n_ditch_scen, simulation_year, fertilization_outputs
            )

            """
            TODO:
                logging resdues to output, to be joined wtih litter in figures
                harvested volume and biomass to outputs
                construct balcances at the end of simulation, join to outputs
            """
            weather_yr = weather_data.loc[str(calendar_year)]
            nonwoodylitter = (
                stand_out.nonwoodylitter
                + stand_out.nonwoody_lresid
                + stand_out.non_woody_litter_mort
                + gv_outputs.nonwoodylitter
            ) / 10000.0
            woodylitter = (
                stand_out.woodylitter
                + stand_out.woody_lresid
                + stand_out.woody_litter_mort
                + gv_outputs.woodylitter
            ) / 10000.0
            inputs_mass = esom.assemble_inputs(
                tair_ts=weather_yr["T"].values,
                tp_top_ts=df_peat_temperatures.iloc[:, 2].values,
                tp_middle_ts=df_peat_temperatures.iloc[:, 8].values,
                tp_bottom_ts=df_peat_temperatures.iloc[:, 9].values,
                water_tables=dfwt.values,
                nonwoodylitter=nonwoodylitter,
                woodylitter=woodylitter,
            )
            state_mass, yr_out_mass = esom.run_yr(
                module_params.esom.mass,
                computed_constants.esom.mass,
                inputs_mass,
                state_mass,
            )
            assert yr_out_mass.daily_cumulative_out is not None
            doc_export = esom.compose_export(
                yr_out_mass.daily_cumulative_out,
                strip_diag,
                df_peat_temperatures.iloc[:, 2].values,
                susi_params.site_parameters.n,
            )
            out.write_esom(
                n_ditch_scen, simulation_year, "Mass", state_mass, yr_out_mass
            )

            n_nonwoodylitter = (
                stand_out.n_nonwoodylitter
                + stand_out.n_nonwoody_lresid
                + stand_out.n_non_woody_litter_mort
                + gv_outputs.n_litter_nw
            ) / 10000.0
            n_woodylitter = (
                stand_out.n_woodylitter
                + stand_out.n_woody_lresid
                + stand_out.n_woody_litter_mort
                + stand_out.n_woody_litter_mort
                + gv_outputs.n_litter_w
            ) / 10000.0
            inputs_N = esom.assemble_inputs(
                tair_ts=weather_yr["T"].values,
                tp_top_ts=df_peat_temperatures.iloc[:, 2].values,
                tp_middle_ts=df_peat_temperatures.iloc[:, 8].values,
                tp_bottom_ts=df_peat_temperatures.iloc[:, 9].values,
                water_tables=dfwt.values,
                nonwoodylitter=n_nonwoodylitter,
                woodylitter=n_woodylitter,
            )
            state_N, yr_out_N = esom.run_yr(
                module_params.esom.n, computed_constants.esom.n, inputs_N, state_N
            )
            out.write_esom(n_ditch_scen, simulation_year, "N", state_N, yr_out_N)

            p_nonwoodylitter = (
                stand_out.p_nonwoodylitter
                + stand_out.p_nonwoody_lresid
                + stand_out.p_non_woody_litter_mort
                + gv_outputs.p_litter_nw
            ) / 10000.0
            p_woodylitter = (
                stand_out.p_woodylitter
                + stand_out.p_woody_lresid
                + stand_out.p_woody_litter_mort
                + gv_outputs.p_litter_w
            ) / 10000.0
            inputs_P = esom.assemble_inputs(
                tair_ts=weather_yr["T"].values,
                tp_top_ts=df_peat_temperatures.iloc[:, 2].values,
                tp_middle_ts=df_peat_temperatures.iloc[:, 8].values,
                tp_bottom_ts=df_peat_temperatures.iloc[:, 9].values,
                water_tables=dfwt.values,
                nonwoodylitter=p_nonwoodylitter,
                woodylitter=p_woodylitter,
            )
            state_P, yr_out_P = esom.run_yr(
                module_params.esom.p, computed_constants.esom.p, inputs_P, state_P
            )
            out.write_esom(n_ditch_scen, simulation_year, "P", state_P, yr_out_P)

            k_nonwoodylitter = (
                stand_out.k_nonwoodylitter
                + stand_out.k_nonwoody_lresid
                + stand_out.k_non_woody_litter_mort
                + gv_outputs.k_litter_nw
            ) / 10000.0
            k_woodylitter = (
                stand_out.k_woodylitter
                + stand_out.k_woody_lresid
                + stand_out.k_woody_litter_mort
                + gv_outputs.k_litter_w
            ) / 10000.0
            inputs_K = esom.assemble_inputs(
                tair_ts=weather_yr["T"].values,
                tp_top_ts=df_peat_temperatures.iloc[:, 2].values,
                tp_middle_ts=df_peat_temperatures.iloc[:, 8].values,
                tp_bottom_ts=df_peat_temperatures.iloc[:, 9].values,
                water_tables=dfwt.values,
                nonwoodylitter=k_nonwoodylitter,
                woodylitter=k_woodylitter,
            )
            state_K, yr_out_K = esom.run_yr(
                module_params.esom.k, computed_constants.esom.k, inputs_K, state_K
            )
            out.write_esom(n_ditch_scen, simulation_year, "K", state_K, yr_out_K)

            _nut_inputs = replace(
                _stand_inputs,
                n_supply=yr_out_N.out_root_lyr
                + susi_params.site_parameters.depoN
                + fertilization_outputs.nutrient_release["N"],
                p_supply=yr_out_P.out_root_lyr
                + susi_params.site_parameters.depoP
                + fertilization_outputs.nutrient_release["P"],
                k_supply=yr_out_K.out_root_lyr
                + susi_params.site_parameters.depoK
                + fertilization_outputs.nutrient_release["K"],
            )
            stand_state = stand.update_nutrient_status(
                stand_state, stand_out, _nut_inputs
            )

            _, ch4_outputs = methane.run_timestep(
                inputs=methane.assemble_inputs(year=calendar_year, dfwt=dfwt),
            )
            out.write_methane(n_ditch_scen, simulation_year, ch4_outputs)

            # BUG: OOP writes previous_nut_stat AFTER update_nutrient_status
            #      (copy of nut_stat before update). Pass state.previous_nut_stat
            #      which was set by update_nutrient_status to nut_stat from this year.
            out.write_stand(
                n_ditch_scen,
                simulation_year,
                stand_out,
                stand_state,
                previous_nut_stat=stand_state.previous_nut_stat,
            )
            out.write_canopy_layer(
                n_ditch_scen,
                simulation_year,
                "dominant",
                stand_state.dominant,
                dom_out,
            )
            out.write_canopy_layer(
                n_ditch_scen,
                simulation_year,
                "subdominant",
                stand_state.subdominant,
                sub_out,
            )
            out.write_canopy_layer(
                n_ditch_scen, simulation_year, "under", stand_state.under, under_out
            )
            out.write_groundvegetation(
                n_ditch_scen, simulation_year, gv_state, gv_outputs
            )
            out.write_export(n_ditch_scen, simulation_year, doc_export)

            out.write_nutrient_balance(
                n_ditch_scen,
                simulation_year,
                "N",
                yr_out_N,
                susi_params.site_parameters.depoN,
                fertilization_outputs.nutrient_release["N"],
                stand_out.n_demand + stand_out.Nleafdemand,
                gv_outputs.nup,
            )
            out.write_nutrient_balance(
                n_ditch_scen,
                simulation_year,
                "P",
                yr_out_P,
                susi_params.site_parameters.depoP,
                fertilization_outputs.nutrient_release["P"],
                stand_out.p_demand + stand_out.Pleafdemand,
                gv_outputs.pup,
            )
            out.write_nutrient_balance(
                n_ditch_scen,
                simulation_year,
                "K",
                yr_out_K,
                susi_params.site_parameters.depoK,
                fertilization_outputs.nutrient_release["K"],
                stand_out.k_demand + stand_out.Kleafdemand,
                gv_outputs.kup,
            )

            out.write_carbon_balance(
                n_ditch_scen,
                simulation_year,
                stand_out,
                gv_outputs,
                yr_out_mass,
                doc_export,
                ch4_outputs,
            )
            #
            start = start + days  # starting point of the next year daily loop

    out.close()


def _build_annual_forcings(
    susi_params: SusiParams,
    params: ModuleParams,
    weather_data: pd.DataFrame,
) -> list[AnnualForcing]:
    n = susi_params.site_parameters.n
    start_year = susi_params.simulation_config.start_date.year
    end_year = susi_params.simulation_config.end_date.year

    hdr_west, hdr20y_west, hdr_east, hdr20y_east = next(
        zip(
            susi_params.site_parameters.ditch_depth_west,
            susi_params.site_parameters.ditch_depth_20y_west,
            susi_params.site_parameters.ditch_depth_east,
            susi_params.site_parameters.ditch_depth_20y_east,
        )
    )
    n_simulation_days = (
        susi_params.simulation_config.end_date
        - susi_params.simulation_config.start_date
    ).days + 1
    h0ts_west_all = drain_depth_development(
        n_simulation_days, hdr_west, hdr20y_west
    )
    h0ts_east_all = drain_depth_development(
        n_simulation_days, hdr_east, hdr20y_east
    )

    temp_sum_scalar = get_temp_sum(weather_data)
    temp_sum = np.full(n, temp_sum_scalar)

    forcings: list[AnnualForcing] = []
    day_offset = 0
    for yr in range(start_year, end_year + 1):
        ndays = (
            datetime.datetime(yr, 12, 31)
            - datetime.datetime(yr, 1, 1)
        ).days + 1
        w_yr = weather_data.iloc[day_offset: day_offset + ndays]

        forcing = AnnualForcing(
            daily_T=w_yr["T"].values.astype(float),
            daily_Rg=w_yr["Rg"].values.astype(float),
            daily_VPD=w_yr["vpd"].values.astype(float),
            daily_Prec=w_yr["Prec"].values.astype(float),
            daily_Par=w_yr["Par"].values.astype(float),
            h0ts_west=h0ts_west_all[day_offset: day_offset + ndays],
            h0ts_east=h0ts_east_all[day_offset: day_offset + ndays],
            valid_days=ndays,
            calendar_year=yr,
            temp_sum=temp_sum,
            do_cutting=yr == susi_params.site_parameters.cutting_yr,
            cutting_to_ba=susi_params.site_parameters.cutting_to_ba,
        )
        forcings.append(forcing)
        day_offset += ndays

    return forcings


def _write_outputs(
    simulation_metadata: SimulationMetaData,
    susi_params: SusiParams,
    module_params: ModuleParams,
    computed_constants: ModuleComputedConstants,
    annual_outputs: list[AnnualOutputs],
    annual_states: list[AllState],
    weather_data: pd.DataFrame,
    initial_state: AllState,
    init_dom_out: canopylayer.Outputs,
    init_sub_out: canopylayer.Outputs,
    init_under_out: canopylayer.Outputs,
    init_gv_out: gvegetation.Outputs,
    forcings: list[AnnualForcing],
) -> None:
    n_simulation_days = (
        susi_params.simulation_config.end_date
        - susi_params.simulation_config.start_date
    ).days + 1
    n_simulation_years = len(annual_outputs)
    n = susi_params.site_parameters.n

    out = Outputs(
        n_scenarios=1,
        n_cols=n,
        n_days=n_simulation_days,
        n_years=n_simulation_years,  # +1 for year 0 is done inside Outputs.__init__
        n_layers=susi_params.site_parameters.nLyrs,
        fname=simulation_metadata.netcdf_output_filepath,
    )
    out.initialize(strip_constants=computed_constants.strip)
    out.write_paras(
        sfc=susi_params.site_parameters.sfc,
        dominant_sp=computed_constants.stand.dominant.tree_species,
        subdominant_sp=computed_constants.stand.subdominant.tree_species,
        under_sp=computed_constants.stand.under.tree_species,
    )

    # ---- Scenario info -------------------------------------------------------
    hdr_west, hdr20y_west, hdr_east, hdr20y_east = next(zip(
        susi_params.site_parameters.ditch_depth_west,
        susi_params.site_parameters.ditch_depth_20y_west,
        susi_params.site_parameters.ditch_depth_east,
        susi_params.site_parameters.ditch_depth_20y_east,
    ))
    out.write_scen(0, hdr_west, hdr_east)

    # ---- Initial year (yr=0) writes -----------------------------------------
    out.write_stand(
        0, 0,
        initial_state.annual.stand_outputs,
        initial_state.annual.stand,
        initial_state.annual.stand.previous_nut_stat,
    )
    out.write_canopy_layer(
        0, 0, "dominant",
        initial_state.annual.stand.dominant, init_dom_out,
    )
    out.write_canopy_layer(
        0, 0, "subdominant",
        initial_state.annual.stand.subdominant, init_sub_out,
    )
    out.write_canopy_layer(
        0, 0, "under",
        initial_state.annual.stand.under, init_under_out,
    )
    out.write_groundvegetation(
        0, 0, initial_state.annual.gv, init_gv_out,
    )
    for sub in ("Mass", "N", "P", "K"):
        state_map = {"Mass": initial_state.annual.esom_mass,
                     "N": initial_state.annual.esom_N,
                     "P": initial_state.annual.esom_P,
                     "K": initial_state.annual.esom_K}
        out.write_esom(0, 0, sub, state_map[sub], inivals=True)

    # ---- Pre-build stpout dict (full simulation) -----------------------------
    total_days = sum(ann_out.daily.wtd.shape[0] for ann_out in annual_outputs)
    stpout: dict[str, np.ndarray] = {}
    stpout["dwts"] = np.zeros((1, total_days, n))
    stpout["hts"] = np.zeros((1, total_days, n))
    stpout["runoff"] = np.zeros((1, total_days))
    stpout["runoffwest"] = np.zeros((1, total_days))
    stpout["runoffeast"] = np.zeros((1, total_days))
    stpout["surfacerunoff"] = np.zeros((1, total_days, n))
    stpout["deltas"] = np.zeros((1, total_days, n))
    stpout["afps"] = np.zeros((1, total_days, n))

    fill_offset = 0
    for ann_out in annual_outputs:
        nd = ann_out.daily.wtd.shape[0]
        stpout["dwts"][0, fill_offset:fill_offset+nd, :] = ann_out.daily.wtd
        stpout["hts"][0, fill_offset:fill_offset+nd, :] = ann_out.daily.H
        stpout["runoff"][0, fill_offset:fill_offset+nd] = ann_out.daily.total_runoff
        stpout["runoffwest"][0, fill_offset:fill_offset+nd] = ann_out.daily.runoffwest
        stpout["runoffeast"][0, fill_offset:fill_offset+nd] = ann_out.daily.roffeast
        stpout["surfacerunoff"][0, fill_offset:fill_offset+nd, :] = ann_out.daily.surface_runoff
        stpout["deltas"][0, fill_offset:fill_offset+nd, :] = ann_out.daily.delta
        stpout["afps"][0, fill_offset:fill_offset+nd, :] = ann_out.daily.afp
        fill_offset += nd

    # ---- Annual loop (yr=1..N) -----------------------------------------------
    day_offset = 0
    for yr_idx, (ann_out, ann_state, forc) in enumerate(zip(annual_outputs, annual_states, forcings)):
        ndays = ann_out.daily.wtd.shape[0]
        cal_yr = forc.calendar_year

        out.write_stand(
            0, yr_idx + 1, ann_out.stand, ann_state.annual.stand,
            ann_state.annual.stand.previous_nut_stat,
        )
        out.write_canopy_layer(
            0, yr_idx + 1, "dominant",
            ann_state.annual.stand.dominant, ann_out.stand_dom,
        )
        out.write_canopy_layer(
            0, yr_idx + 1, "subdominant",
            ann_state.annual.stand.subdominant, ann_out.stand_sub,
        )
        out.write_canopy_layer(
            0, yr_idx + 1, "under",
            ann_state.annual.stand.under, ann_out.stand_under,
        )
        out.write_groundvegetation(
            0, yr_idx + 1, ann_state.annual.gv, ann_out.gv,
        )
        for sub in ("Mass", "N", "P", "K"):
            state_map = {"Mass": ann_state.annual.esom_mass,
                         "N": ann_state.annual.esom_N,
                         "P": ann_state.annual.esom_P,
                         "K": ann_state.annual.esom_K}
            out_map = {"Mass": ann_out.esom_mass,
                       "N": ann_out.esom_N,
                       "P": ann_out.esom_P,
                       "K": ann_out.esom_K}
            out.write_esom(0, yr_idx + 1, sub, state_map[sub], out_map[sub])

        out.write_cpy(
            0, day_offset, ndays, yr_idx + 1,
            ann_out.daily.interc, ann_out.daily.evap,
            ann_out.daily.et, ann_out.daily.transpi,
            ann_out.daily.efloor, ann_out.daily.swe,
        )
        out.write_temperature(
            0, day_offset, ndays, ann_out.daily.T_soil_hydro,
        )
        out.write_methane(0, yr_idx + 1, ann_out.methane)
        out.write_fertilization(0, yr_idx + 1, ann_out.fertilization)
        out.write_ojanen(0, yr_idx + 1, ann_out.Rhet, ann_out.soil_co2_balance)
        out.write_export(0, yr_idx + 1, ann_out.doc_export)

        # Strip
        sday = datetime.datetime(cal_yr, 1, 1)
        dfwt = pd.DataFrame(
            stpout["dwts"][0, day_offset:day_offset+ndays, :],
            index=pd.date_range(sday, periods=ndays),
        )
        out.write_strip(
            0, day_offset, ndays, cal_yr, yr_idx + 1,
            dfwt, stpout, susi_params.output_parameters,
            ann_out.strip_diag,
        )

        # Nutrient balances
        out.write_nutrient_balance(
            0, yr_idx + 1, "N", ann_out.esom_N,
            susi_params.site_parameters.depoN,
            ann_out.fertilization.nutrient_release["N"],
            ann_out.stand.n_demand + ann_out.stand.Nleafdemand,
            ann_out.gv.nup,
        )
        out.write_nutrient_balance(
            0, yr_idx + 1, "P", ann_out.esom_P,
            susi_params.site_parameters.depoP,
            ann_out.fertilization.nutrient_release["P"],
            ann_out.stand.p_demand + ann_out.stand.Pleafdemand,
            ann_out.gv.pup,
        )
        out.write_nutrient_balance(
            0, yr_idx + 1, "K", ann_out.esom_K,
            susi_params.site_parameters.depoK,
            ann_out.fertilization.nutrient_release["K"],
            ann_out.stand.k_demand + ann_out.stand.Kleafdemand,
            ann_out.gv.kup,
        )

        # Carbon balance
        out.write_carbon_balance(
            0, yr_idx + 1,
            ann_out.stand, ann_out.gv, ann_out.esom_mass,
            ann_out.doc_export, ann_out.methane,
        )

        day_offset += ndays

    out.close()
    print("NetCDF output written to", simulation_metadata.netcdf_output_filepath)


def run(simulation_params: SimulationParams) -> SimulationOutput:
    susi_params = simulation_params.susi_params
    simulation_metadata = simulation_params.metadata

    _create_output_folder(simulation_metadata)

    weather_data = read_FMI_weather(
        ID=0,
        start_date=susi_params.simulation_config.start_date,
        end_date=susi_params.simulation_config.end_date,
        sourcefile=susi_params.weather_parameters.FMI_weather_filepath,
    )

    module_params = _build_params(susi_params)
    computed_constants = _compute_constants(module_params, susi_params, weather_data)
    buffer = strip.make_numerical_buffer(module_params.strip)
    forcings = _build_annual_forcings(susi_params, module_params, weather_data)

    # Initialize all state, capturing extra layer outputs for year-0 write
    initial_state, dom_out_init, sub_out_init, under_out_init = _init_state(
        module_params, computed_constants, susi_params,
    )

    # Compute initial ground vegetation output (runs one timestep with year-0 stand)
    init_gv_state, init_gv_out = gvegetation.run_timestep(
        module_params.gvegetation, computed_constants.gvegetation,
        gvegetation.assemble_inputs(
            ts=forcings[0].temp_sum,
            vol=initial_state.annual.stand_outputs.volume,
            stems=initial_state.annual.stand_outputs.stems,
            ba=initial_state.annual.stand_outputs.basalarea,
            age=susi_params.site_parameters.age["dominant"],
        ),
        initial_state.annual.gv,
    )
    initial_state = replace(initial_state, annual=replace(
        initial_state.annual, gv=init_gv_state,
    ))

    susi_io.print_site_description(susi_params.site_parameters)

    state = initial_state
    annual_outputs: list[AnnualOutputs] = []
    annual_states: list[AllState] = []

    for yr_idx, forcing in enumerate(forcings):
        state, ann_out = _run_annual_step(
            state, forcing, weather_data,
            module_params, computed_constants, buffer,
        )
        annual_outputs.append(ann_out)
        annual_states.append(state)

    result = SimulationOutput(annual=annual_outputs, final_state=state)

    _write_outputs(
        simulation_metadata, susi_params, module_params,
        computed_constants, annual_outputs, annual_states,
        weather_data,
        initial_state=initial_state,
        init_dom_out=dom_out_init,
        init_sub_out=sub_out_init,
        init_under_out=under_out_init,
        init_gv_out=init_gv_out,
        forcings=forcings,
    )
    write_params_and_metadata(simulation_metadata, susi_params)

    return result


def _init_state(
    params: ModuleParams,
    constants: ModuleComputedConstants,
    susi_params: SusiParams,
) -> tuple[AllState, canopylayer.Outputs, canopylayer.Outputs, canopylayer.Outputs]:
    stand_state, stand_out, dom_out, sub_out, under_out = stand.compute_initial_state(
        params.stand, constants.stand,
        susi_params.site_parameters.age,
        susi_params.site_parameters.n,
    )
    canopy_state = canopygrid.compute_initial_state(params.canopygrid)
    canopy_state = canopygrid.update_amax(stand_state.nut_stat, canopy_state)
    moss_state = mosslayer.compute_initial_state(params.mosslayer, constants.mosslayer)
    peat_T_state = temperature.compute_initial_state(constants.temperature)
    strip_state = strip.compute_initial_state(params.strip, constants.strip)
    gv_state = gvegetation.compute_initial_state(params.gvegetation, constants.gvegetation)
    esom_mass = esom.compute_initial_state(params.esom.mass, constants.esom.mass)
    esom_N = esom.compute_initial_state(params.esom.n, constants.esom.n)
    esom_P = esom.compute_initial_state(params.esom.p, constants.esom.p)
    esom_K = esom.compute_initial_state(params.esom.k, constants.esom.k)
    state = AllState(
        daily=DailyState(
            canopy=canopy_state, moss=moss_state,
            strip=strip_state, peat_T=peat_T_state,
        ),
        annual=AnnualState(
            stand=stand_state, stand_outputs=stand_out, gv=gv_state,
            esom_mass=esom_mass, esom_N=esom_N, esom_P=esom_P, esom_K=esom_K,
        ),
    )
    return state, dom_out, sub_out, under_out


def _run_daily_step(
    daily: DailyState,
    forcing: DailyForcing,
    hdom: np.ndarray,
    leafarea: np.ndarray,
    params: ModuleParams,
    constants: ModuleComputedConstants,
    buffer: strip.NumericalBuffer,
) -> tuple[DailyState, DailyOutputs]:
    n = params.strip.n
    # TODO: dwt frozen at initial value (legacy behavior)
    # dwt = daily.strip.H - constants.strip.ele  # ← should be this
    dwt = params.strip.initial_h * np.ones(n)
    rew = rew_drylimit(dwt)

    # 1. Canopy hydrology
    cpy_inputs = canopygrid.assemble_inputs(
        WeatherForcings(T=forcing.T, Prec=forcing.Prec,
                        Rg=forcing.Rg, Par=forcing.Par, VPD=forcing.VPD),
        hc=hdom, LAIconif=leafarea, Rew=rew, beta=daily.moss.Ree,
    )
    cpy_state, cpy_out = canopygrid.run_timestep(
        params.canopygrid, cpy_inputs, daily.canopy,
    )

    # 2. Moss interception (modifies potinf, evap)
    moss_state, moss_interc_out = mosslayer.run_interception(
        constants.mosslayer,
        mosslayer.assemble_interception_inputs(
            potinf=cpy_out.potinf, evap=cpy_out.efloor,
        ),
        daily.moss,
    )

    # 3. Coupling: water available for soil (uses moss-modified potinf)
    moss_efloor = moss_interc_out.evap
    delta = moss_interc_out.potinf - cpy_out.transpi

    # 4. Strip exfil — cap by air volume
    exfil_out = strip.compute_exfil(
        daily.strip, constants.strip,
        forcing.h0ts_west, forcing.h0ts_east, delta,
    )

    # 5. Moss return flow
    moss_state, moss_rf_out = mosslayer.run_returnflow(
        params.mosslayer, constants.mosslayer,
        mosslayer.assemble_returnflow_inputs(
            rflow=exfil_out.exfil, interception_mbe=moss_interc_out.mbe,
        ),
        moss_state,
    )

    # 6. Strip PDE
    strip_state, ts_out = strip.run_timestep(
        params.strip, constants.strip, daily.strip,
        strip.assemble_timestep_inputs(
            h0ts_west=forcing.h0ts_west, h0ts_east=forcing.h0ts_east,
            exfil_out=exfil_out, moss_rf_out=moss_rf_out,
        ),
        buffer=buffer,
    )

    # 7. Peat temperature (uses moss-modified efloor)
    peat_T_state = temperature.run_timestep(
        params.temperature, constants.temperature,
        temperature.assemble_inputs(
            T_air=forcing.T, swe=cpy_out.swe, efloor=moss_efloor,
        ),
        daily.peat_T,
    )[0]

    n_hydro = params.temperature.n_layers_hydro
    new_daily = replace(daily,
        canopy=cpy_state, moss=moss_state,
        strip=strip_state, peat_T=peat_T_state,
    )
    return new_daily, DailyOutputs(
        wtd=strip_state.H - constants.strip.ele,
        afp=ts_out.afp,
        T_soil_hydro=peat_T_state.T_soil[:n_hydro],
        delta=delta,
        total_runoff=np.asarray(ts_out.roff + np.mean(moss_rf_out.surface_runoff)),
        surface_runoff=moss_rf_out.surface_runoff,
        interc=cpy_out.interc,
        evap=cpy_out.evap,
        et=cpy_out.et,
        transpi=cpy_out.transpi,
        efloor=cpy_out.efloor,
        swe=cpy_out.swe,
        H=strip_state.H,
        runoffwest=np.asarray(ts_out.roffwest),
        roffeast=np.asarray(ts_out.roffeast),
    )


def _run_annual_step(
    state: AllState,
    year: AnnualForcing,
    weather_data: pd.DataFrame,
    params: ModuleParams,
    constants: ModuleComputedConstants,
    buffer: strip.NumericalBuffer,
) -> tuple[AllState, AnnualOutputs]:

    ndays = year.valid_days
    n = params.strip.n
    n_hydro = params.temperature.n_layers_hydro

    # ── Pre-daily: update canopy amax from nutrient status ────────
    canopy_state = canopygrid.update_amax(state.annual.stand.nut_stat, state.daily.canopy)
    daily = replace(state.daily, canopy=canopy_state)

    # ── Pre-allocate daily output arrays ──────────────────────────
    wtd_yr = np.zeros((ndays, n))
    afp_yr = np.zeros((ndays, n))
    T_soil_hydro_yr = np.zeros((ndays, n_hydro))
    delta_yr = np.zeros((ndays, n))
    total_runoff_yr = np.zeros(ndays)
    surface_runoff_yr = np.zeros((ndays, n))
    interc_yr = np.zeros((ndays, n))
    evap_yr = np.zeros((ndays, n))
    et_yr = np.zeros((ndays, n))
    transpi_yr = np.zeros((ndays, n))
    efloor_yr = np.zeros((ndays, n))
    swe_yr = np.zeros((ndays, n))
    H_yr = np.zeros((ndays, n))
    runoffwest_yr = np.zeros(ndays)
    roffeast_yr = np.zeros(ndays)

    # ── Inner daily loop (Python for) ─────────────────────────────
    for dd in range(ndays):
        forcing = DailyForcing(
            T=year.daily_T[dd], Prec=year.daily_Prec[dd],
            Rg=year.daily_Rg[dd], Par=year.daily_Par[dd],
            VPD=year.daily_VPD[dd],
            h0ts_west=year.h0ts_west[dd], h0ts_east=year.h0ts_east[dd],
        )
        daily, daily_out = _run_daily_step(
            daily, forcing,
            hdom=state.annual.stand_outputs.hdom,
            leafarea=state.annual.stand_outputs.leafarea,
            params=params, constants=constants, buffer=buffer,
        )
        wtd_yr[dd] = daily_out.wtd
        afp_yr[dd] = daily_out.afp
        T_soil_hydro_yr[dd] = daily_out.T_soil_hydro
        delta_yr[dd] = daily_out.delta
        total_runoff_yr[dd] = daily_out.total_runoff
        surface_runoff_yr[dd] = daily_out.surface_runoff
        interc_yr[dd] = daily_out.interc
        evap_yr[dd] = daily_out.evap
        et_yr[dd] = daily_out.et
        transpi_yr[dd] = daily_out.transpi
        efloor_yr[dd] = daily_out.efloor
        swe_yr[dd] = np.full(n, np.mean(daily_out.swe))
        H_yr[dd] = daily_out.H
        runoffwest_yr[dd] = daily_out.runoffwest
        roffeast_yr[dd] = daily_out.roffeast

    # ── Stack into DailyOutputs ───────────────────────────────────
    stacked_daily = DailyOutputs(
        wtd=wtd_yr, afp=afp_yr, T_soil_hydro=T_soil_hydro_yr,
        delta=delta_yr, total_runoff=total_runoff_yr,
        surface_runoff=surface_runoff_yr, interc=interc_yr,
        evap=evap_yr, et=et_yr, transpi=transpi_yr,
        efloor=efloor_yr, swe=swe_yr,
        H=H_yr, runoffwest=runoffwest_yr, roffeast=roffeast_yr,
    )

    # ── Aggregate daily outputs ──────────────────────────────────
    strip_diag = strip.compute_residence_time(
        params.strip, constants.strip, wtd_yr.mean(axis=0),
    )

    # ── Annual biogeochemistry (DataFrames preserved — deferred) ─
    sday = datetime.datetime(year.calendar_year, 1, 1)
    dfwt = pd.DataFrame(wtd_yr, index=pd.date_range(sday, periods=ndays))
    dfafp = pd.DataFrame(afp_yr, index=pd.date_range(sday, periods=ndays))
    df_peat_temps = pd.DataFrame(
        T_soil_hydro_yr, index=pd.date_range(sday, periods=ndays),
    )

    # Heterotrophic respiration
    _, _co2, Rhet = heterotrophic_respiration_yr(
        df_peat_temps, year.calendar_year, dfwt,
        state.annual.stand_outputs.volume, constants.spara,
    )

    # Soil CO2 balance
    soil_co2_balance = ojanen_2019(
        constants.spara, year.calendar_year, dfwt,
    )

    # Ground vegetation
    gv_state, gv_out = gvegetation.run_timestep(
        params.gvegetation, constants.gvegetation,
        gvegetation.assemble_inputs(
            ts=year.temp_sum,
            vol=state.annual.stand_outputs.volume,
            stems=state.annual.stand_outputs.stems,
            ba=state.annual.stand_outputs.basalarea,
            age=constants.age,
        ),
        state.annual.gv,
    )

    # Stand growth
    weather_yr = weather_data.loc[str(year.calendar_year)]
    stand_inputs = stand.Inputs(
        photopara=constants.photo_parameters,
        forc=weather_yr, wt=dfwt, afp=dfafp,
        n_supply=np.zeros(n), p_supply=np.zeros(n), k_supply=np.zeros(n),
        groundvegetation_outputs=gv_out,
        previous_nut_stat=state.annual.stand.previous_nut_stat,
        calendar_year=year.calendar_year,
    )
    stand_state, stand_out, dom_out, sub_out, under_out = stand.grow_stand(
        state.annual.stand, constants.stand, stand_inputs,
    )

    # Cutting (conditional)
    if year.do_cutting:
        cut_inputs = replace(stand_inputs, cutting_to_ba=year.cutting_to_ba)
        stand_state, stand_out, _cut_out = stand.cut_stand(
            stand_state, constants.stand, stand_out, cut_inputs,
        )

    # Fertilization
    _, fert_out = fertilization.run_timestep(
        params.fertilization, constants.fertilization,
        fertilization.assemble_inputs(params.fertilization, year.calendar_year),
    )
    pH_inc = fert_out.pH_increment

    # ESOM pH update + run_yr for mass, N, P, K
    def _update_pH(esom_state: esom.State, sub: str) -> esom.State:
        return replace(esom_state, pH=esom.update_soil_pH(
            esom_state.pH, getattr(params.esom, sub), pH_inc,
        ))

    state_mass = _update_pH(state.annual.esom_mass, "mass")
    state_N = _update_pH(state.annual.esom_N, "n")
    state_P = _update_pH(state.annual.esom_P, "p")
    state_K = _update_pH(state.annual.esom_K, "k")

    # ── ESOM mass ────────────────────────────────────────────────
    nonwoodylitter = (
        stand_out.nonwoodylitter + stand_out.nonwoody_lresid
        + stand_out.non_woody_litter_mort + gv_out.nonwoodylitter
    ) / 10000.0
    woodylitter = (
        stand_out.woodylitter + stand_out.woody_lresid
        + stand_out.woody_litter_mort + gv_out.woodylitter
    ) / 10000.0
    inputs_mass = esom.assemble_inputs(
        tair_ts=year.daily_T,
        tp_top_ts=df_peat_temps.iloc[:, 2].values,
        tp_middle_ts=df_peat_temps.iloc[:, 8].values,
        tp_bottom_ts=df_peat_temps.iloc[:, 9].values,
        water_tables=dfwt.values,
        nonwoodylitter=nonwoodylitter, woodylitter=woodylitter,
    )
    state_mass, yr_out_mass = esom.run_yr(
        params.esom.mass, constants.esom.mass, inputs_mass, state_mass,
    )
    assert yr_out_mass.daily_cumulative_out is not None
    doc_export = esom.compose_export(
        yr_out_mass.daily_cumulative_out, strip_diag,
        df_peat_temps.iloc[:, 2].values, n,
    )

    # ── ESOM N ───────────────────────────────────────────────────
    n_nonwoodylitter = (
        stand_out.n_nonwoodylitter + stand_out.n_nonwoody_lresid
        + stand_out.n_non_woody_litter_mort + gv_out.n_litter_nw
    ) / 10000.0
    n_woodylitter = (
        stand_out.n_woodylitter + stand_out.n_woody_lresid
        + stand_out.n_woody_litter_mort
        + stand_out.n_woody_litter_mort   # intentional duplicate (legacy)
        + gv_out.n_litter_w
    ) / 10000.0
    inputs_N = esom.assemble_inputs(
        tair_ts=year.daily_T,
        tp_top_ts=df_peat_temps.iloc[:, 2].values,
        tp_middle_ts=df_peat_temps.iloc[:, 8].values,
        tp_bottom_ts=df_peat_temps.iloc[:, 9].values,
        water_tables=dfwt.values,
        nonwoodylitter=n_nonwoodylitter, woodylitter=n_woodylitter,
    )
    state_N, yr_out_N = esom.run_yr(
        params.esom.n, constants.esom.n, inputs_N, state_N,
    )

    # ── ESOM P ───────────────────────────────────────────────────
    p_nonwoodylitter = (
        stand_out.p_nonwoodylitter + stand_out.p_nonwoody_lresid
        + stand_out.p_non_woody_litter_mort + gv_out.p_litter_nw
    ) / 10000.0
    p_woodylitter = (
        stand_out.p_woodylitter + stand_out.p_woody_lresid
        + stand_out.p_woody_litter_mort + gv_out.p_litter_w
    ) / 10000.0
    inputs_P = esom.assemble_inputs(
        tair_ts=year.daily_T,
        tp_top_ts=df_peat_temps.iloc[:, 2].values,
        tp_middle_ts=df_peat_temps.iloc[:, 8].values,
        tp_bottom_ts=df_peat_temps.iloc[:, 9].values,
        water_tables=dfwt.values,
        nonwoodylitter=p_nonwoodylitter, woodylitter=p_woodylitter,
    )
    state_P, yr_out_P = esom.run_yr(
        params.esom.p, constants.esom.p, inputs_P, state_P,
    )

    # ── ESOM K ───────────────────────────────────────────────────
    k_nonwoodylitter = (
        stand_out.k_nonwoodylitter + stand_out.k_nonwoody_lresid
        + stand_out.k_non_woody_litter_mort + gv_out.k_litter_nw
    ) / 10000.0
    k_woodylitter = (
        stand_out.k_woodylitter + stand_out.k_woody_lresid
        + stand_out.k_woody_litter_mort + gv_out.k_litter_w
    ) / 10000.0
    inputs_K = esom.assemble_inputs(
        tair_ts=year.daily_T,
        tp_top_ts=df_peat_temps.iloc[:, 2].values,
        tp_middle_ts=df_peat_temps.iloc[:, 8].values,
        tp_bottom_ts=df_peat_temps.iloc[:, 9].values,
        water_tables=dfwt.values,
        nonwoodylitter=k_nonwoodylitter, woodylitter=k_woodylitter,
    )
    state_K, yr_out_K = esom.run_yr(
        params.esom.k, constants.esom.k, inputs_K, state_K,
    )

    # Nutrient status update
    nut_inputs = replace(stand_inputs,
        n_supply=yr_out_N.out_root_lyr + constants.depoN + fert_out.nutrient_release["N"],
        p_supply=yr_out_P.out_root_lyr + constants.depoP + fert_out.nutrient_release["P"],
        k_supply=yr_out_K.out_root_lyr + constants.depoK + fert_out.nutrient_release["K"],
    )
    stand_state = stand.update_nutrient_status(stand_state, stand_out, nut_inputs)

    # Methane
    _, ch4_out = methane.run_timestep(
        inputs=methane.assemble_inputs(year=year.calendar_year, dfwt=dfwt),
    )

    new_annual = replace(state.annual,
        stand=stand_state, stand_outputs=stand_out,
        gv=gv_state, esom_mass=state_mass, esom_N=state_N,
        esom_P=state_P, esom_K=state_K,
    )
    new_state = replace(state, daily=daily, annual=new_annual)
    return new_state, AnnualOutputs(
        daily=stacked_daily,
        stand=stand_out, stand_dom=dom_out, stand_sub=sub_out,
        stand_under=under_out,
        gv=gv_out,
        esom_mass=yr_out_mass, esom_N=yr_out_N,
        esom_P=yr_out_P, esom_K=yr_out_K,
        methane=ch4_out, fertilization=fert_out,
        Rhet=Rhet, soil_co2_balance=soil_co2_balance,
        doc_export=doc_export, strip_diag=strip_diag,
    )
