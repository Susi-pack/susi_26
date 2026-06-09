# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:38:10 2018

@author: lauren
"""

from supersusi.io.execution_config import SimulationParams

from typing import cast
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


def run(simulation_params: SimulationParams):
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

    # ******** Soil and strip parameterization *************************

    peat_T_state = temperature.compute_initial_state(
        computed_constants=computed_constants.temperature
    )

    ets = np.zeros(
        (n_simulation_days, susi_params.site_parameters.n)
    )  # Evapotranspiration, mm/day

    # ********initialize result arrays***************************
    scen = susi_params.site_parameters.scenario_name  # scenario name for outputs
    rounds = len(
        susi_params.site_parameters.ditch_depth_east
    )  # number of ditch depth scenarios (used in comparison of management)

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

        print("***********************************")
        print(
            "Computing canopy and soil hydrology ",
            n_simulation_days,
            " days",
            "scenario:",
            scen[n_ditch_scen],
        )

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
    # del stand.dominant
    # del stand.subdominant
    # del stand.under
    # del stand, gv_state, cpy, moss, stp, pt
