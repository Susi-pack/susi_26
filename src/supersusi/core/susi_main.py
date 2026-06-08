# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:38:10 2018

@author: lauren
"""

from typing import cast
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import datetime

from supersusi.io.execution_config import SimulationParams
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
    canopygrid: canopygrid.Params
    mosslayer: mosslayer.Params
    methane: methane.Params
    temperature: temperature.Params
    fertilization: fertilization.Params
    strip: strip.Params
    gvegetation: gvegetation.Params


class Susi:
    def __init__(self, simulation_parameters: SimulationParams):
        self.metadata: SimulationMetaData = simulation_parameters.metadata
        self.parameters: SusiParams = simulation_parameters.susi_params

        self.weather_forcing = read_FMI_weather(
            ID=0,
            start_date=simulation_parameters.susi_params.simulation_config.start_date,
            end_date=simulation_parameters.susi_params.simulation_config.end_date,
            sourcefile=simulation_parameters.susi_params.weather_parameters.FMI_weather_filepath,
        )

    def run(
        self,
    ):
        module_params = _build_params(self.parameters)

        # Create output folder for the simulation results.
        self.create_output_folder()

        # simulation time in days
        n_simulation_days = (
            self.parameters.simulation_config.end_date
            - self.parameters.simulation_config.start_date
        ).days + 1

        # simulation time in years
        n_simulation_years = (
            self.parameters.simulation_config.end_date.year
            - self.parameters.simulation_config.start_date.year
            + 1
        )
        temperature_sun_days_degree = get_temp_sum(self.weather_forcing)

        # The location of the weather file determines the simulation location
        lat, lon = (
            self.weather_forcing["lat"].iloc[0],
            self.weather_forcing["lon"].iloc[0],
        )

        stand_constants = stand.compute_constants(
            module_params.stand,
            self.parameters.allometry_parameters,
        )
        stand_state, stand_out, *_ = stand.compute_initial_state(
            module_params.stand,
            stand_constants,
            self.parameters.site_parameters.age,
            self.parameters.site_parameters.n,
        )

        out = Outputs(
            n_scenarios=len(self.parameters.site_parameters.ditch_depth_east),
            n_cols=self.parameters.site_parameters.n,
            n_days=n_simulation_days,
            n_years=n_simulation_years,
            n_layers=self.parameters.site_parameters.nLyrs,
            fname=self.metadata.netcdf_output_filepath,
        )

        out.initialize_scens()  # write number scenario attributes: ditch depth,
        out.initialize_paras()  # write tree species, sfc
        out.initialize_stand()  # create output variables to netCDF
        out.initialize_canopy_layer("dominant")  # output variables of trees
        out.initialize_canopy_layer("subdominant")
        out.initialize_canopy_layer("under")
        out.write_paras(
            sfc=self.parameters.site_parameters.sfc,
            dominant_sp=stand_constants.dominant.tree_species,
            subdominant_sp=stand_constants.subdominant.tree_species,
            under_sp=stand_constants.under.tree_species,
        )

        # describe site parameters for user
        susi_io.print_site_description(self.parameters.site_parameters)

        gv_cc = gvegetation.compute_constants(
            params=module_params.gvegetation,
            lon=lon,
            lat=lat,
            dominant_tree_species=stand_constants.dominant.tree_species,
        )
        gv_state = gvegetation.compute_initial_state(module_params.gvegetation, gv_cc)
        _, _ = gvegetation.run_timestep(
            module_params.gvegetation,
            gv_cc,
            gvegetation.assemble_inputs(
                ts=temperature_sun_days_degree,
                vol=stand_out.volume,
                stems=stand_out.stems,
                ba=stand_out.basalarea,
                age=self.parameters.site_parameters.age["dominant"],
            ),
            gv_state,
        )
        out.initialize_gv()  # output variables to netCDF

        spara = self.parameters.site_parameters
        dz = np.ones(spara.nLyrs) * spara.dzLyr
        if spara.vonP:
            vpost = spara.vonP_bottom * np.ones(spara.nLyrs)
            vpost[: len(spara.vonP_top)] = spara.vonP_top
            bd = 0.035 + 0.0159 * vpost
        else:
            bd = spara.bd_bottom * np.ones(spara.nLyrs)
            bd[: len(spara.bd_top)] = spara.bd_top

        peat_key = {"N": spara.peatN, "P": spara.peatP, "K": spara.peatK}

        h_mor_val = cast(
            float, spara.h_mor
        )  # pydantic validator always resolves to float
        params_mass = esom.build_params(
            "Mass",
            spara.n,
            spara.nLyrs,
            dz,
            bd,
            spara.sfc,
            h_mor_val,
            spara.rho_mor,
            spara.enable_peattop,
            spara.enable_peatmiddle,
            spara.enable_peatbottom,
        )
        cc_mass = esom.compute_constants(params_mass)

        params_N = esom.build_params(
            "N",
            spara.n,
            spara.nLyrs,
            dz,
            bd,
            spara.sfc,
            h_mor_val,
            spara.rho_mor,
            spara.enable_peattop,
            spara.enable_peatmiddle,
            spara.enable_peatbottom,
            peat_override=peat_key["N"],
        )
        cc_N = esom.compute_constants(params_N)

        params_P = esom.build_params(
            "P",
            spara.n,
            spara.nLyrs,
            dz,
            bd,
            spara.sfc,
            h_mor_val,
            spara.rho_mor,
            spara.enable_peattop,
            spara.enable_peatmiddle,
            spara.enable_peatbottom,
            peat_override=peat_key["P"],
        )
        cc_P = esom.compute_constants(params_P)

        params_K = esom.build_params(
            "K",
            spara.n,
            spara.nLyrs,
            dz,
            bd,
            spara.sfc,
            h_mor_val,
            spara.rho_mor,
            spara.enable_peattop,
            spara.enable_peatmiddle,
            spara.enable_peatbottom,
            peat_override=peat_key["K"],
        )
        cc_K = esom.compute_constants(params_K)

        fertilization_static_inputs = fertilization.compute_constants(
            module_params.fertilization
        )

        out.initialize_esom("Mass")  # creating output variables for organic matter
        out.initialize_esom("N")
        out.initialize_esom("P")
        out.initialize_esom("K")
        out.initialize_fertilization()  # creating output variables for fertilization
        out.initialize_nutrient_balance("N")  # and for nutrient balance
        out.initialize_nutrient_balance("P")
        out.initialize_nutrient_balance("K")
        out.initialize_carbon_balance()
        out.initialize_ojanen()

        # ********* Above ground hydrology initialization ***************

        canopy_state = canopygrid.compute_initial_state(module_params.canopygrid)
        canopy_state = canopygrid.update_amax(stand_state.nut_stat, canopy_state)
        out.initialize_cpy()

        moss_constants = mosslayer.compute_constants(module_params.mosslayer)
        moss_state = mosslayer.compute_initial_state(
            module_params.mosslayer, moss_constants
        )
        print("Canopy and moss layer hydrology initialized")

        # ******** Soil and strip parameterization *************************
        strip_constants = strip.compute_constants(
            module_params.strip
        )  # initialize soil hydrology model
        strip_buffer = strip.make_numerical_buffer(module_params.strip)
        out.initialize_strip(strip_constants)  # outputs for soil hydrology

        static_inputs_peat_T = temperature.compute_constants(
            params=module_params.temperature,
            T_air_mean=self.weather_forcing["T"].mean(),
        )
        state_peat_T = temperature.compute_initial_state(
            computed_constants=static_inputs_peat_T
        )
        out.initialize_temperature()

        _ = methane.initial_state()
        out.initialize_methane()

        out.initialize_export()  # create output variables for DOC components, east and west ditch
        print("Soil hydrology, temperature and DOC models initialized")

        ets = np.zeros(
            (n_simulation_days, self.parameters.site_parameters.n)
        )  # Evapotranspiration, mm/day

        # ********initialize result arrays***************************
        scen = (
            self.parameters.site_parameters.scenario_name
        )  # scenario name for outputs
        rounds = len(
            self.parameters.site_parameters.ditch_depth_east
        )  # number of ditch depth scenarios (used in comparison of management)

        n = self.parameters.site_parameters.n
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
            (rounds, n_simulation_days, self.parameters.site_parameters.nLyrs)
        )  # daily peat temperature profiles

        intercs = np.zeros(
            (rounds, n_simulation_days, self.parameters.site_parameters.n)
        )
        evaps = np.zeros_like(intercs)
        ETs = np.zeros_like(intercs)
        transpis = np.zeros_like(intercs)
        efloors = np.zeros_like(intercs)
        swes = np.zeros_like(intercs)

        # ***********Scenario loop ********************************************************

        for n_ditch_scen, dr in enumerate(
            zip(
                self.parameters.site_parameters.ditch_depth_west,
                self.parameters.site_parameters.ditch_depth_20y_west,
                self.parameters.site_parameters.ditch_depth_east,
                self.parameters.site_parameters.ditch_depth_20y_east,
            )
        ):
            dwt = self.parameters.site_parameters.initial_h * np.ones(
                self.parameters.site_parameters.n
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
                    stand_constants,
                    self.parameters.site_parameters.age,
                    self.parameters.site_parameters.n,
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
                module_params.gvegetation, gv_cc
            )
            gv_state, gv_outputs = gvegetation.run_timestep(
                module_params.gvegetation,
                gv_cc,
                gvegetation.assemble_inputs(
                    ts=temperature_sun_days_degree,
                    vol=stand_out.volume,
                    stems=stand_out.stems,
                    ba=stand_out.basalarea,
                    age=self.parameters.site_parameters.age["dominant"],
                ),
                gv_state,
            )
            out.write_groundvegetation(n_ditch_scen, 0, gv_state, gv_outputs)

            state_mass = esom.compute_initial_state(params_mass, cc_mass)
            state_N = esom.compute_initial_state(params_N, cc_N)
            state_P = esom.compute_initial_state(params_P, cc_P)
            state_K = esom.compute_initial_state(params_K, cc_K)

            out.write_esom(n_ditch_scen, 0, "Mass", state_mass, inivals=True)
            out.write_esom(n_ditch_scen, 0, "N", state_N, inivals=True)
            out.write_esom(n_ditch_scen, 0, "P", state_P, inivals=True)
            out.write_esom(n_ditch_scen, 0, "K", state_K, inivals=True)

            strip_state = strip.compute_initial_state(
                module_params.strip, strip_constants
            )

            d = 0  # day index
            start = 0  # day counter in annual loop
            # *************** Annual loop *****************************************************************
            # simulation_year starts at 1
            # calendar_year starts at the year of the starting date, e.g., 2004.
            for simulation_year, calendar_year in enumerate(
                range(
                    self.parameters.simulation_config.start_date.year,
                    self.parameters.simulation_config.end_date.year + 1,
                ),
                start=1,
            ):
                days = (
                    datetime.datetime(calendar_year, 12, 31)
                    - datetime.datetime(calendar_year, 1, 1)
                ).days + 1

                canopy_state = canopygrid.update_amax(
                    stand_state.nut_stat, canopy_state
                )

                # **********  Daily loop ************************************************************
                for dd in range(days):  # day loop
                    # -------Canopy hydrology--------------------------
                    reww = rew_drylimit(
                        dwt
                    )  # for each column: moisture limitation from ground water level (Feddes-function)
                    forcings = WeatherForcings(
                        T=self.weather_forcing.iloc[d, 4],
                        Prec=self.weather_forcing.iloc[d, 7],
                        Rg=self.weather_forcing.iloc[d, 8],
                        Par=self.weather_forcing.iloc[d, 10],
                        VPD=self.weather_forcing.iloc[d, 13],
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
                        moss_constants,
                        mosslayer.assemble_interception_inputs(
                            potinf=potinf, evap=efloor
                        ),
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
                        strip_constants,
                        h0ts_west[d],
                        h0ts_east[d],
                        stpout["deltas"][n_ditch_scen, d, :],
                    )  # how much water soil cannot store
                    moss_state, moss_rf_out = mosslayer.run_returnflow(
                        module_params.mosslayer,
                        moss_constants,
                        mosslayer.assemble_returnflow_inputs(
                            rflow=exfil_out.exfil,
                            interception_mbe=moss_interception_outputs.mbe,
                        ),
                        moss_state,
                    )
                    strip_state, ts_out = strip.run_timestep(
                        module_params.strip,
                        strip_constants,
                        strip_state,
                        strip.assemble_timestep_inputs(
                            h0ts_west=h0ts_west[d],
                            h0ts_east=h0ts_east[d],
                            exfil_out=exfil_out,
                            moss_rf_out=moss_rf_out,
                        ),
                        buffer=strip_buffer,
                    )  # strip/peat hydrology
                    stpout["dwts"][n_ditch_scen, d, :] = (
                        strip_state.H - strip_constants.ele
                    )
                    stpout["hts"][n_ditch_scen, d, :] = strip_state.H
                    stpout["afps"][n_ditch_scen, d, :] = ts_out.afp
                    stpout["runoff"][n_ditch_scen, d] = ts_out.roff + np.mean(
                        moss_rf_out.surface_runoff
                    )
                    stpout["runoffwest"][n_ditch_scen, d] = ts_out.roffwest
                    stpout["runoffeast"][n_ditch_scen, d] = ts_out.roffeast
                    stpout["surfacerunoff"][n_ditch_scen, d, :] = (
                        moss_rf_out.surface_runoff
                    )

                    state_peat_T, _ = temperature.run_timestep(
                        params=module_params.temperature,
                        computed_constants=static_inputs_peat_T,
                        inputs=temperature.assemble_inputs(
                            T_air=forcings.T, swe=SWE, efloor=efloor
                        ),
                        state=state_peat_T,
                    )
                    peat_temperatures[n_ditch_scen, d, :] = state_peat_T.T_soil[
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
                    module_params.strip, strip_constants, mean_dwt
                )
                out.write_strip(
                    n_ditch_scen,
                    start,
                    days,
                    calendar_year,
                    simulation_year,
                    dfwt,
                    stpout,
                    self.parameters.output_parameters,
                    strip_diag,
                )

                # **************  Biogeochemistry ***********************************
                v = stand_out.volume
                _, _co2, Rhet = heterotrophic_respiration_yr(
                    df_peat_temperatures,
                    calendar_year,
                    dfwt,
                    v,
                    self.parameters.site_parameters,
                )  # Rhet is total annual heterotrophic respiration in kg/ha/yr CO2, per computation node
                soil_co2_balance = ojanen_2019(
                    self.parameters.site_parameters, calendar_year, dfwt
                )
                out.write_ojanen(n_ditch_scen, simulation_year, Rhet, soil_co2_balance)

                gv_state, gv_outputs = gvegetation.run_timestep(
                    module_params.gvegetation,
                    gv_cc,
                    gvegetation.assemble_inputs(
                        ts=temperature_sun_days_degree,
                        vol=stand_out.volume,
                        stems=stand_out.stems,
                        ba=stand_out.basalarea,
                        age=self.parameters.site_parameters.age["dominant"],
                    ),
                    gv_state,
                )

                _stand_inputs = stand.Inputs(
                    photopara=self.parameters.photo_parameters,
                    forc=self.weather_forcing.loc[str(calendar_year)],
                    wt=dfwt.loc[str(calendar_year)],
                    afp=dfafp.loc[str(calendar_year)],
                    n_supply=np.zeros(self.parameters.site_parameters.n),
                    p_supply=np.zeros(self.parameters.site_parameters.n),
                    k_supply=np.zeros(self.parameters.site_parameters.n),
                    groundvegetation_outputs=gv_outputs,
                    previous_nut_stat=stand_state.previous_nut_stat,
                    calendar_year=calendar_year,
                )
                stand_state, stand_out, dom_out, sub_out, under_out = stand.grow_stand(
                    stand_state, stand_constants, _stand_inputs
                )

                # --------- Locate cuttings here--------------------
                print("calculating year " + str(calendar_year))
                if calendar_year == self.parameters.site_parameters.cutting_yr:
                    print("xxxxxxxxxxxx   VOL before cutting xxxxxxxxxxxxxxxx")
                    print(str(np.round(np.mean(stand_out.volume), 1)))
                    print(
                        "cutting now "
                        + str(calendar_year)
                        + " from basal area "
                        + str(np.round(np.mean(stand_out.basalarea), 1))
                        + " to "
                        + str(self.parameters.site_parameters.cutting_to_ba)
                    )

                    _cut_inputs = replace(
                        _stand_inputs,
                        cutting_to_ba=self.parameters.site_parameters.cutting_to_ba,
                    )
                    stand_state, stand_out, _cutting_out = stand.cut_stand(
                        stand_state, stand_constants, stand_out, _cut_inputs
                    )
                else:
                    _cutting_out = canopylayer.CuttingOutputs(
                        harvested_volume=np.zeros(self.parameters.site_parameters.n),
                        harvested_log_volume=np.zeros(
                            self.parameters.site_parameters.n
                        ),
                        harvested_pulp_volume=np.zeros(
                            self.parameters.site_parameters.n
                        ),
                        harvested_biomass=np.zeros(self.parameters.site_parameters.n),
                        harvested_stems=np.zeros(self.parameters.site_parameters.n),
                        nonwoody_lresid=np.zeros(self.parameters.site_parameters.n),
                        n_nonwoody_lresid=np.zeros(self.parameters.site_parameters.n),
                        p_nonwoody_lresid=np.zeros(self.parameters.site_parameters.n),
                        k_nonwoody_lresid=np.zeros(self.parameters.site_parameters.n),
                        woody_lresid=np.zeros(self.parameters.site_parameters.n),
                        n_woody_lresid=np.zeros(self.parameters.site_parameters.n),
                        p_woody_lresid=np.zeros(self.parameters.site_parameters.n),
                        k_woody_lresid=np.zeros(self.parameters.site_parameters.n),
                    )

                #      """ ATTN distinct stem mortaility from the logging -> fate of stems different!!!"""
                # ---------- Organic matter decomposition and nutrient release-----------------

                # ---------------- Fertilization --------------------------------

                _, fertilization_outputs = fertilization.run_timestep(
                    params=module_params.fertilization,
                    computed_constants=fertilization_static_inputs,
                    inputs=fertilization.assemble_inputs(
                        params=module_params.fertilization, calendar_year=calendar_year
                    ),
                )
                pH_inc = fertilization_outputs.pH_increment
                state_mass = esom.State(
                    M=state_mass.M,
                    i=state_mass.i,
                    previous_mass=state_mass.previous_mass,
                    pH=esom.update_soil_pH(state_mass.pH, params_mass, pH_inc),
                )
                state_N = esom.State(
                    M=state_N.M,
                    i=state_N.i,
                    previous_mass=state_N.previous_mass,
                    pH=esom.update_soil_pH(state_N.pH, params_N, pH_inc),
                )
                state_P = esom.State(
                    M=state_P.M,
                    i=state_P.i,
                    previous_mass=state_P.previous_mass,
                    pH=esom.update_soil_pH(state_P.pH, params_P, pH_inc),
                )
                state_K = esom.State(
                    M=state_K.M,
                    i=state_K.i,
                    previous_mass=state_K.previous_mass,
                    pH=esom.update_soil_pH(state_K.pH, params_K, pH_inc),
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
                weather_yr = self.weather_forcing.loc[str(calendar_year)]
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
                    params_mass, cc_mass, inputs_mass, state_mass
                )
                assert yr_out_mass.daily_cumulative_out is not None
                doc_export = esom.compose_export(
                    yr_out_mass.daily_cumulative_out,
                    strip_diag,
                    df_peat_temperatures.iloc[:, 2].values,
                    spara.n,
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
                state_N, yr_out_N = esom.run_yr(params_N, cc_N, inputs_N, state_N)
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
                state_P, yr_out_P = esom.run_yr(params_P, cc_P, inputs_P, state_P)
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
                state_K, yr_out_K = esom.run_yr(params_K, cc_K, inputs_K, state_K)
                out.write_esom(n_ditch_scen, simulation_year, "K", state_K, yr_out_K)

                _nut_inputs = replace(
                    _stand_inputs,
                    n_supply=yr_out_N.out_root_lyr
                    + self.parameters.site_parameters.depoN
                    + fertilization_outputs.nutrient_release["N"],
                    p_supply=yr_out_P.out_root_lyr
                    + self.parameters.site_parameters.depoP
                    + fertilization_outputs.nutrient_release["P"],
                    k_supply=yr_out_K.out_root_lyr
                    + self.parameters.site_parameters.depoK
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
                    self.parameters.site_parameters.depoN,
                    fertilization_outputs.nutrient_release["N"],
                    stand_out.n_demand + stand_out.Nleafdemand,
                    gv_outputs.nup,
                )
                out.write_nutrient_balance(
                    n_ditch_scen,
                    simulation_year,
                    "P",
                    yr_out_P,
                    self.parameters.site_parameters.depoP,
                    fertilization_outputs.nutrient_release["P"],
                    stand_out.p_demand + stand_out.Pleafdemand,
                    gv_outputs.pup,
                )
                out.write_nutrient_balance(
                    n_ditch_scen,
                    simulation_year,
                    "K",
                    yr_out_K,
                    self.parameters.site_parameters.depoK,
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

    def create_output_folder(self) -> None:
        assert self.metadata.experiment_folder_path is not None
        assert not self.metadata.experiment_folder_path.is_dir()
        assert not self.metadata.experiment_folder_path.exists()
        io_utils.create_folder(path=self.metadata.experiment_folder_path)
        return None

    def write_params_and_metadata(self) -> None:
        self.metadata.record_end_timestamp()
        self.metadata.dump_json_to_file()
        self.parameters.dump_json_to_file(
            filepath=self.metadata.parameter_output_filepath
        )
        return None


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
