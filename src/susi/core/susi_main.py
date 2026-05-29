# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:38:10 2018

@author: lauren
"""

import numpy as np
import pandas as pd
import datetime

from susi.io.execution_config import SimulationParams
from susi.io.metadata_model import SimulationMetaData
from susi.io.susi_parameter_model import (
    CanopyStateParamsArray,
    OrganicLayerParamsArray,
    SusiParams,
)
from susi.core.canopygrid import CanopyGrid
from susi.core.mosslayer import MossLayer
from susi.core.strip import StripHydrology, drain_depth_development
from susi.core.gvegetation import Gvegetation
from susi.core.esom import Esom
from susi.core.stand import Stand
from susi.core.fertilization import initialize_fertilization
from susi.core.susi_utils import rew_drylimit
from susi.core.susi_utils import get_temp_sum, heterotrophic_respiration_yr, ojanen_2019
import susi.io.susi_io as susi_io
from susi.io.outputs import Outputs
import susi.io.utils as io_utils
from susi.core.susi_utils import read_FMI_weather

from supersusi.core import methane
from supersusi.core import temperature


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

        peat_T_params = temperature.Params(
            n_layers_hydro=self.parameters.site_parameters.nLyrs,
            dz=self.parameters.site_parameters.dzLyr,
            timestep=self.parameters.site_parameters.peat_temperature.timestep,
            n_subtimesteps=self.parameters.site_parameters.peat_temperature.n_subtimesteps,
            D=self.parameters.site_parameters.peat_temperature.D,
            heat_of_vaporization=self.parameters.site_parameters.peat_temperature.heat_of_vaporization,
        )
        print(
            "******** Susi-peatland simulator v.12 (2026) c Annamari Laurén *********************"
        )
        print("           ")
        print("Initializing stand and site:")

        # Create output folder for the simulation results.
        self.create_output_folder()

        switches = {"Ojanen2010_2019": True}

        n_simulation_days = (
            self.parameters.simulation_config.end_date
            - self.parameters.simulation_config.start_date
        ).days + 1  # simulation time in days
        n_simulation_years = (
            self.parameters.simulation_config.end_date.year
            - self.parameters.simulation_config.start_date.year
            + 1
        )  # simulation time in years

        temperature_sun_days_degree = get_temp_sum(self.weather_forcing)

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

        # The location of the weather file determines the simulation location
        lat, lon = (
            self.weather_forcing["lat"].iloc[0],
            self.weather_forcing["lon"].iloc[0],
        )
        print(
            "      - Weather input:",
            ", start:",
            self.parameters.simulation_config.start_date.year,
            ", end:",
            self.parameters.simulation_config.end_date.year,
        )
        print("      - Latitude:", lat, ", Longitude:", lon)

        stand = Stand(
            n_scenarios=len(self.parameters.site_parameters.ditch_depth_east),
            n_yrs=n_simulation_years,
            canopylayers=self.parameters.site_parameters.canopylayers,
            n_cols=self.parameters.site_parameters.n,
            sfc=self.parameters.site_parameters.sfc,
            agearr=self.parameters.site_parameters.age,
            allometry_params=self.parameters.allometry_parameters,
            photopara=self.parameters.photo_parameters,
        )  # create stand class
        stand.update()

        out.initialize_stand()  # create output variables to netCDF
        out.initialize_canopy_layer("dominant")  # output variables of trees
        out.initialize_canopy_layer("subdominant")
        out.initialize_canopy_layer("under")

        out.write_paras(
            sfc=self.parameters.site_parameters.sfc,
            dominant_sp=stand.dominant.tree_species,
            subdominant_sp=stand.subdominant.tree_species,
            under_sp=stand.under.tree_species,
        )

        # describe site parameters for user
        susi_io.print_site_description(self.parameters.site_parameters)

        groundvegetation = Gvegetation(
            n=self.parameters.site_parameters.n,
            lat=lat,
            lon=lon,
            sfc=self.parameters.site_parameters.sfc,
            species=stand.dominant.species,
        )  # creates ground vegetation class
        groundvegetation.run(
            stand.basalarea,
            stand.stems,
            stand.volume,
            stand.dominant.species,
            temperature_sun_days_degree,
            age=self.parameters.site_parameters.age["dominant"],
        )
        out.initialize_gv()  # output variables to netCDF

        esmass = Esom(
            spara=self.parameters.site_parameters,
            sfc=self.parameters.site_parameters.sfc,
            days=366 * n_simulation_years,
            substance="Mass",
        )  # initializing organic matter decomposition instace for mass
        esN = Esom(
            spara=self.parameters.site_parameters,
            sfc=self.parameters.site_parameters.sfc,
            days=366 * n_simulation_years,
            substance="N",
        )  # initializing organic matter decomposition instace for N
        esP = Esom(
            spara=self.parameters.site_parameters,
            sfc=self.parameters.site_parameters.sfc,
            days=366 * n_simulation_years,
            substance="P",
        )  # initializing organic matter decomposition instace for P
        esK = Esom(
            spara=self.parameters.site_parameters,
            sfc=self.parameters.site_parameters.sfc,
            days=366 * n_simulation_years,
            substance="K",
        )  # initializing organic matter decomposition instace for K

        ferti = initialize_fertilization(
            fertilization_params=self.parameters.site_parameters.fertilization,
            n_cols=self.parameters.site_parameters.n,
            simulation_end_year=self.parameters.simulation_config.end_date.year,
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

        if switches["Ojanen2010_2019"]:
            out.initialize_ojanen()
        # ********* Above ground hydrology initialization ***************
        # cmask = np.ones(self.parameters.site_parameters.n)
        canopy_state_parameters_array = CanopyStateParamsArray(
            canopy_state_parameters=self.parameters.canopy_parameters.state,
            array_length=self.parameters.site_parameters.n,
        )
        cpy = CanopyGrid(
            cpara=self.parameters.canopy_parameters,
            state=canopy_state_parameters_array,
            outputs=False,
        )  # initialize above ground vegetation hydrology model
        cpy.update_amax(stand.nut_stat)
        out.initialize_cpy()

        org_para_array = OrganicLayerParamsArray(
            organic_layer_parameters=self.parameters.organic_layer_parameters,
            array_length=self.parameters.site_parameters.n,
        )
        moss = MossLayer(org_para_array=org_para_array, outputs=True)
        print("Canopy and moss layer hydrology initialized")

        # ******** Soil and strip parameterization *************************
        stp = StripHydrology(
            self.parameters.site_parameters
        )  # initialize soil hydrology model
        out.initialize_strip(stp)  # outputs for soil hydrology

        static_inputs_peat_T = temperature.compute_static_inputs(
            params=peat_T_params,
            T_air_mean=self.weather_forcing["T"].mean(),
        )
        state_peat_T = temperature.compute_initial_state(
            static_inputs=static_inputs_peat_T
        )
        out.initialize_temperature()

        _ = methane.initialize(n_cols=self.parameters.site_parameters.n)
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

        stpout = stp.create_outarrays(
            rounds, n_simulation_days, self.parameters.site_parameters.n
        )  # create output variables for WT, afp, runoff etc.

        peat_temperatures = np.zeros(
            (rounds, n_simulation_days, self.parameters.site_parameters.nLyrs)
        )  # daily peat temperature profiles

        intercs, evaps, ETs, transpis, efloors, swes = cpy.create_outarrays(
            rounds, n_simulation_days, self.parameters.site_parameters.n
        )  # outputs for canopy hydrology model

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

            stand.reset_domain(self.parameters.site_parameters.age)

            out.write_scen(n_ditch_scen, hdr_west, hdr_east)

            out.write_stand(n_ditch_scen, 0, stand)
            out.write_canopy_layer(n_ditch_scen, 0, "dominant", stand.dominant)
            out.write_canopy_layer(n_ditch_scen, 0, "subdominant", stand.subdominant)
            out.write_canopy_layer(n_ditch_scen, 0, "under", stand.under)

            groundvegetation.reset_domain()
            groundvegetation.run(
                stand.basalarea,
                stand.stems,
                stand.volume,
                stand.dominant.species,
                temperature_sun_days_degree,
                age=self.parameters.site_parameters.age["dominant"],
            )
            out.write_groundvegetation(n_ditch_scen, 0, groundvegetation)

            esmass.reset_storages()
            esN.reset_storages()
            esP.reset_storages()
            esK.reset_storages()

            out.write_esom(n_ditch_scen, 0, "Mass", esmass, inivals=True)
            out.write_esom(n_ditch_scen, 0, "N", esN, inivals=True)
            out.write_esom(n_ditch_scen, 0, "P", esP, inivals=True)
            out.write_esom(n_ditch_scen, 0, "K", esK, inivals=True)

            stp.reset_domain(initial_h=self.parameters.site_parameters.initial_h)

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

                # CHECK THIS AND TEST
                cpy.update_amax(stand.nut_stat)

                # **********  Daily loop ************************************************************
                for dd in range(days):  # day loop
                    # -------Canopy hydrology--------------------------
                    reww = rew_drylimit(
                        dwt
                    )  # for each column: moisture limitation from ground water level (Feddes-function)
                    doy = self.weather_forcing.iloc[d, 14]  # day of the year
                    ta = self.weather_forcing.iloc[d, 4]  # air temperature deg C
                    vpd = self.weather_forcing.iloc[d, 13]  # vapor pressure deficit
                    rg = self.weather_forcing.iloc[d, 8]  # solar radiation
                    par = self.weather_forcing.iloc[
                        d, 10
                    ]  # photosynthetically active radiation
                    prec = self.weather_forcing.iloc[d, 7] / 86400.0  # precipitation

                    potinf, trfall, interc, evap, ET, transpi, efloor, MBE, SWE = (
                        cpy.run_timestep(
                            self.parameters.canopy_parameters,
                            doy,
                            self.parameters.canopy_parameters.dt,
                            ta,
                            prec,
                            rg,
                            par,
                            vpd,
                            hc=stand.hdom,
                            LAIconif=stand.leafarea,
                            Rew=reww,
                            beta=moss.Ree,
                        )
                    )  # canopy hydrology computation

                    intercs, evaps, ETs, transpis, efloors, SWEs = cpy.update_outarrays(
                        n_ditch_scen, d, interc, evap, ET, transpi, efloor, SWE
                    )

                    potinf, efloor, MBE2 = moss.interception(
                        potinf, efloor
                    )  # ground vegetation and moss hydrology
                    stpout["deltas"][n_ditch_scen, d, :] = (
                        potinf - transpi
                    )  # water flux thru soil surface
                    ets[d] = efloor + transpi + interc  # evapotranspiration components

                    if d % 365 == 0:
                        print(
                            "  - day #",
                            d,
                            " hdom ",
                            np.round(np.mean(stand.hdom), 2),
                            " m, ",
                            "LAI ",
                            np.round(np.mean(stand.leafarea), 2),
                            " m2 m-2",
                        )

                    # --------Soil hydrology-----------------
                    stp.run_timestep(
                        d,
                        h0ts_west[d],
                        h0ts_east[d],
                        stpout["deltas"][n_ditch_scen, d, :],
                        moss,
                    )  # strip/peat hydrology
                    stpout = stp.update_outarrays(n_ditch_scen, d, stpout)

                    peat_temperature = temperature.step(
                        params=peat_T_params,
                        static_inputs=static_inputs_peat_T,
                        dynamic_inputs=temperature.DynamicInputs(
                            T_air=ta, swe=np.mean(SWE), efloor=np.mean(efloor)
                        ),
                        state=state_peat_T,
                    )
                    peat_temperatures[n_ditch_scen, d, :] = peat_temperature.T_soil

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
                    SWEs[n_ditch_scen, start : start + days, :],
                )

                out.write_temperature(
                    n_ditch_scen,
                    start,
                    days,
                    peat_temperatures[n_ditch_scen, start : start + days, :],
                )

                stp.update_residence_time(dfwt)
                out.write_strip(
                    n_ditch_scen,
                    start,
                    days,
                    calendar_year,
                    simulation_year,
                    dfwt,
                    stpout,
                    self.parameters.output_parameters,
                    stp,
                )

                # **************  Biogeochemistry ***********************************
                if switches["Ojanen2010_2019"]:
                    v = stand.volume
                    _, co2, Rhet = heterotrophic_respiration_yr(
                        df_peat_temperatures,
                        calendar_year,
                        dfwt,
                        v,
                        self.parameters.site_parameters,
                    )  # Rhet is total annual heterotrophic respiration in kg/ha/yr CO2, per computation node
                    soil_co2_balance = ojanen_2019(
                        self.parameters.site_parameters, calendar_year, dfwt
                    )
                    out.write_ojanen(
                        n_ditch_scen, simulation_year, Rhet, soil_co2_balance
                    )

                groundvegetation.run(
                    stand.basalarea,
                    stand.stems,
                    stand.volume,
                    stand.dominant.species,
                    temperature_sun_days_degree,
                    age=self.parameters.site_parameters.age["dominant"],
                )

                stand.assimilate(
                    self.parameters.photo_parameters,
                    self.weather_forcing.loc[str(calendar_year)],
                    dfwt.loc[str(calendar_year)],
                    dfafp.loc[str(calendar_year)],
                )
                stand.update()

                # --------- Locate cuttings here--------------------
                print("calculating year " + str(calendar_year))
                if calendar_year == self.parameters.site_parameters.cutting_yr:
                    print("xxxxxxxxxxxx   VOL before cutting xxxxxxxxxxxxxxxx")
                    print(str(np.round(np.mean(stand.volume), 1)))
                    print(
                        "cutting now "
                        + str(calendar_year)
                        + " from basal area "
                        + str(np.round(np.mean(stand.basalarea), 1))
                        + " to "
                        + str(self.parameters.site_parameters.cutting_to_ba)
                    )

                    stand.dominant.cutting(
                        calendar_year,
                        nut_stat=stand.nut_stat,
                        to_ba=self.parameters.site_parameters.cutting_to_ba,
                    )
                    stand.update_logging()

                #      """ ATTN distinct stem mortaility from the logging -> fate of stems different!!!"""
                # ---------- Organic matter decomposition and nutrient release-----------------

                # ---------------- Fertilization --------------------------------

                fertilization_effect = ferti.compute_effect(year=calendar_year)
                if fertilization_effect.is_active:
                    for es in (esmass, esN, esP, esK):
                        es.update_soil_pH(fertilization_effect.pH_increment)

                out.write_fertilization(
                    n_ditch_scen, simulation_year, fertilization_effect
                )

                """
                TODO:
                    logging resdues to output, to be joined wtih litter in figures
                    harvested volume and biomass to outputs
                    construct balcances at the end of simulation, join to outputs
                """
                nonwoodylitter = (
                    stand.nonwoodylitter
                    + stand.nonwoody_lresid
                    + stand.non_woody_litter_mort
                    + groundvegetation.nonwoodylitter
                ) / 10000.0  # conversion kg/ha/yr -> kg/m2/yr
                woodylitter = (
                    stand.woodylitter
                    + stand.woody_lresid
                    + stand.woody_litter_mort
                    + groundvegetation.woodylitter
                ) / 10000.0
                esmass.run_yr(
                    self.weather_forcing.loc[str(calendar_year)],
                    df_peat_temperatures,
                    dfwt,
                    nonwoodylitter,
                    woodylitter,
                )
                esmass.compose_export(stp, df_peat_temperatures)
                out.write_esom(n_ditch_scen, simulation_year, "Mass", esmass)

                n_nonwoodylitter = (
                    stand.n_nonwoodylitter
                    + stand.n_nonwoody_lresid
                    + stand.n_non_woody_litter_mort
                    + groundvegetation.n_litter_nw
                ) / 10000.0
                n_woodylitter = (
                    stand.n_woodylitter
                    + stand.n_woody_lresid
                    + stand.n_woody_litter_mort
                    + stand.n_woody_litter_mort
                    + groundvegetation.n_litter_w
                ) / 10000.0
                esN.run_yr(
                    self.weather_forcing.loc[str(calendar_year)],
                    df_peat_temperatures,
                    dfwt,
                    n_nonwoodylitter,
                    n_woodylitter,
                )
                out.write_esom(n_ditch_scen, simulation_year, "N", esN)

                p_nonwoodylitter = (
                    stand.p_nonwoodylitter
                    + stand.p_nonwoody_lresid
                    + stand.p_non_woody_litter_mort
                    + groundvegetation.p_litter_nw
                ) / 10000.0
                p_woodylitter = (
                    stand.p_woodylitter
                    + stand.p_woody_lresid
                    + stand.p_woody_litter_mort
                    + groundvegetation.p_litter_w
                ) / 10000.0
                esP.run_yr(
                    self.weather_forcing.loc[str(calendar_year)],
                    df_peat_temperatures,
                    dfwt,
                    p_nonwoodylitter,
                    p_woodylitter,
                )
                out.write_esom(n_ditch_scen, simulation_year, "P", esP)

                k_nonwoodylitter = (
                    stand.k_nonwoodylitter
                    + stand.k_nonwoody_lresid
                    + stand.k_non_woody_litter_mort
                    + groundvegetation.k_litter_nw
                ) / 10000.0
                k_woodylitter = (
                    stand.k_woodylitter
                    + stand.k_woody_lresid
                    + stand.k_woody_litter_mort
                    + groundvegetation.k_litter_w
                ) / 10000.0
                esK.run_yr(
                    self.weather_forcing.loc[str(calendar_year)],
                    df_peat_temperatures,
                    dfwt,
                    k_nonwoodylitter,
                    k_woodylitter,
                )
                out.write_esom(n_ditch_scen, simulation_year, "K", esK)

                stand.update_nutrient_status(
                    groundvegetation,
                    esN.out_root_lyr
                    + self.parameters.site_parameters.depoN
                    + fertilization_effect.nutrient_release["N"],
                    esP.out_root_lyr
                    + self.parameters.site_parameters.depoP
                    + fertilization_effect.nutrient_release["P"],
                    esK.out_root_lyr
                    + self.parameters.site_parameters.depoK
                    + fertilization_effect.nutrient_release["K"],
                )

                # move stand.assimilate here, if first year, take foliage litter from 'table growth (interpolation functions)'
                # stand.assimilate(self.weather_forcing.loc[str(yr)], dfwt.loc[str(yr)], dfafp.loc[str(yr)])
                # stand.update()

                ch4_state = methane.step(
                    dynamic_inputs=methane.DynamicInputs(year=calendar_year, dfwt=dfwt)
                )
                out.write_methane(n_ditch_scen, simulation_year, ch4_state)

                out.write_stand(n_ditch_scen, simulation_year, stand)
                out.write_canopy_layer(
                    n_ditch_scen, simulation_year, "dominant", stand.dominant
                )
                out.write_canopy_layer(
                    n_ditch_scen, simulation_year, "subdominant", stand.subdominant
                )
                out.write_canopy_layer(
                    n_ditch_scen, simulation_year, "under", stand.under
                )
                out.write_groundvegetation(
                    n_ditch_scen, simulation_year, groundvegetation
                )
                out.write_export(n_ditch_scen, simulation_year, esmass)

                out.write_nutrient_balance(
                    n_ditch_scen,
                    simulation_year,
                    "N",
                    esN,
                    self.parameters.site_parameters.depoN,
                    fertilization_effect.nutrient_release["N"],
                    stand.n_demand + stand.n_leaf_demand,
                    groundvegetation.nup,
                )
                out.write_nutrient_balance(
                    n_ditch_scen,
                    simulation_year,
                    "P",
                    esP,
                    self.parameters.site_parameters.depoP,
                    fertilization_effect.nutrient_release["P"],
                    stand.p_demand + stand.p_leaf_demand,
                    groundvegetation.pup,
                )
                out.write_nutrient_balance(
                    n_ditch_scen,
                    simulation_year,
                    "K",
                    esK,
                    self.parameters.site_parameters.depoK,
                    fertilization_effect.nutrient_release["K"],
                    stand.k_demand + stand.k_leaf_demand,
                    groundvegetation.kup,
                )

                out.write_carbon_balance(
                    n_ditch_scen,
                    simulation_year,
                    stand,
                    groundvegetation,
                    esmass,
                    ch4_state,
                )

                stand.reset_logging()  # reset logging in general
                #
                start = start + days  # starting point of the next year daily loop

        out.close()
        # del stand.dominant
        # del stand.subdominant
        # del stand.under
        # del stand, groundvegetation, esmass, esN, esP, esK, ferti, cpy, moss, stp, pt

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
