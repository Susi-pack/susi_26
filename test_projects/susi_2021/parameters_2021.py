# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:03:26 2019

@author: alauren
"""

from dataclasses import dataclass
from typing import NewType
import datetime

from susi.io.project_layout import allometry_dir_for_project, data_dir_for_project
from susi.io.utils import repo_root
from susi.io.susi_parameter_model import (
    PeatTypes,
    SiteParams,
    StandParams,
    WeatherParams,
    SimulationConfig,
    SusiParams,
    AllometryFileAndSpecies,
    CanopyLayerAllometry,
    CanopyLayerName,
    CanopyParams,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
)


# A testing project lives in the checkout, so it names its folder from the
# repo root rather than through `project_dir()` (ADR 0006). The one definition:
# susi_2021.py runs into it and figures_2021.py reads back from it.
PROJECT_DIR = repo_root() / "test_projects" / "susi_2021"
RUN_ID = "run_01"

# Motti growth tables are allometry, so they live in inputs/allometry/; the
# weather files and field measurements are raw data under data/.
ALLOMETRY_DIR = allometry_dir_for_project(PROJECT_DIR)
WEATHER_DIR = data_dir_for_project(PROJECT_DIR) / "weather"
MEASUREMENTS_DIR = data_dir_for_project(PROJECT_DIR) / "measurements"


@dataclass(frozen=True)
class VaryingSusiParams:
    ntubes: int
    file: str
    wfile: str
    mottifile: str
    ddepth: float
    Swidth: float
    vonP: list[int]
    ptype: list[str]
    start_date: datetime.datetime
    end_date: datetime.datetime
    status: str
    bulk_dens: float
    thinning: bool
    drain_age: int
    vol: list[float]
    sfc: int
    hdom: list[float]
    mark: str
    Aini: int
    depoN: float
    depoP: float
    depoK: float


SiteLabel = NewType("SiteLabel", str)


def get_stand_label_from_site_label(site_label: SiteLabel) -> str:
    return str(site_label)[:-2]


def get_scenario_label_from_site_label(site_label: SiteLabel) -> str:
    return str(site_label)[-2:]


PARAMS_PER_SITE = {
    SiteLabel("ansa21"): VaryingSusiParams(
        ntubes=6,
        file="DWTansa21Interp.csv",
        wfile="muhos_weather.csv",  # drained 1967 -1982 -> 1968
        mottifile="ansa21_A.xls",
        ddepth=-0.45,
        Swidth=40.0,
        vonP=[2, 3, 4, 4, 6, 6],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2008, 1, 1),
        end_date=datetime.datetime(2014, 12, 31),
        status="wet",
        bulk_dens=128.7,
        thinning=False,
        drain_age=40,
        vol=[140.0, 187.07],
        sfc=3,
        hdom=[12.8, 14.6],
        mark="bo",
        Aini=40,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("ansa26"): VaryingSusiParams(
        ntubes=3,
        file="DWTansa26Interp.csv",
        wfile="muhos_weather.csv",
        mottifile="ansa26_A.xls",
        ddepth=-0.45,
        Swidth=40.0,
        vonP=[2, 3, 4, 4, 6, 6],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2008, 1, 1),
        end_date=datetime.datetime(2014, 12, 31),
        status="dry",
        bulk_dens=128.7,
        thinning=False,
        drain_age=40,
        vol=[109.9, 140.86],
        sfc=3,
        hdom=[12.6, 15.4],
        mark="bo",
        Aini=40,
        depoN=4.0,
        depoP=0.1,
        depoK=0.55,
    ),
    SiteLabel("jaakkoin61"): VaryingSusiParams(
        ntubes=6,
        file="DWTjaakkoin61Interp.csv",
        wfile="jaakkoinsuo_weather.csv",  # drained 1908
        mottifile="jaakkoin61_A.xls",
        ddepth=-0.85,
        Swidth=40.0,
        vonP=[4, 8, 7, 7, 7, 7],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2013, 12, 31),
        status="wet",
        bulk_dens=134.0,
        thinning=False,
        drain_age=110,
        vol=[144.3, 172.6],
        sfc=5,
        hdom=[17.6, 18.5],
        mark="b*",
        Aini=86,
        depoN=4.0,
        depoP=0.1,
        depoK=0.95,
    ),
    SiteLabel("jaakkoin62"): VaryingSusiParams(
        ntubes=8,
        file="DWTjaakkoin62Interp.csv",
        wfile="jaakkoinsuo_weather.csv",
        mottifile="jaakkoin62_A.xls",
        ddepth=-0.85,
        Swidth=40.0,
        vonP=[4, 8, 7, 7, 7, 7],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2013, 12, 31),
        status="dry",
        bulk_dens=134.0,
        thinning=False,
        drain_age=110,
        vol=[149.3, 186.6],
        sfc=5,
        hdom=[18.1, 19.5],
        mark="b*",
        Aini=86,
        depoN=4.0,
        depoP=0.1,
        depoK=0.95,
    ),
    SiteLabel("koira11"): VaryingSusiParams(
        ntubes=3,
        file="DWTkoira11Interp.csv",
        wfile="koirasuo_weather.csv",
        mottifile="koira11_A.xls",
        ddepth=-0.85,
        Swidth=37.0,
        vonP=[4, 4, 5, 6, 7, 7],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2008, 1, 1),
        end_date=datetime.datetime(2014, 12, 31),
        status="wet",
        bulk_dens=108.6,
        thinning=False,
        drain_age=55,  # drained 1950
        vol=[90.9, 122.0],
        sfc=2,
        hdom=[12.9, 13.9],
        mark="bs",
        Aini=77,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("koira12"): VaryingSusiParams(
        ntubes=3,
        file="DWTkoira12Interp.csv",
        wfile="koirasuo_weather.csv",
        mottifile="koira12_A.xls",
        ddepth=-0.85,
        Swidth=37.0,
        vonP=[4, 4, 5, 6, 6, 6],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2008, 1, 1),
        end_date=datetime.datetime(2014, 12, 31),
        status="dry",
        bulk_dens=108.6,
        thinning=False,
        drain_age=55,
        vol=[122.9, 164.0],
        sfc=2,
        hdom=[13.2, 16.4],
        mark="bs",
        Aini=77,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("koira21"): VaryingSusiParams(
        ntubes=3,
        file="DWTkoira21Interp.csv",
        wfile="koirasuo_weather.csv",
        mottifile="koira21_harvennus_A.xls",
        ddepth=-0.85,
        Swidth=37.0,
        vonP=[4, 4, 5, 6, 6, 6],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2008, 1, 1),
        end_date=datetime.datetime(2009, 12, 31),
        status="wet",
        bulk_dens=108.6,
        thinning=True,
        drain_age=55,
        vol=[97.4, 91.9 + 30.6],
        sfc=2,
        hdom=[12.7, 14.5],
        mark="rs",
        Aini=77,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("koira22"): VaryingSusiParams(
        ntubes=3,
        file="DWTkoira22Interp.csv",
        wfile="koirasuo_weather.csv",
        mottifile="koira22_harvennus_A.xls",
        ddepth=-0.85,
        Swidth=37.0,
        vonP=[4, 4, 5, 6, 6, 6],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2008, 1, 1),
        end_date=datetime.datetime(2009, 12, 31),
        status="dry",
        bulk_dens=108.6,
        thinning=True,
        drain_age=55,
        vol=[92.2, 87.13 + 33.92],
        sfc=2,
        hdom=[11.1, 14.3],
        mark="rs",
        Aini=77,
        depoN=4.0,
        depoP=0.1,
        depoK=0.95,
    ),
    SiteLabel("neva11"): VaryingSusiParams(
        ntubes=6,
        file="DWTneva11Interp.csv",
        wfile="nevajarvi_weather.csv",
        mottifile="neva11_A.xls",
        ddepth=-1.03,
        Swidth=30.0,
        vonP=[5, 5, 5, 4, 4, 4],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2014, 12, 31),
        status="dry",
        bulk_dens=113.25,
        thinning=False,
        drain_age=55,  # drainage 2. times 1950
        vol=[163.4, 227.5],
        sfc=3,
        hdom=[16.1, 17.8],
        mark="b+",
        Aini=80,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("neva14"): VaryingSusiParams(
        ntubes=6,
        file="DWTneva14Interp.csv",
        wfile="nevajarvi_weather.csv",
        mottifile="neva14_A.xls",
        ddepth=-1.03,
        Swidth=30.0,
        vonP=[5, 5, 5, 4, 4, 4],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2014, 12, 31),
        status="wet",
        bulk_dens=113.25,
        thinning=False,
        drain_age=55,
        vol=[142.9, 191.4],
        sfc=3,
        hdom=[16.5, 18.4],
        mark="b+",
        Aini=80,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("neva21"): VaryingSusiParams(
        ntubes=7,
        file="DWTneva21Interp.csv",
        wfile="nevajarvi_weather.csv",
        mottifile="neva21_harvennus_A.xls",
        ddepth=-1.07,
        Swidth=30.0,
        vonP=[4, 4, 4, 4, 4, 4],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2008, 12, 31),
        status="dry",
        bulk_dens=97.8,
        thinning=True,
        drain_age=55,
        vol=[100.32, 109.89 + 32.5],
        sfc=3,
        hdom=[15.7, 16.7],
        mark="r+",
        Aini=80,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("neva24"): VaryingSusiParams(
        ntubes=6,
        file="DWTneva24Interp.csv",
        wfile="nevajarvi_weather.csv",
        mottifile="neva24_harvennus_A.xls",
        ddepth=-1.07,
        Swidth=30.0,
        vonP=[4, 4, 4, 4, 4, 4],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2008, 12, 31),
        status="wet",
        bulk_dens=97.8,
        thinning=True,
        drain_age=55,
        vol=[143.6, 124.7 + 53.66],
        sfc=3,
        hdom=[14.6, 17.1],
        mark="r+",
        Aini=80,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("neva31"): VaryingSusiParams(
        ntubes=10,
        file="DWTneva31Interp.csv",
        wfile="nevajarvi_weather.csv",
        mottifile="neva31_A.xls",
        ddepth=-1.08,
        Swidth=30.0,
        vonP=[5, 5, 4, 5, 5, 5],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2014, 12, 31),
        status="dry",
        bulk_dens=122.5,
        thinning=False,
        drain_age=55,
        vol=[127.2, 161.5],
        sfc=3,
        hdom=[16.5, 17.0],
        mark="b+",
        Aini=80,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("neva34"): VaryingSusiParams(
        ntubes=10,
        file="DWTneva34Interp.csv",
        wfile="nevajarvi_weather.csv",
        mottifile="neva34_A.xls",
        ddepth=-1.08,
        Swidth=30.0,
        vonP=[5, 5, 4, 5, 5, 5],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2014, 12, 31),
        status="wet",
        bulk_dens=122.5,
        thinning=False,
        drain_age=55,
        vol=[100.4, 147.3],
        sfc=3,
        hdom=[15.3, 17.1],
        mark="b+",
        Aini=80,
        depoN=4.0,
        depoP=0.1,
        depoK=0.65,
    ),
    SiteLabel("parkano11"): VaryingSusiParams(
        ntubes=12,
        file="DWTparkano11Interp.csv",
        wfile="parkano_weather.csv",
        mottifile="parkano11_A.xls",
        ddepth=-0.86,
        Swidth=65.0,
        vonP=[3, 3, 6, 6, 6, 6],
        ptype=["S", "S", "S", "S", "S", "S"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2011, 12, 31),
        status="dry",
        bulk_dens=99.35,
        thinning=False,
        drain_age=80,  # drained first 1930s, Häädetjärvi 1950s
        vol=[167.7, 192.4],
        sfc=3,
        hdom=[17.3, 18.7],
        mark="bD",
        Aini=131,
        depoN=4.0,
        depoP=0.1,
        depoK=0.95,
    ),
    SiteLabel("parkano12"): VaryingSusiParams(
        ntubes=14,
        file="DWTparkano12Interp.csv",
        wfile="parkano_weather.csv",
        mottifile="parkano12_harvennus_A.xls",
        ddepth=-0.89,
        Swidth=65.0,
        vonP=[3, 5, 6, 6, 6, 6],
        ptype=["A", "A", "A", "A", "A", "A"],
        start_date=datetime.datetime(2007, 1, 1),
        end_date=datetime.datetime(2009, 12, 31),
        status="wet",
        bulk_dens=99.35,
        thinning=True,
        drain_age=80,
        vol=[263.0, 173.0 + 112.9],
        sfc=5,
        hdom=[20.0, 20.7],
        mark="rD",
        Aini=131,
        depoN=4.0,
        depoP=0.1,
        depoK=0.95,
    ),
}


def _rho_mor_from_sfc(sfc: int) -> float:
    if sfc < 3:
        rho_mor = 110.0
    elif sfc == 3:
        rho_mor = 100.0
    elif sfc == 4:
        rho_mor = 85.0
    elif sfc == 5:
        rho_mor = 80.0
    else:
        rho_mor = 60.0
    return rho_mor


def assign_susi_params_to_site(site_label: SiteLabel) -> SusiParams:
    site_params = PARAMS_PER_SITE[site_label]

    L = site_params.Swidth
    n = int(site_params.Swidth / 2)

    return SusiParams(
        weather_parameters=WeatherParams(
            FMI_weather_filepath=WEATHER_DIR / site_params.wfile,
        ),
        simulation_config=SimulationConfig(
            start_date=site_params.start_date,
            end_date=site_params.end_date,
        ),
        stand_params=StandParams(
            site_fertility_class=site_params.sfc,
            canopy_layer_allometry=CanopyLayerAllometry(
                allometry_file_registry={
                    1: AllometryFileAndSpecies(
                        file_path=ALLOMETRY_DIR / site_params.mottifile,
                        species_id=1,
                    )
                },
                pointers={
                    CanopyLayerName.dominant: [1] * n,
                    CanopyLayerName.subdominant: None,
                    CanopyLayerName.under: None,
                },
            ),
        ),
        canopy_parameters=CanopyParams(),
        organic_layer_parameters=OrganicLayerParams(),
        output_parameters=OutputParams(),
        photo_parameters=get_photo_parameters_by_location(
            location=LocationsForPhotoParams("All_data")
        ),
        site_parameters=SiteParams(
            L=L,
            n=n,
            initial_canopylayer_age_years={
                CanopyLayerName.dominant: site_params.Aini,
                CanopyLayerName.subdominant: 0.0,
                CanopyLayerName.under: 0.0,
            },
            sitename="susirun",
            sfc_specification=1,
            hdom=None,
            vol=site_params.vol,
            smc="Peatland",
            nLyrs=50,
            dzLyr=0.05,
            ditch_depth_west=[site_params.ddepth],
            ditch_depth_east=[site_params.ddepth],
            ditch_depth_20y_west=[site_params.ddepth],
            ditch_depth_20y_east=[site_params.ddepth],
            scenario_name=[site_label],  # kasvunlisaykset
            drain_age=50.0,
            initial_h=-0.2,
            slope=0.0,
            peat_type=[
                PeatTypes.generic if pt == "A" else PeatTypes.sphagnum
                for pt in site_params.ptype
            ],
            peat_type_bottom=[PeatTypes.generic],
            anisotropy=10.0,
            vonP=True,
            vonP_top=site_params.vonP,
            vonP_bottom=9,
            bd_top=site_params.bulk_dens / 1000.0,
            bd_bottom=0.16,
            peatN=None,
            peatP=None,
            peatK=None,
            enable_peattop=True,
            enable_peatmiddle=True,
            enable_peatbottom=True,
            rho_mor=_rho_mor_from_sfc(site_params.sfc),
            h_mor=h_mor_from_drainage_and_mass_mor_Pitkanen,
            cutting_management=None,
            depoN=site_params.depoN,
            depoP=site_params.depoP,
            depoK=site_params.depoK * 1.1,
            fertilization=None,
            peat_temperature=PeatTemperatureParams(),
        ),
    )
