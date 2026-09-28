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
# susi_2021.py runs into it and figures_2021.ipynb reads back from it.
PROJECT_DIR = repo_root() / "test_projects" / "susi_2021"
RUN_ID = "run_01"

# The Motti files (.xls) are raw data under data/motti/; convert_motti.py turns
# the ones PARAMS_PER_SITE references into the CSV allometry files in
# inputs/allometry/, which is what the runs read. The weather files and field
# measurements are raw data under data/ too.
ALLOMETRY_DIR = allometry_dir_for_project(PROJECT_DIR)
MOTTI_DIR = data_dir_for_project(PROJECT_DIR) / "motti"
WEATHER_DIR = data_dir_for_project(PROJECT_DIR) / "weather"
MEASUREMENTS_DIR = data_dir_for_project(PROJECT_DIR) / "measurements"


@dataclass(frozen=True)
class VaryingSusiParams:
    """The simulation-side record of one site, as the original `dwts_para.py`
    held it. Field measurements the figures compare against live apart, in
    MEASUREMENTS_PER_SITE. Fields marked "not used" are kept so nothing the
    original recorded is lost (#294)."""

    # Not used. Number of groundwater tubes; always equals the number of
    # tubes listed in MEASUREMENTS_PER_SITE.
    ntubes: int
    # Not used. Interpolated WT file name; no such file is in the data.
    file: str
    wfile: str
    # CSV converted from the Motti file of the same stem in MOTTI_DIR (the
    # original called this field `mottifile`; see "Motti file" in CONTEXT.md).
    allometry_file: str
    # Species of the allometry zone (1 = pine). The Motti file records it on
    # its second sheet; convert_motti.py checks the two agree (#206).
    species_id: int
    ddepth: float
    Swidth: float
    vonP: list[int]
    ptype: list[str]
    start_date: datetime.datetime
    end_date: datetime.datetime
    # Not used. "wet" or "dry" plot of the pair.
    status: str
    bulk_dens: float
    # Not used. Whether the site was thinned during the measurement period.
    # The 11 sites in SITE_LABELS are those where it is False (#291).
    thinning: bool
    drain_age: int
    # Measured stand volume [start, end], m3/ha. Not passed to the model
    # (the original didn't either); the figures read the observed volume
    # growth from it.
    vol: list[float]
    sfc: int
    # Not used. Measured dominant height [start, end], m. Not passed to the
    # model, as in the original.
    hdom: list[float]
    # Not used. The original's matplotlib marker for the site.
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
        allometry_file="ansa21_A.csv",
        species_id=1,
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
        allometry_file="ansa26_A.csv",
        species_id=1,
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
        allometry_file="jaakkoin61_A.csv",
        species_id=1,
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
        allometry_file="jaakkoin62_A.csv",
        species_id=1,
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
        allometry_file="koira11_A.csv",
        species_id=1,
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
        allometry_file="koira12_A.csv",
        species_id=1,
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
        allometry_file="koira21_harvennus_A.csv",
        species_id=1,
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
        allometry_file="koira22_harvennus_A.csv",
        species_id=1,
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
        allometry_file="neva11_A.csv",
        species_id=1,
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
        allometry_file="neva14_A.csv",
        species_id=1,
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
        allometry_file="neva21_harvennus_A.csv",
        species_id=1,
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
        allometry_file="neva24_harvennus_A.csv",
        species_id=1,
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
        allometry_file="neva31_A.csv",
        species_id=1,
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
        allometry_file="neva34_A.csv",
        species_id=1,
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
        allometry_file="parkano11_A.csv",
        species_id=1,
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
        allometry_file="parkano12_harvennus_A.csv",
        species_id=1,
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


@dataclass(frozen=True)
class SiteMeasurements:
    """Field measurements of one site, as the original `wt_figures.py` held
    them (its `wt_meas` table and its `names` list). Only the figures read
    them; nothing here feeds the simulation."""

    # Display name in the figures and the residual table.
    name: str
    # Measured WT workbook, in MEASUREMENTS_DIR / "Pohjavesiaineistot".
    file: str
    # Groundwater tube numbers: the columns of `file` that belong to the site.
    tubes: list[int]
    # Not used. Distance of each tube from the ditch, m, in the order of
    # `tubes`. The original wrote them from the ditch spacing (mid-strip
    # tubes at s/2, Nevajärvi's at s/4 and s/2, with s = 40 m at Ansasaari,
    # 37 m at Koirasuo, 30 m at Nevajärvi); these are the evaluated values.
    # None where the original recorded none (the Parkano sites).
    dist: list[float] | None


MEASUREMENTS_PER_SITE = {
    SiteLabel("ansa21"): SiteMeasurements(
        name="Ansasaari21",
        file="muhos_2_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3, 4, 5, 6],
        dist=[5.0, 20.0, 5.0, 5.0, 20.0, 5.0],
    ),
    SiteLabel("ansa26"): SiteMeasurements(
        name="Ansasaari26",
        file="muhos_2_pohjavesi_koottu.xlsx",
        tubes=[28, 29, 30],
        dist=[5.0, 20.0, 5.0],
    ),
    SiteLabel("jaakkoin61"): SiteMeasurements(
        name="Jaakkoinsuo61",
        file="jaakkoinsuo_pohjavesi_koottu.xlsx",
        tubes=[2, 3, 12, 13, 14, 15],
        dist=[5.0, 5.0, 16.0, 16.0, 17.5, 27.0],
    ),
    SiteLabel("jaakkoin62"): SiteMeasurements(
        name="Jaakkoinsuo62",
        file="jaakkoinsuo_pohjavesi_koottu.xlsx",
        tubes=[28, 30, 36, 39, 42, 43, 44, 45],
        dist=[32.0, 26.0, 21.0, 13.0, 15.0, 23.0, 37.5, 46.0],
    ),
    SiteLabel("koira11"): SiteMeasurements(
        name="Koirasuo11",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[13, 14, 15],
        dist=[5.0, 18.5, 5.0],
    ),
    SiteLabel("koira12"): SiteMeasurements(
        name="Koirasuo12",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3],
        dist=[5.0, 18.5, 5.0],
    ),
    SiteLabel("koira21"): SiteMeasurements(
        name="Koirasuo21",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[19, 20, 21],
        dist=[5.0, 18.5, 5.0],
    ),
    SiteLabel("koira22"): SiteMeasurements(
        name="Koirasuo22",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[31, 32, 33],
        dist=[5.0, 18.5, 5.0],
    ),
    SiteLabel("neva11"): SiteMeasurements(
        name="Nevajärvi11",
        file="nevajarvi_1_pohjavesi_koottu.xlsx",
        tubes=[5, 6, 7, 8, 9, 10],
        dist=[5.0, 5.0, 7.5, 15.0, 7.5, 5.0],
    ),
    SiteLabel("neva14"): SiteMeasurements(
        name="Nevajärvi14",
        file="nevajarvi_1_pohjavesi_koottu.xlsx",
        tubes=[36, 41, 42, 43, 44, 45],
        dist=[5.0, 5.0, 7.5, 15.0, 7.5, 5.0],
    ),
    SiteLabel("neva21"): SiteMeasurements(
        name="Nevajärvi21",
        file="nevajarvi_2_pohjavesi_koottu.xlsx",
        tubes=[5, 6, 7, 8, 9, 10, 11],
        # Six distances for seven tubes, as in the original.
        dist=[5.0, 5.0, 7.5, 15.0, 7.5, 5.0],
    ),
    SiteLabel("neva24"): SiteMeasurements(
        name="Nevajärvi24",
        file="nevajarvi_2_pohjavesi_koottu.xlsx",
        tubes=[36, 41, 42, 43, 44, 45],
        dist=[5.0, 5.0, 7.5, 15.0, 7.5, 5.0],
    ),
    SiteLabel("neva31"): SiteMeasurements(
        name="Nevajärvi31",
        file="nevajarvi_3_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dist=[5.0, 7.5, 15.0, 7.5, 5.0, 5.0, 7.5, 15.0, 7.5, 5.0],
    ),
    SiteLabel("neva34"): SiteMeasurements(
        name="Nevajärvi34",
        file="nevajarvi_3_pohjavesi_koottu.xlsx",
        tubes=[41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        dist=[5.0, 7.5, 15.0, 7.5, 5.0, 5.0, 7.5, 15.0, 7.5, 5.0],
    ),
    SiteLabel("parkano11"): SiteMeasurements(
        name="Parkano11",
        file="parkano_1_pohjavesi_koottu.xlsx",
        tubes=[22, 23, 24, 27, 28, 29, 32, 33, 34, 37, 38, 39],
        dist=None,
    ),
    SiteLabel("parkano12"): SiteMeasurements(
        name="Parkano12",
        file="parkano_3_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 9, 10, 11, 12, 19, 20, 21, 22, 29, 30, 31, 32],
        dist=None,
    ),
}


# The sites susi_2021.py runs and figures_2021.ipynb plots: the 11 whose
# `thinning` is False, as in the original (the thinning sites are #291). The
# order is the original figures' order, which sets each site's panel letter
# and colour; the run doesn't depend on it.
SITE_LABELS = [
    SiteLabel("koira11"),
    SiteLabel("koira12"),
    SiteLabel("ansa21"),
    SiteLabel("ansa26"),
    SiteLabel("neva11"),
    SiteLabel("neva14"),
    SiteLabel("neva31"),
    SiteLabel("neva34"),
    SiteLabel("jaakkoin61"),
    SiteLabel("jaakkoin62"),
    SiteLabel("parkano11"),
]


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
                        file_path=ALLOMETRY_DIR / site_params.allometry_file,
                        species_id=site_params.species_id,
                    )
                },
                pointers={
                    CanopyLayerName.dominant: [1] * n,
                    CanopyLayerName.subdominant: None,
                    CanopyLayerName.under: None,
                },
            ),
            initial_canopylayer_age_years={
                CanopyLayerName.dominant: site_params.Aini,
                CanopyLayerName.subdominant: 0.0,
                CanopyLayerName.under: 0.0,
            },
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
            sitename="susirun",
            sfc_specification=1,
            # Measured values, not model inputs: None, as in the original's
            # `wbal_scens` defaults.
            hdom=None,
            vol=None,
            smc="Peatland",
            nLyrs=50,
            dzLyr=0.05,
            ditch_depth_west=[site_params.ddepth],
            ditch_depth_east=[site_params.ddepth],
            ditch_depth_20y_west=[site_params.ddepth],
            ditch_depth_20y_east=[site_params.ddepth],
            scenario_name=[site_label],  # kasvunlisaykset
            # The site's own drainage age, as the original computed h_mor per
            # site (vesitase_call.py). The previous port hardcoded 50 (#152).
            # h_mor is drain_age's only consumer.
            drain_age=site_params.drain_age,
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
