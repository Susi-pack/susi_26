import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WeatherForcings:
    T: float = field(doc="Air temperature [degC]")
    Prec: float = field(doc="Precipitation [mm/day]")
    Rg: float = field(doc="Global solar radiation [W m-2]")
    Par: float = field(doc="Photosynthetically active radiation [W m-2]")
    VPD: float = field(doc="Vapor pressure deficit [kPa]")


def read_FMI_weather(ID, start_date, end_date, sourcefile=None) -> pd.DataFrame:
    """
    reads FMI interpolated daily weather data from file
    IN:
        ID - sve catchment ID. set ID=0 if all data wanted
        start_date - 'yyyy-mm-dd'
        end_date - 'yyyy-mm-dd'
    OUT:
        fmi - pd.dataframe with datetimeindex
            fmi columns:['ID','Kunta','aika','lon','lat','T','Tmax','Tmin','Prec','Rg','h2o','dds','Prec_a','Par','RH','esa','VPD','doy']
            units: T, Tmin, Tmax, dds[degC], VPD, h2o,esa[kPa], Prec, Prec_a[mm], Rg,Par[Wm-2],lon,lat[deg]
    """

    # OmaTunniste;OmaItä;OmaPohjoinen;Kunta;siteid;vuosi;kk;paiva;longitude;latitude;t_mean;t_max;t_min;
    # rainfall;radiation;hpa;lamposumma_v;rainfall_v;lamposumma;lamposumma_cum
    # -site number
    # -date (yyyy mm dd)
    # -latitude (in KKJ coordinates, metres)
    # -longitude (in KKJ coordinates, metres)
    # -T_mean (degrees celcius)
    # -T_max (degrees celcius)
    # -T_min (degrees celcius)
    # -rainfall (mm)
    # -global radiation (per day in kJ/m2)
    # -H2O partial pressure (hPa)

    ID = 0
    print(" + Reading meteorological input file from ")
    print("    -", sourcefile)
    ID = 0
    # import forcing data
    # fmi=pd.read_csv(sourcefile, sep=';', header='infer', usecols=['OmaTunniste','Kunta','aika','longitude','latitude','t_mean','t_max','t_min',\
    #'rainfall','radiation','hpa'],parse_dates='aika')
    # fmi=pd.read_csv(sourcefile, sep=';', header='infer', usecols=['OmaTunniste','Kunta','aika','longitude','latitude','t_mean','t_max','t_min','rainfall','radiation','hpa'])
    # time=pd.to_datetime(fmi['aika'],format='%Y%m%d')

    fmi = pd.read_csv(
        sourcefile,
        sep=";",
        header="infer",
        usecols=[
            "OmaTunniste",
            "Kunta",
            "aika",
            "longitude",
            "latitude",
            "t_mean",
            "t_max",
            "t_min",
            "rainfall",
            "radiation",
            "hpa",
        ],
        encoding="ISO-8859-1",
    )

    # print pd.to_datetime(fmi['aika'][0], format="%Y%m%d")
    # print pd.tseries.tools.to_datetime(fmi['aika'][0], format="%Y%m%d")
    time = pd.to_datetime(fmi["aika"], format="%Y%m%d")

    fmi.index = time
    fmi = fmi.rename(
        columns={
            "OmaTunniste": "ID",
            "longitude": "lon",
            "latitude": "lat",
            "t_mean": "T",
            "t_max": "Tmax",
            "t_min": "Tmin",
            "rainfall": "Prec",
            "radiation": "Rg",
            "hpa": "h2o",
        }
    )

    fmi["h2o"] = 1e-1 * fmi["h2o"]  # hPa-->kPa
    fmi["Rg"] = 1e3 / 86400.0 * fmi["Rg"]  # kJ/m2/d-1 to Wm-2
    fmi["Par"] = 0.5 * fmi["Rg"]

    # saturated vapor pressure
    esa = 0.6112 * np.exp((17.67 * fmi["T"]) / (fmi["T"] + 273.16 - 29.66))  # kPa
    vpd = esa - fmi["h2o"]  # kPa
    vpd[vpd < 0] = 1e-5
    rh = 100.0 * fmi["h2o"] / esa
    rh[rh < 0] = 1e-6
    rh[rh > 100] = 100.0

    fmi["RH"] = rh
    fmi["esa"] = esa
    fmi["vpd"] = vpd
    fmi["doy"] = fmi.index.dayofyear
    # fmi=fmi.drop(['aika'],axis = 1)
    fmi = fmi.drop(columns=["aika"])

    # replace nan's in prec with 0.0
    fmi["Prec"].fillna(value=0.0)
    # del dat, fields, n, k, time

    # get desired period
    fmi = fmi[(fmi.index >= start_date) & (fmi.index <= end_date)]
    if ID > 0:
        fmi = fmi[fmi["ID"] == ID]

    return fmi
