# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:04:05 2020

@author: alauren, modified by iurzainki
"""

from datetime import datetime

from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from netCDF4 import Dataset

import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from susi.io.project_layout import run_dir
from parameters_2021 import (
    MEASUREMENTS_DIR,
    PROJECT_DIR,
    RUN_ID,
    SiteLabel,
    get_scenario_label_from_site_label,
    get_stand_label_from_site_label,
    assign_susi_params_to_site,
)


sns.set()

# Constants
SUBPANEL_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 11.0
SHADE_COLOR1 = "yellow"
SHADE_COLOR2 = "grey"
SHADE_ALPHA = 0.3
REGRESSION_LINESTYLE = "k--"
REGRESSION_LINEWIDTH = 2
ERRORBAR_CAPSIZE = 4
SCATTER_MARKERSIZE = 10
SITE_FERTILITY_CLASSES = (2, 2, 3, 3, 3, 3, 3, 3, 5, 5, 5)

# %%


@dataclass(frozen=True)
class WTMeasurementInfo:
    name: str
    file: str  # WT measurements file
    tubes: list[int]  # Number of groundwater tubes in the dataset
    volume_ini: float
    volume_end: float
    hdom_ini: float
    hdom_end: float


# %%


def get_netcdf_path_from_site_label(
    site_label: SiteLabel, project_folderpath: Path
) -> Path:
    stand_label = get_stand_label_from_site_label(site_label)
    scenario_label = get_scenario_label_from_site_label(site_label)

    return project_folderpath / stand_label / scenario_label / "susi.nc"


@contextmanager
def open_susi_netcdf(site_label: SiteLabel):
    ncf = Dataset(
        get_netcdf_path_from_site_label(
            site_label=SiteLabel(site_label), project_folderpath=PROJECT_FOLDER
        ),
        mode="r",
    )
    try:
        yield ncf
    finally:
        ncf.close()


def load_simulation_wt(
    site_label: SiteLabel,
) -> tuple[pd.DataFrame, datetime, datetime]:
    site_params = assign_susi_params_to_site(site_label)
    start_date = site_params.simulation_config.start_date
    end_date = site_params.simulation_config.end_date
    with open_susi_netcdf(site_label) as ncf:
        dwt = ncf["strip"]["dwt"][0, :, 1:-1]
        days, cols = np.shape(dwt)
        dfsim = pd.DataFrame(
            dwt, columns=range(cols), index=pd.date_range(start_date, periods=days)
        )
    return dfsim, start_date, end_date


def load_stand_growth(
    site_label: SiteLabel,
) -> tuple[float, float, float, float]:
    with open_susi_netcdf(site_label) as ncf:
        growth = np.array(ncf["stand"]["volumegrowth"][0, :, 1:-1])
        print(site_label, growth)
        _, ncols = np.shape(growth)
        dfgrowth = pd.DataFrame(growth, columns=range(ncols))

        bm_growth = np.array(
            ncf["stand"]["dominant"]["NPP"][0, :, 1:-1]
            * ncf["stand"]["stems"][0, :, 1:-1]
        )
        dfbm = pd.DataFrame(bm_growth, columns=range(ncols))

    gro = np.mean(dfgrowth.mean(axis=0).values)
    grosd = np.std(dfgrowth.mean(axis=0).values)
    bm_gr = np.mean(dfbm.mean(axis=0).values)
    bm_gr_sd = np.std(dfbm.mean(axis=0).values)
    return gro, grosd, bm_gr, bm_gr_sd


def load_co2_data(
    site_label: SiteLabel,
) -> tuple[
    np.ndarray, int, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray
]:
    site_params = assign_susi_params_to_site(site_label)
    sday = site_params.simulation_config.start_date
    end_date = site_params.simulation_config.end_date
    sfc = site_params.stand_params.site_fertility_class
    with open_susi_netcdf(site_label) as ncf:
        vol = np.array(ncf["stand"]["volume"][0, :, 1:-1])
        _, COLS = np.shape(vol)
        dfvol = pd.DataFrame(vol, columns=range(COLS))

        dwt = ncf["strip"]["dwtyr_growingseason"][0, :, 1:-1]
        yrs, COLS = np.shape(dwt)
        dfwt = pd.DataFrame(dwt, columns=range(COLS))

        peat_t = ncf["temperature"]["T"][0, :, 3]
        days = np.shape(peat_t)[0]
        dft = pd.DataFrame(
            peat_t, columns=pd.Index(["T"]), index=pd.date_range(sday, periods=days)
        )

        esom_co2 = ncf["esom"]["Mass"]["co2"][0, :, 1:-1] / 10.0

    empirical = np.zeros(np.shape(dwt))
    for i, yr in enumerate(range(sday.year, end_date.year + 1)):
        t_peat_yr = np.ravel(dft.loc[str(yr)].values)
        rhet = ojanen_2010(sfc, vol[i + 1], t_peat_yr, dfwt.loc[i + 1].values * -100.0)
        empirical[i + 1, :] = rhet

    return vol, yrs, dfvol, dfwt, dft, esom_co2, empirical


def ojanen_2010(sfc, stand_v, t_peat, gs_wt):

    bd_d = {
        2: 0.14,
        3: 0.11,
        4: 0.10,
        5: 0.08,
    }  # Mese study: bulk densities in different fertility classes     g/cm3                                                            # peat layer thickness, cm
    bd = (
        bd_d[sfc] * 1000.0
    )  # set the bulk density according to site fertility class        kg/m3
    B = 350.0
    T5zero = -46.02
    T5ref = 10.0
    Rref = (
        0.0695
        + 3.7 * 10 ** (-4) * stand_v
        + 5.4 * 10 ** (-4) * bd
        + 1.2 * 10 ** (-3) * gs_wt
    ) * 24.0  # unit g/m2/h CO2-> g/m2/day
    Rhet = [
        rref * np.exp(B * ((1.0 / (T5ref - T5zero)) - (1.0 / (t_peat - T5zero))))
        for rref in Rref
    ]
    Rhet = np.array(Rhet).T
    return np.sum(Rhet, axis=0)


def load_wt_measurements(site_label: SiteLabel) -> pd.DataFrame:
    file_meas = WT_MEASUREMENT_INFO[site_label].file
    tubes = WT_MEASUREMENT_INFO[site_label].tubes
    dfmeas = pd.read_excel(MEASUREMENTS_FOLDER / file_meas, sheet_name="CSV")
    dfmeas["date"] = pd.to_datetime(
        dict(year=dfmeas.vuosi, month=dfmeas.kk, day=dfmeas.pv)
    )
    dfmeas = dfmeas.set_index("date")
    dfmeas = dfmeas.drop(["vuosi", "kk", "pv"], axis=1)
    dfmeas = dfmeas.drop(columns=[col for col in dfmeas if col not in tubes])
    dfmeas = dfmeas / 100.0 * -1
    return dfmeas


def add_quadrant_shading(x0: float, x1: float) -> None:
    plt.fill_between(
        [x0, x1], [x1, x1], [x0, x1], color=SHADE_COLOR1, alpha=SHADE_ALPHA
    )
    plt.fill_between(
        [x0, x1], [x0, x1], [x0, x0], color=SHADE_COLOR2, alpha=SHADE_ALPHA
    )


def add_subplot_label(ax, label: str) -> None:
    ax.text(
        0.04,
        0.95,
        label,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=SUBPANEL_FONTSIZE,
        transform=ax.transAxes,
        fontweight="bold",
    )


def fit_regression(obs: np.ndarray, sim: np.ndarray) -> tuple[float, str]:
    a, _, _, _ = np.linalg.lstsq(obs[:, np.newaxis], sim, rcond=None)
    eq = "y = " + str(np.round(a[0], 3)) + "x"
    return a[0], eq


def plot_regression_line(x_range: np.ndarray, a: float) -> None:
    plt.plot(x_range, a * x_range, REGRESSION_LINESTYLE, linewidth=REGRESSION_LINEWIDTH)


def plot_scatter_site(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    xerr: np.ndarray | None = None,
    label: str = "",
    color: str | None = None,
    capsize: int = ERRORBAR_CAPSIZE,
    markersize: int = SCATTER_MARKERSIZE,
) -> None:
    plt.plot(x, y, "o", markersize=markersize, label=label, color=color)
    if xerr is not None:
        plt.errorbar(x, y, yerr * 2, xerr * 2, "none", color=color, capsize=capsize)
    else:
        plt.errorbar(x, y, yerr * 2, 0, "none", color=color, capsize=capsize)


# %%

MEASUREMENTS_FOLDER = MEASUREMENTS_DIR / "Pohjavesiaineistot"

# The run susi_2021.py writes: outputs/<run_id>/<stand>/<scenario>/susi.nc
PROJECT_FOLDER = run_dir(PROJECT_DIR, RUN_ID)

BIO_FILEPATH = MEASUREMENTS_DIR / "gr_bio.xlsx"

WT_MEASUREMENT_INFO: dict[SiteLabel, WTMeasurementInfo] = {
    SiteLabel("ansa21"): WTMeasurementInfo(
        name="Ansasaari21",
        file="muhos_2_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3, 4, 5, 6],
        volume_ini=140.0,
        volume_end=187.07,
        hdom_ini=12.8,
        hdom_end=14.6,
    ),
    SiteLabel("ansa26"): WTMeasurementInfo(
        name="Ansasaari26",
        file="muhos_2_pohjavesi_koottu.xlsx",
        tubes=[28, 29, 30],
        volume_ini=109.0,
        volume_end=140.86,
        hdom_ini=12.6,
        hdom_end=15.4,
    ),
    SiteLabel("koira11"): WTMeasurementInfo(
        name="Koirasuo11",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[13, 14, 15],
        volume_ini=90.9,
        volume_end=122.0,
        hdom_ini=12.9,
        hdom_end=13.9,
    ),
    SiteLabel("koira12"): WTMeasurementInfo(
        name="Koirasuo12",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3],
        volume_ini=122.9,
        volume_end=164.0,
        hdom_ini=13.2,
        hdom_end=16.4,
    ),
    SiteLabel("koira21"): WTMeasurementInfo(
        name="Koirasuo21",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[19, 20, 21],
        volume_ini=97.4,
        volume_end=91.9 + 30.6,
        hdom_ini=12.7,
        hdom_end=14.5,
    ),
    SiteLabel("koira22"): WTMeasurementInfo(
        name="Koirasuo22",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[31, 32, 33],
        volume_ini=92.2,
        volume_end=87.13 + 33.92,
        hdom_ini=11.1,
        hdom_end=14.3,
    ),
    SiteLabel("neva11"): WTMeasurementInfo(
        name="Nevajärvi11",
        file="nevajarvi_1_pohjavesi_koottu.xlsx",
        tubes=[5, 6, 7, 8, 9, 10],
        volume_ini=163.4,
        volume_end=227.5,
        hdom_ini=16.1,
        hdom_end=17.8,
    ),
    SiteLabel("neva14"): WTMeasurementInfo(
        name="Nevajärvi14",
        file="nevajarvi_1_pohjavesi_koottu.xlsx",
        tubes=[36, 41, 42, 43, 44, 45],
        volume_ini=142.9,
        volume_end=191.4,
        hdom_ini=16.5,
        hdom_end=18.4,
    ),
    SiteLabel("neva21"): WTMeasurementInfo(
        name="Nevajärvi21",
        file="nevajarvi_2_pohjavesi_koottu.xlsx",
        tubes=[5, 6, 7, 8, 9, 10, 11],
        volume_ini=100.32,
        volume_end=109.89 + 32.5,
        hdom_ini=15.7,
        hdom_end=16.7,
    ),
    SiteLabel("neva24"): WTMeasurementInfo(
        name="Nevajärvi24",
        file="nevajarvi_2_pohjavesi_koottu.xlsx",
        tubes=[36, 41, 42, 43, 44, 45],
        volume_ini=143.6,
        volume_end=124.7 + 53.66,
        hdom_ini=14.6,
        hdom_end=17.1,
    ),
    SiteLabel("neva31"): WTMeasurementInfo(
        name="Nevajärvi31",
        file="nevajarvi_3_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        volume_ini=127.2,
        volume_end=161.5,
        hdom_ini=16.5,
        hdom_end=17.0,
    ),
    SiteLabel("neva34"): WTMeasurementInfo(
        name="Nevajärvi34",
        file="nevajarvi_3_pohjavesi_koottu.xlsx",
        tubes=[41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        volume_ini=100.4,
        volume_end=147.3,
        hdom_ini=15.3,
        hdom_end=17.1,
    ),
    SiteLabel("jaakkoin61"): WTMeasurementInfo(
        name="Jaakkoinsuo61",
        file="jaakkoinsuo_pohjavesi_koottu.xlsx",
        tubes=[2, 3, 12, 13, 14, 15],
        volume_ini=144.3,
        volume_end=172.6,
        hdom_ini=17.6,
        hdom_end=18.5,
    ),
    SiteLabel("jaakkoin62"): WTMeasurementInfo(
        name="Jaakkoinsuo62",
        file="jaakkoinsuo_pohjavesi_koottu.xlsx",
        tubes=[28, 30, 36, 39, 42, 43, 44, 45],
        volume_ini=149.3,
        volume_end=186.6,
        hdom_ini=18.1,
        hdom_end=19.5,
    ),
    SiteLabel("parkano11"): WTMeasurementInfo(
        name="Parkano11",
        file="parkano_1_pohjavesi_koottu.xlsx",
        tubes=[22, 23, 24, 27, 28, 29, 32, 33, 34, 37, 38, 39],
        volume_ini=167.7,
        volume_end=192.4,
        hdom_ini=17.3,
        hdom_end=18.7,
    ),
    SiteLabel("parkano12"): WTMeasurementInfo(
        name="Parkano12",
        file="parkano_3_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 9, 10, 11, 12, 19, 20, 21, 22, 29, 30, 31, 32],
        volume_ini=263.0,
        volume_end=173.0 + 112.9,
        hdom_ini=20.0,
        hdom_end=20.7,
    ),
}


# %% WT time series figure
# Measured and modelled WT as time series

COLS = 2
ROWS = 6

SITES = (
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
)

fig = plt.figure(constrained_layout=True, figsize=(10.0, 15.0))
gs = gridspec.GridSpec(ROWS, COLS, figure=fig)
coordinates = [
    gs[0, 0],
    gs[0, 1],
    gs[1, 0],
    gs[1, 1],
    gs[2, 0],
    gs[2, 1],
    gs[3, 0],
    gs[3, 1],
    gs[4, 0],
    gs[4, 1],
    gs[5, 0],
]


abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

assert len(SITES) == len(abc)
assert len(SITES) == len(coordinates)

for i, (crd, site_label, tx) in enumerate(zip(coordinates, SITES, abc)):
    print(crd, site_label)

    dfmeas = load_wt_measurements(site_label)

    dfsim, sday, end_date = load_simulation_wt(site_label)

    fs = 8
    wt_max = np.max(dfsim, axis=1)
    wt_min = np.min(dfsim, axis=1)
    ax = fig.add_subplot(crd)
    ax.fill_between(dfsim.index, wt_max, wt_min, color="grey", alpha=0.3)
    ax.set_ylim((-1.0, 0.0))
    ax.plot(dfmeas, "bo", markersize=2)
    ax.set_xlim((mdates.date2num(sday), mdates.date2num(end_date)))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    site_name = WT_MEASUREMENT_INFO[site_label].name

    ax.text(
        0.02,
        0.95,
        tx + "  " + site_name,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=14,
        transform=ax.transAxes,
    )

    for xlabel_i in ax.get_xticklabels():
        xlabel_i.set_fontsize(TICK_FONTSIZE)
        xlabel_i.set_y(-0.01)

    if i in [1, 3, 5, 7, 9]:
        for ylabel_i in ax.get_yticklabels():
            ylabel_i.set_fontsize(0.0)
            ylabel_i.set_visible(False)
    else:
        for ylabel_i in ax.get_yticklabels():
            ylabel_i.set_fontsize(TICK_FONTSIZE)
            ylabel_i.set_x(-0.025)

        ax.set_ylabel("WT, m", fontsize=14)

    if i in [9, 10]:
        ax.set_xlabel("Time", fontsize=14)
plt.tight_layout(h_pad=1.5)
plt.show()

# %% calculate measured and modelled late-summer WT and save to out{} dictionary


out = {}
for site_label in SITES:
    dfmeas = load_wt_measurements(site_label)

    dfsim, start_date, end_date = load_simulation_wt(site_label)

    print(site_label, start_date, end_date)
    dfmeas = dfmeas.sort_index()[str(dfsim.index[0]) : str(dfsim.index[-1])]

    out[site_label] = {}
    meansim = []
    stdsim = []
    meanmeas = []
    stdmeas = []
    for yr in range(start_date.year, end_date.year + 1):
        wtsim = (
            dfsim[str(yr) + "-07-01" : str(yr) + "-08-31"].mean().values
        )  # .values[:-1])
        meansim.append(wtsim.mean())
        stdsim.append(wtsim.std())
        wtmeas = (
            dfmeas[str(yr) + "-07-01" : str(yr) + "-08-31"].mean().values
        )  # .values[:-1])
        meanmeas.append(wtmeas.mean())
        stdmeas.append(wtmeas.std())
    out[site_label]["meansim"] = meansim
    out[site_label]["stdsim"] = stdsim
    out[site_label]["meanmeas"] = meanmeas
    out[site_label]["stdmeas"] = stdmeas
print(out)


# %% Collect biomass and volume growth from netcdf files

gro = np.zeros(len(SITES))
grosd = np.zeros(len(SITES))
bm_gr = np.zeros(len(SITES))
bm_gr_sd = np.zeros(len(SITES))

for i, site_label in enumerate(SITES):
    gro[i], grosd[i], bm_gr[i], bm_gr_sd[i] = load_stand_growth(site_label)


dvol = {"sites": SITES, "grsim": gro, "grsd": grosd, "bmgr": bm_gr, "bmgrsd": bm_gr_sd}
dfvols = pd.DataFrame(data=dvol)
dfvols = dfvols.set_index("sites")

print(dfvols)


# %% KUVA N: wt, biomass, vol
"""
needed: 
    grsim = vend - vini
    yrs
"""


dfbio = pd.read_excel(BIO_FILEPATH, index_col="site")

obs = []
pred = []
dfvols["grobs"] = dfvols["grsim"] * 0.0
dfvols["yrs"] = dfvols["grsim"] * 0.0
dfvols["bioobs"] = dfvols["grsim"] * 0.0

for site in SITES:
    site_params = assign_susi_params_to_site(site)
    print(
        site,
        np.round(
            WT_MEASUREMENT_INFO[site].volume_end - WT_MEASUREMENT_INFO[site].volume_ini
        ),
        dfvols.loc[site]["grsim"],
    )
    dfvols.at[site, "grobs"] = (
        WT_MEASUREMENT_INFO[site].volume_end - WT_MEASUREMENT_INFO[site].volume_ini
    )
    dfvols.at[site, "yrs"] = (
        site_params.simulation_config.end_date.year
        - site_params.simulation_config.start_date.year
        + 1.0
    )
    dfvols.at[site, "bioobs"] = dfbio.at[site, "gr_bio"]

fs = AXIS_LABEL_FONTSIZE
nsites = len(SITES)
figsi = (10, 10)
fig = plt.figure(num="growth", figsize=figsi)
gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, wspace=0.25, hspace=0.25)


colors = plt.colormaps["jet"](np.linspace(0, 1, nsites))

# -----------WT figure ------------------------------------------------

ax0 = fig.add_subplot(gs[0, 0])

add_quadrant_shading(-1.0, 0.0)
si = []
ob = []
for c, site_label in enumerate(SITES):
    site_name = WT_MEASUREMENT_INFO[site_label].name
    data = out[site_label]
    obs = np.array(data["meanmeas"])
    ob.extend(obs)
    obserr = np.array(data["stdmeas"])
    pre = np.array(data["meansim"])
    si.extend(pre)
    preerr = np.array(data["stdsim"])
    plot_scatter_site(obs, pre, preerr, xerr=obserr, label=site_name, color=colors[c])
si = np.array(si)
ob = np.array(ob)

diff = ob - si
diff = diff[~np.isnan(diff)]
rmse = np.round(np.sqrt(np.square(diff).mean()), 3)

data = {"meas": ob, "sim": si}
dfdata = pd.DataFrame.from_dict(data)
dfdata = dfdata.dropna()
# DEAD: slope, intercept, r_value, p_value, std_err = stats.linregress(dfdata['meas'].values, dfdata['sim'].values)

a_wt, eq_wt = fit_regression(dfdata["meas"].values, dfdata["sim"].values)
x_wt = np.arange(-0.9, -0.03, 0.05)
plot_regression_line(x_wt, a_wt)


plt.xlim([-1.0, 0.0])
plt.ylim([-1.0, 0.0])
plt.text(-0.9, -0.1, eq_wt, fontsize=AXIS_LABEL_FONTSIZE - 1)
plt.text(-0.9, -0.15, "RMSE " + str(rmse), fontsize=AXIS_LABEL_FONTSIZE - 1)
plt.xlabel("Observed $\it{WT}$, m", fontsize=AXIS_LABEL_FONTSIZE)
plt.ylabel("Predicted $\it{WT}$, m", fontsize=AXIS_LABEL_FONTSIZE)

add_subplot_label(ax0, "a")

# ----------BM figure ---------------------------------------------------
ax1 = fig.add_subplot(gs[0, 1])

mval = 10000.0
add_quadrant_shading(0.0, mval)

for c, site_label in enumerate(SITES):
    site_name = WT_MEASUREMENT_INFO[site_label].name
    obs = dfvols.loc[site_label]["bioobs"]
    pre = dfvols.loc[site_label]["bmgr"]
    preerr = dfvols.loc[site_label]["bmgrsd"]
    plot_scatter_site(obs, pre, preerr, label=site_name, color=colors[c])
plt.xlabel("Observed bm growth, $kg \ ha^{-1} yr^{-1}$ ", fontsize=AXIS_LABEL_FONTSIZE)
plt.ylabel(
    "Predicted bm growth, $kg \ ha^{-1} yr^{-1}$ ",
    fontsize=AXIS_LABEL_FONTSIZE,
    labelpad=-7.5,
)
plt.xlim([0, mval])
plt.ylim([0, mval])
rmse = np.round(
    np.sqrt(
        np.square(dfvols["bioobs"].values[:, np.newaxis] - dfvols["bmgr"].values).mean()
    ),
    0,
)
a_bm, eq_bm = fit_regression(dfvols["bioobs"].values, dfvols["bmgr"].values)
x_bm = np.arange(1000.0, 9000.0, 100)
plot_regression_line(x_bm, a_bm)
plt.text(5000, 8300, "RMSE " + str(rmse), fontsize=AXIS_LABEL_FONTSIZE - 1)
plt.text(5000, 8900, eq_bm, fontsize=AXIS_LABEL_FONTSIZE - 1)


add_subplot_label(ax1, "b")

# -------------------Vol figure ------------------
ax2 = fig.add_subplot(gs[1, 0])

mval = 12.0
add_quadrant_shading(0.0, mval)
obsvols = []
prevols = []

for c, site_label in enumerate(SITES):
    site_name = WT_MEASUREMENT_INFO[site_label].name

    site_params = assign_susi_params_to_site(site_label)

    yrs = (
        site_params.simulation_config.end_date.year
        - site_params.simulation_config.start_date.year
        + 1.0
    )
    obs = dfvols.loc[site_label]["grobs"] / yrs
    pre = dfvols.loc[site_label]["grsim"]
    preerr = dfvols.loc[site_label]["grsd"]
    plot_scatter_site(obs, pre, preerr, label=site_name, color=colors[c])
    obsvols.append(obs)
    prevols.append(pre)

ax1.legend(
    loc="upper center",
    bbox_to_anchor=(-0.15, 1.3),
    ncol=4,
    fontsize=AXIS_LABEL_FONTSIZE - 3,
)
plt.xlim([0, mval])
plt.ylim([0, mval])

plt.xlabel(
    "Observed $\it{i_V}$, $m^{3} ha^{-1} yr^{-1}$ ", fontsize=AXIS_LABEL_FONTSIZE
)
plt.ylabel(
    "Predicted $\it{i_V}$, $m^{3} ha^{-1} yr^{-1}$ ", fontsize=AXIS_LABEL_FONTSIZE
)

obs_growths = dfvols["grobs"].values / dfvols["yrs"].values
sim_growths = dfvols["grsim"].values
# DEAD: slope, intercept, r_value, p_value, std_err = stats.linregress(obs_growths, sim_growths)

diffv = obs_growths - sim_growths
rmse = np.round(np.sqrt(np.square(diffv).mean()), 3)
a_v, eq_v = fit_regression(obs_growths, sim_growths)

x_v = np.arange(1.0, 8.0, 0.5)
plt.text(6, 8.3, "RMSE " + str(rmse), fontsize=AXIS_LABEL_FONTSIZE - 1)
plot_regression_line(x_v, a_v)
predicted_v = np.ravel(a_v * obs_growths)
plt.text(6, 9, eq_v, fontsize=AXIS_LABEL_FONTSIZE - 1)

add_subplot_label(ax2, "c")

# -------------------CO2 figure ------------------
ax3 = fig.add_subplot(gs[1, 1])

mval = 3000.0
add_quadrant_shading(0.0, mval)
esarr = np.empty(0)
emps = np.empty(0)
for c, site_label in enumerate(SITES):
    site_name = WT_MEASUREMENT_INFO[site_label].name

    vol, yrs, dfvol, dfwt, dft, esom_co2, empirical = load_co2_data(site_label)

    with open_susi_netcdf(site_label) as ncf:
        ojanen_2019 = np.mean(ncf["ojanen"]["soil_co2_balance"][0, :, 1:-1])
        soil_bal = (
            np.mean(
                ncf["balance"]["C"]["stand_litter_in"][0, :, 1:-1]
                + ncf["balance"]["C"]["gv_litter_in"][0, :, 1:-1]
                - ncf["balance"]["C"]["co2c_release"][0, :, 1:-1]
            )
            * 44
            / 12
        )

        print(site_label, ojanen_2019, soil_bal)
    for m in range(1, yrs):
        plt.plot(empirical[m, :], esom_co2[m, :], color=colors[c])

    esarr = np.append(esarr, np.ravel(esom_co2[1:, :]))
    emps = np.append(emps, np.ravel(empirical[1:, :]))

plt.xlim([0, mval])
plt.ylim([0, mval])

plt.xlabel("Empirical $CO_2$, $g m^{-2} yr^{-1}$ ", fontsize=AXIS_LABEL_FONTSIZE)
plt.ylabel(
    "Esom $CO_2$, $g m^{-3} yr^{-2}$ ", fontsize=AXIS_LABEL_FONTSIZE, labelpad=-5.0
)

# DEAD: slope, intercept, r_value, p_value, std_err = stats.linregress(emps, esarr)

a_c, eq_c = fit_regression(emps, esarr)
x_c = np.arange(200.0, 2000.0, 0.5)
plot_regression_line(x_c, a_c)
predicted_c = np.ravel(a_c * emps)
plt.text(2000, 2000, eq_c, fontsize=AXIS_LABEL_FONTSIZE - 1)

add_subplot_label(ax3, "d")

plt.show()

# %%


for site_label in SITES[:1]:
    _, _, _, _, _, esom_co2, empirical = load_co2_data(site_label)

dfesom = pd.DataFrame(data=esom_co2)
dfempirical = pd.DataFrame(data=empirical)
# %%

site_names = [WT_MEASUREMENT_INFO[site_label].name for site_label in SITES]
sfcs = list(SITE_FERTILITY_CLASSES)
print("***********************")
dfresid = pd.DataFrame(
    list(zip(site_names, sfcs, obsvols, prevols)),
    columns=pd.Index(["name", "sfc", "obs", "pre"]),
)
dfresid["residual"] = dfresid["pre"] - dfresid["obs"]
print(dfresid)
