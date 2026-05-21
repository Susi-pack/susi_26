# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:04:05 2020

@author: alauren, modified by iurzainki
"""

from pathlib import Path
from typing import NewType
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats

# from dwts_para import para
from netCDF4 import Dataset

# from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from susi.io.app_settings import AppSettings
from inputs.parameters.para_2021 import (
    SiteLabel,
    get_scenario_label_from_site_label,
    get_stand_label_from_site_label,
    assign_susi_params_to_site,
)


sns.set()
# %%


@dataclass(frozen=True)
class WTMeasurementInfo:
    name: str
    file: str
    tubes: list[int]


# %%


def get_netcdf_path_from_site_label(
    site_label: SiteLabel, project_folderpath: Path
) -> Path:
    stand_label = get_stand_label_from_site_label(site_label)
    scenario_label = get_scenario_label_from_site_label(site_label)

    return project_folderpath / stand_label / scenario_label / "susi.nc"


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


# %%

_app_settings = AppSettings()

MEASUREMENTS_FOLDER = (
    _app_settings.project_root_path / "inputs/susi_2021/Pohjavesiaineistot"
)

PROJECT_FOLDER = _app_settings.output_folder / "susi_2021"

# params = para(period="start-end")

WT_MEASUREMENT_INFO: dict[SiteLabel, WTMeasurementInfo] = {
    SiteLabel("ansa21"): WTMeasurementInfo(
        name="Ansasaari21",
        file="muhos_2_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3, 4, 5, 6],
    ),
    SiteLabel("ansa26"): WTMeasurementInfo(
        name="Ansasaari26",
        file="muhos_2_pohjavesi_koottu.xlsx",
        tubes=[28, 29, 30],
    ),
    SiteLabel("koira11"): WTMeasurementInfo(
        name="Koirasuo11",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[13, 14, 15],
    ),
    SiteLabel("koira12"): WTMeasurementInfo(
        name="Koirasuo12",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3],
    ),
    SiteLabel("koira21"): WTMeasurementInfo(
        name="Koirasuo21",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[19, 20, 21],
    ),
    SiteLabel("koira22"): WTMeasurementInfo(
        name="Koirasuo22",
        file="koiraoja_pohjavesi_koottu.xlsx",
        tubes=[31, 32, 33],
    ),
    SiteLabel("neva11"): WTMeasurementInfo(
        name="Nevajärvi11",
        file="nevajarvi_1_pohjavesi_koottu.xlsx",
        tubes=[5, 6, 7, 8, 9, 10],
    ),
    SiteLabel("neva14"): WTMeasurementInfo(
        name="Nevajärvi14",
        file="nevajarvi_1_pohjavesi_koottu.xlsx",
        tubes=[36, 41, 42, 43, 44, 45],
    ),
    SiteLabel("neva21"): WTMeasurementInfo(
        name="Nevajärvi21",
        file="nevajarvi_2_pohjavesi_koottu.xlsx",
        tubes=[5, 6, 7, 8, 9, 10, 11],
    ),
    SiteLabel("neva24"): WTMeasurementInfo(
        name="Nevajärvi24",
        file="nevajarvi_2_pohjavesi_koottu.xlsx",
        tubes=[36, 41, 42, 43, 44, 45],
    ),
    SiteLabel("neva31"): WTMeasurementInfo(
        name="Nevajärvi31",
        file="nevajarvi_3_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ),
    SiteLabel("neva34"): WTMeasurementInfo(
        name="Nevajärvi34",
        file="nevajarvi_3_pohjavesi_koottu.xlsx",
        tubes=[41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    ),
    SiteLabel("jaakkoin61"): WTMeasurementInfo(
        name="Jaakkoinsuo61",
        file="jaakkoinsuo_pohjavesi_koottu.xlsx",
        tubes=[2, 3, 12, 13, 14, 15],
    ),
    SiteLabel("jaakkoin62"): WTMeasurementInfo(
        name="Jaakkoinsuo62",
        file="jaakkoinsuo_pohjavesi_koottu.xlsx",
        tubes=[28, 30, 36, 39, 42, 43, 44, 45],
    ),
    SiteLabel("parkano11"): WTMeasurementInfo(
        name="Parkano11",
        file="parkano_1_pohjavesi_koottu.xlsx",
        tubes=[22, 23, 24, 27, 28, 29, 32, 33, 34, 37, 38, 39],
    ),
    SiteLabel("parkano12"): WTMeasurementInfo(
        name="Parkano12",
        file="parkano_3_pohjavesi_koottu.xlsx",
        tubes=[1, 2, 9, 10, 11, 12, 19, 20, 21, 22, 29, 30, 31, 32],
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

# sites =['neva11']
i = 0
assert len(SITES) == len(abc)
assert len(SITES) == len(coordinates)

for crd, site_label, tx in zip(coordinates, SITES, abc):
    print(crd, site_label)

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

    site_params = assign_susi_params_to_site(site_label)
    sday = site_params.simulation_config.start_date
    end_date = site_params.simulation_config.end_date
    ncf = Dataset(
        get_netcdf_path_from_site_label(
            site_label=SiteLabel(site_label), project_folderpath=PROJECT_FOLDER
        ),
        mode="r",
    )
    dwt = ncf["strip"]["dwt"][0, :, 1:-1]
    days, COLS = np.shape(dwt)
    dfsim = pd.DataFrame(
        dwt, columns=range(COLS), index=pd.date_range(sday, periods=days)
    )
    ncf.close()

    fs = 8
    wt_max = np.max(dfsim, axis=1)
    wt_min = np.min(dfsim, axis=1)
    ax = fig.add_subplot(crd)
    ax.fill_between(dfsim.index, wt_max, wt_min, color="grey", alpha=0.3)
    ax.set_ylim((-1.0, 0.0))
    ax.plot(dfmeas, "bo", markersize=2)
    ax.set_xlim((sday, end_date))
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
        xlabel_i.set_fontsize(11.0)
        xlabel_i.set_y(-0.01)

    if i in [1, 3, 5, 7, 9]:
        for ylabel_i in ax.get_yticklabels():
            ylabel_i.set_fontsize(0.0)
            ylabel_i.set_visible(False)
    else:
        for ylabel_i in ax.get_yticklabels():
            ylabel_i.set_fontsize(11.0)
            ylabel_i.set_x(-0.025)

        ax.set_ylabel("WT, m", fontsize=14)

    if i in [9, 10]:
        ax.set_xlabel("Time", fontsize=14)
    i += 1
plt.tight_layout(h_pad=1.5)
plt.show()

# %% calculate measured and modelled late-summer WT and save to out{} dictionary


out = {}
for site_label in SITES:
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

    ff = (
        r"C:/Users/laurenan/OneDrive - University of Helsinki/SUSI/vesitase/vesitase_out/"
        + site_label
        + ".nc"
    )
    site_params = assign_susi_params_to_site(site_label)
    start_date = site_params.simulation_config.start_date
    end_date = site_params.simulation_config.end_date

    ncf = Dataset(
        get_netcdf_path_from_site_label(
            site_label=SiteLabel(site_label), project_folderpath=PROJECT_FOLDER
        ),
        mode="r",
    )
    dwt = ncf["strip"]["dwt"][0, :, 1:-1]
    days, COLS = np.shape(dwt)
    dfsim = pd.DataFrame(
        dwt, columns=range(COLS), index=pd.date_range(start_date, periods=days)
    )
    ncf.close()

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
    site_params = assign_susi_params_to_site(site_label)
    sday = site_params.simulation_config.start_date
    end_date = site_params.simulation_config.end_date
    ncf = Dataset(
        get_netcdf_path_from_site_label(
            site_label=SiteLabel(site_label), project_folderpath=PROJECT_FOLDER
        ),
        mode="r",
    )
    vol = np.array(ncf["stand"]["volume"][0, :, 1:-1])
    growth = np.array(ncf["stand"]["volumegrowth"][0, :, 1:-1])
    print(site_label, growth)
    # growth = np.diff(vol, axis=0)
    yrs, COLS = np.shape(growth)
    dfgrowth = pd.DataFrame(growth, columns=range(COLS))

    bm_growth = np.array(
        ncf["stand"]["dominant"]["NPP"][0, :, 1:-1] * ncf["stand"]["stems"][0, :, 1:-1]
    )
    dfbm = pd.DataFrame(bm_growth, columns=range(COLS))

    ncf.close()
    gro[i] = np.mean(dfgrowth.mean(axis=0).values)
    grosd[i] = np.std(dfgrowth.mean(axis=0).values)
    bm_gr[i] = np.mean(dfbm.mean(axis=0).values)
    bm_gr_sd[i] = np.std(dfbm.mean(axis=0).values)


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


fbio = r"C:/Users/laurenan/OneDrive - University of Helsinki/SUSI/vesitase/gr_bio.xlsx"
dfbio = pd.read_excel(fbio, index_col="site")

obs = []
pred = []
dfvols["grobs"] = dfvols["grsim"] * 0.0
dfvols["yrs"] = dfvols["grsim"] * 0.0
dfvols["bioobs"] = dfvols["grsim"] * 0.0

for s, i in zip(sites, names):
    print(
        s, np.round(params[s]["vol"][1] - params[s]["vol"][0]), dfvols.loc[s]["grsim"]
    )
    dfvols.at[s, "grobs"] = params[s]["vol"][1] - params[s]["vol"][0]
    dfvols.at[s, "yrs"] = (
        params[s]["end_date"].year - params[s]["start_date"].year + 1.0
    )
    dfvols.at[s, "bioobs"] = dfbio.at[s, "gr_bio"]

fs = 16
nsites = 11
figsi = (10, 10)
fig = plt.figure(num="growth", figsize=figsi)  # Figsize(w,h), tuple inches
col1 = "yellow"
col2 = "grey"
gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, wspace=0.25, hspace=0.25)


colors = plt.cm.jet(np.linspace(0, 1, nsites))
# -----------WT figure ------------------------------------------------
# ax0 = fig.add_axes([0.08, 0.15, 0.25, 0.75]) #left, bottom, width, height)
ax0 = fig.add_subplot(gs[0, 0])

mval = 0.0
col1 = "yellow"
col2 = "grey"
plt.fill_between([-1.0, mval], [mval, mval], [-1.0, mval], color=col1, alpha=0.3)
plt.fill_between([-1.0, mval], [-1.0, 0.0], [-1.0, -1.0], color=col2, alpha=0.3)

colors = plt.cm.jet(np.linspace(0, 1, len(sites)))
si = []
ob = []
c = 0
for site_label, i in zip(sites, names):
    data = out[site_label]
    obs = np.array(data["meanmeas"])
    ob.extend(obs)
    obserr = np.array(data["stdmeas"])
    pre = np.array(data["meansim"])
    si.extend(pre)
    preerr = np.array(data["stdsim"])
    plt.plot(obs, pre, "o", markersize=10, label=i, color=colors[c])
    plt.errorbar(obs, pre, preerr * 2, obserr * 2, "none", color=colors[c], capsize=4)
    c += 1
# plt.legend(loc='lower right', ncol=3)
si = np.array(si)
ob = np.array(ob)

diff = ob - si
diff = diff[~np.isnan(diff)]
rmse = np.round(np.sqrt(np.square(diff).mean()), 3)

data = {"meas": ob, "sim": si}
dfdata = pd.DataFrame.from_dict(data)
dfdata = dfdata.dropna()
# slope, intercept, r_value, p_value, std_err = stats.linregress(dfdata['meas'].values, dfdata['sim'].values)

fs = 16

obs_wt = dfdata["meas"].values[:, np.newaxis]
sim_wt = dfdata["sim"].values
a, _, _, _ = np.linalg.lstsq(obs_wt, sim_wt, rcond=None)
eq = "y = " + str(np.round(a[0], 3)) + "x"
x = np.arange(-0.9, -0.03, 0.05)
plt.plot(x, a * x, "k--", linewidth=2)


plt.xlim([-1.0, 0.0])
plt.ylim([-1.0, 0.0])
plt.text(-0.9, -0.1, eq, fontsize=fs - 1)
plt.text(-0.9, -0.15, "RMSE " + str(rmse), fontsize=fs - 1)
plt.xlabel("Observed $\it{WT}$, m", fontsize=fs)
plt.ylabel("Predicted $\it{WT}$, m", fontsize=fs)

# for tick in ax0.xaxis.get_major_ticks():
#    tick.label.set_fontsize(14)
# for tick in ax0.yaxis.get_major_ticks():
#    tick.label.set_fontsize(14)

ax0.text(
    0.04,
    0.95,
    "a",
    horizontalalignment="left",
    verticalalignment="top",
    fontsize=18,
    transform=ax0.transAxes,
    fontweight="bold",
)

# ----------BM figure ---------------------------------------------------
ax1 = fig.add_subplot(gs[0, 1])

mval = 10000.0
plt.fill_between([0.0, mval], [mval, mval], [0.0, mval], color=col1, alpha=0.3)
plt.fill_between([0.0, mval], [0.0, mval], [0.0, 0.0], color=col2, alpha=0.3)

# dfvols.to_excel(folder_out+'dfvols.xlsx')
c = 0
for s, i in zip(sites, names):
    obs = dfvols.loc[s]["bioobs"]
    pre = dfvols.loc[s]["bmgr"]
    plt.plot(obs, pre, "o", markersize=10, label=i, color=colors[c])
    preerr = dfvols.loc[s]["bmgrsd"]
    plt.errorbar(obs, pre, preerr * 2, 0, "none", color=colors[c], capsize=4)

    c += 1
plt.xlabel("Observed bm growth, $kg \ ha^{-1} yr^{-1}$ ", fontsize=fs)
plt.ylabel("Predicted bm growth, $kg \ ha^{-1} yr^{-1}$ ", fontsize=fs, labelpad=-7.5)
plt.xlim([0, mval])
plt.ylim([0, mval])
obs_growths = dfvols["bioobs"].values[:, np.newaxis]
sim_growths = dfvols["bmgr"].values

rmse = np.round(np.sqrt(np.square(obs_growths - sim_growths).mean()), 0)
a, _, _, _ = np.linalg.lstsq(obs_growths, sim_growths, rcond=None)
eq = "y = " + str(np.round(a[0], 3)) + "x"
x = np.arange(1000.0, 9000.0, 100)
plt.plot(x, a * x, "k--", linewidth=2)
plt.text(5000, 8300, "RMSE " + str(rmse), fontsize=fs - 1)


plt.text(5000, 8900, eq, fontsize=fs - 1)


ax1.text(
    0.04,
    0.95,
    "b",
    horizontalalignment="left",
    verticalalignment="top",
    fontsize=18,
    transform=ax1.transAxes,
    fontweight="bold",
)

# for tick in ax1.xaxis.get_major_ticks():
#    tick.label.set_fontsize(14)
# for tick in ax1.yaxis.get_major_ticks():
#    tick.label.set_fontsize(14)

# -------------------Vol figure ------------------
ax2 = fig.add_subplot(gs[1, 0])

mval = 12.0
plt.fill_between([0.0, mval], [mval, mval], [0.0, mval], color=col1, alpha=0.3)
plt.fill_between([0.0, mval], [0.0, mval], [0.0, 0.0], color=col2, alpha=0.3)
c = 0
obsvols = []
prevols = []

for s, i in zip(sites, names):
    yrs = params[s]["end_date"].year - params[s]["start_date"].year + 1.0
    obs = dfvols.loc[s]["grobs"] / yrs
    pre = dfvols.loc[s]["grsim"]
    plt.plot(obs, pre, "o", markersize=10, label=i, color=colors[c])
    preerr = dfvols.loc[s]["grsd"]
    plt.errorbar(obs, pre, preerr * 2, 0, "none", color=colors[c], capsize=4)
    obsvols.append(obs)
    prevols.append(pre)
    c += 1

ax1.legend(loc="upper center", bbox_to_anchor=(-0.15, 1.3), ncol=4, fontsize=fs - 3)
# ax1.legend(loc='lower left', ncol=1, fontsize=fs-3)
plt.xlim([0, mval])
plt.ylim([0, mval])

plt.xlabel("Observed $\it{i_V}$, $m^{3} ha^{-1} yr^{-1}$ ", fontsize=fs)
plt.ylabel("Predicted $\it{i_V}$, $m^{3} ha^{-1} yr^{-1}$ ", fontsize=fs)

obs_growths = dfvols["grobs"].values / dfvols["yrs"].values
sim_growths = dfvols["grsim"].values
slope, intercept, r_value, p_value, std_err = stats.linregress(obs_growths, sim_growths)

diffv = dfvols["grobs"].values / dfvols["yrs"].values - dfvols["grsim"].values
rmse = np.round(np.sqrt(np.square(diffv).mean()), 3)
obs_growths = obs_growths[:, np.newaxis]
a, _, _, _ = np.linalg.lstsq(obs_growths, sim_growths, rcond=None)


eq = "y = " + str(np.round(a[0], 3)) + "x"
x = np.arange(1.0, 8.0, 0.5)
plt.text(6, 8.3, "RMSE " + str(rmse), fontsize=fs - 1)

plt.plot(x, a * x, "k--", linewidth=2)
predicted = np.ravel(a[0] * obs_growths)

plt.text(6, 9, eq, fontsize=fs - 1)

ax2.text(
    0.04,
    0.95,
    "c",
    horizontalalignment="left",
    verticalalignment="top",
    fontsize=18,
    transform=ax2.transAxes,
    fontweight="bold",
)

# for tick in ax2.xaxis.get_major_ticks():
#    tick.label.set_fontsize(14)
# for tick in ax2.yaxis.get_major_ticks():
#    tick.label.set_fontsize(14)

# -------------------CO2 figure ------------------
ax3 = fig.add_subplot(gs[1, 1])

mval = 3000.0
plt.fill_between([0.0, mval], [mval, mval], [0.0, mval], color=col1, alpha=0.3)
plt.fill_between([0.0, mval], [0.0, mval], [0.0, 0.0], color=col2, alpha=0.3)
esarr = np.empty(0)
emps = np.empty(0)
c = 0
for site_label in sites:
    ff = (
        r"C:/Users/laurenan/OneDrive - University of Helsinki/SUSI/vesitase/vesitase_out/"
        + site_label
        + ".nc"
    )
    params = para(period="start-end")
    sday = params[site_label]["start_date"]
    end_date = params[site_label]["end_date"]
    sfc = params[site_label]["sfc"]
    ncf = Dataset(ff, mode="r")  # water netCDF, open in reading mode
    vol = np.array(ncf["stand"]["volume"][0, :, 1:-1])
    yrs, COLS = np.shape(vol)
    dfvol = pd.DataFrame(vol, columns=range(COLS))

    dwt = ncf["strip"]["dwtyr_growingseason"][0, :, 1:-1]
    yrs, COLS = np.shape(dwt)
    dfwt = pd.DataFrame(dwt, columns=range(COLS))

    peat_t = ncf["temperature"]["T"][0, :, 3]
    days = np.shape(peat_t)[0]
    dft = pd.DataFrame(peat_t, columns=["T"], index=pd.date_range(sday, periods=days))

    esom_co2 = (
        ncf["esom"]["Mass"]["co2"][0, :, 1:-1] / 10.0
    )  # kg ha-1 yr-1 ->g m-2 yr-1

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

    ncf.close()
    empirical = np.zeros(np.shape(dwt))
    for i, yr in enumerate(range(sday.year, end_date.year + 1)):
        t_peat_yr = np.ravel(dft.loc[str(yr)].values)
        rhet = ojanen_2010(sfc, vol[i + 1], t_peat_yr, dfwt.loc[i + 1].values * -100.0)
        empirical[i + 1, :] = rhet
    for m in range(1, yrs):
        plt.plot(empirical[m, :], esom_co2[m, :], color=colors[c])

    c += 1

    esarr = np.append(esarr, np.ravel(esom_co2[1:, :]))
    emps = np.append(emps, np.ravel(empirical[1:, :]))

plt.xlim([0, mval])
plt.ylim([0, mval])

plt.xlabel("Empirical $CO_2$, $g m^{-2} yr^{-1}$ ", fontsize=fs)
plt.ylabel("Esom $CO_2$, $g m^{-3} yr^{-2}$ ", fontsize=fs, labelpad=-5.0)

obs_growths = emps
sim_growths = esarr
slope, intercept, r_value, p_value, std_err = stats.linregress(obs_growths, sim_growths)

# diffv = dfvols['grobs'].values/dfvols['yrs'].values - dfvols['grsim'].values
# rmse = np.round(np.sqrt(np.square(diffv).mean()),3)
obs_growths = obs_growths[:, np.newaxis]
a, _, _, _ = np.linalg.lstsq(obs_growths, sim_growths, rcond=None)


eq = "y = " + str(np.round(a[0], 3)) + "x"
x = np.arange(200.0, 2000.0, 0.5)
# plt.text(5, 8.5, 'RMSE ' + str(rmse), fontsize=fs-1)

plt.plot(x, a * x, "k--", linewidth=2)
predicted = np.ravel(a[0] * obs_growths)

plt.text(2000, 2000, eq, fontsize=fs - 1)

ax3.text(
    1.3,
    0.95,
    "d",
    horizontalalignment="left",
    verticalalignment="top",
    fontsize=18,
    transform=ax2.transAxes,
    fontweight="bold",
)

# for tick in ax3.xaxis.get_major_ticks():
#   tick.label.set_fontsize(14)
# for tick in ax3.yaxis.get_major_ticks():
#    tick.label.set_fontsize(14)


# %%


for i, site_label in enumerate(SITES[:1]):
    ff = (
        r"C:/Users/laurenan/OneDrive - University of Helsinki/SUSI/vesitase/vesitase_out/"
        + site_label
        + ".nc"
    )
    site_params = assign_susi_params_to_site(site_label)
    sday = site_params.simulation_config.start_date
    end_date = site_params.simulation_config.end_date
    sfc = site_params.site_parameters.site_fertility_class
    ncf = Dataset(
        get_netcdf_path_from_site_label(
            site_label=SiteLabel(site_label), project_folderpath=PROJECT_FOLDER
        ),
        mode="r",
    )
    vol = np.array(ncf["stand"]["volume"][0, :, 1:-1])
    yrs, COLS = np.shape(vol)
    dfvol = pd.DataFrame(vol, columns=range(COLS))

    dwt = ncf["strip"]["dwtyr_growingseason"][0, :, 1:-1]
    yrs, COLS = np.shape(dwt)
    dfwt = pd.DataFrame(dwt, columns=range(COLS))

    peat_t = ncf["temperature"]["T"][0, :, 3]
    days = np.shape(peat_t)[0]
    dft = pd.DataFrame(peat_t, columns=["T"], index=pd.date_range(sday, periods=days))

    esom_co2 = (
        ncf["esom"]["Mass"]["co2"][0, :, 1:-1] / 10.0
    )  # kg ha-1 yr-1 ->g m-2 yr-1

    ncf.close()
    empirical = np.zeros(np.shape(dwt))
    for i, yr in enumerate(range(sday.year, end_date.year + 1)):
        t_peat_yr = np.ravel(dft.loc[str(yr)].values)
        rhet = ojanen_2010(sfc, vol[i + 1], t_peat_yr, dfwt.loc[i + 1].values * -100.0)
        empirical[i + 1, :] = rhet

dfesom = pd.DataFrame(data=esom_co2)
dfempirical = pd.DataFrame(data=empirical)
# %%

site_names = [WT_MEASUREMENT_INFO[site_label].name for site_label in SITES]
sfcs = [2, 2, 3, 3, 3, 3, 3, 3, 5, 5, 5]
print("***********************")
dfresid = pd.DataFrame(
    list(zip(site_names, sfcs, obsvols, prevols)), columns=["name", "sfc", "obs", "pre"]
)
dfresid["residual"] = dfresid["pre"] - dfresid["obs"]
print(dfresid.groupby(["sfc"]).mean())
print(dfresid)
