# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:04:53 2019

@author: lauren
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

sns.set()


def dwt_to_excel(gwl, outpara, scen):
    df = pd.DataFrame(gwl)
    f = outpara["outfolder"] + outpara["gwl_file"] + "_" + scen + ".xlsx"
    df.to_excel(f)
    print("DWT to ", f)


def runoff_to_excel(runoff, swe, outpara, scen):
    datadic = {"runoff": runoff, "swe": swe}
    df = pd.DataFrame(data=datadic)
    f = outpara["outfolder"] + outpara["runfile"] + "_" + scen + ".xlsx"
    df.to_excel(f)


def outfig(
    summer_dwt, co2_respi, growth_response, ditch_depth, relative_response, _rounds
):
    plt.figure(
        num="Susi drainage",
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(18.0, 10.0),
    )  # Figsize(w,h), tuple inches
    growth_response = np.ravel(growth_response)
    summer_dwt = np.array(summer_dwt)
    co2_respi = np.array(co2_respi)
    growth_response = np.array(growth_response)
    ditch_depth = np.array(ditch_depth)
    # growth_response= np.insert(growth_response, 0,0.0)
    plt.subplot(221)
    plt.plot(ditch_depth, -1 * summer_dwt, "bo-")
    plt.xlabel("ditch depth m")
    plt.ylabel("summer dwt m")
    plt.subplot(222)
    plt.plot(ditch_depth, co2_respi, "ko-")
    plt.xlabel("ditch depth m")
    plt.ylabel("heterotrophic respiration kg/ha CO2")
    plt.subplot(223)
    plt.plot(ditch_depth, growth_response, "go-")
    plt.xlabel("ditch depth m")
    plt.ylabel("growth response m3 ha-1")
    plt.subplot(224)
    plt.plot(ditch_depth, relative_response, "go-")
    plt.xlabel("ditch depth m")
    plt.ylabel("relative growth response %")

    # plt.plot()
    plt.show()


def print_site_description(spara):
    print("  + Site:")
    print("   + Number of columns:", spara.n)
    print("    - Site fertility class:")
    print("    ", spara.sfc)
    print("  + Stand:")
    # print ('    - vol:', np.round(spara['vol'],0),'m3/ha' )
    print("    - age:")
    print("        dominant:", spara.age["dominant"], "yrs")
    print("        subdominat:", spara.age["subdominant"], "yrs")
    print("        under:", spara.age["under"], "yrs")

    print("  + Soil: ")
    if spara.vonP:
        print("    - peat top:", spara.peat_type[0], ", von P:", spara.vonP_top[0])
    else:
        print(
            "    - peat top:",
            spara.peat_type[0],
            ", bulk density:",
            spara.bd_top[0],
        )
    print(
        "    - peat bottom:",
        spara.peat_type_bottom[0],
        ", von P:",
        spara.vonP_bottom,
    )
    print("  + Drainage:")
    print("    - Strip widht:", spara.L, "m")
    print("    - Ditch depth west:", spara.ditch_depth_west)
    print("    - Ditch depth east:", spara.ditch_depth_east)
    print("    - Scenarios:", spara.scenario_name)


def fig_stand_growth(
    r, rounds, ageSim, start_yr, end_yr, ageToVol, agearray, nppvol, nutvol, name
):
    sns.set()

    plt.figure(
        num="Growth and production",
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(20.0, 10.0),
    )
    gr_limit = 0.15  # allowed difference from table growth
    if rounds < 3:
        layout = "12"
    elif rounds < 5:
        layout = "22"
    else:
        layout = "33"

    fg_no = layout + str(r + 1)
    plt.subplot(fg_no)
    length = end_yr - start_yr + 1.0
    start = max(0.0, ageSim - 3.0)
    x = np.arange(start, ageSim + length, 0.1)
    y = ageToVol(x)
    gr_age = np.arange(ageSim, ageSim + length, 0.1)
    gr = np.gradient(ageToVol(gr_age))
    gr_up = ageToVol(gr_age) + np.cumsum(gr) * (1.0 + gr_limit)
    gr_low = ageToVol(gr_age) - np.cumsum(gr) * (1.0 - gr_limit)
    plt.plot(x, y)
    plt.plot(ageSim, ageToVol(ageSim), "ro")
    plt.plot(gr_age, gr_up, "g-")
    plt.plot(gr_age, gr_low, "g-")
    plt.fill_between(gr_age, gr_low, gr_up, color="gray", alpha=0.3)
    plt.plot(agearray, nppvol, "m-")
    plt.plot(agearray, nutvol, "r-")
    plt.title(name)
    plt.ylabel("Stand volume $m^3 ha^{-1}$")
    endvol = np.round(min(nppvol[-1], nutvol[-1]), 3)
    plt.text(ageSim, y[-1], str(endvol))
    plt.show()
    # import sys;sys.exit()


def fig_stand_growth_node_bck(
    r, rounds, ageSim, start_yr, end_yr, ageToVol, agearray, nppvol, nutvol, name
):
    sns.set()

    plt.figure(
        num="Growth and production",
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(20.0, 10.0),
    )
    gr_limit = 0.15  # allowed difference from table growth
    if rounds < 3:
        layout = "12"
    elif rounds < 5:
        layout = "22"
    else:
        layout = "33"

    fg_no = layout + str(r + 1)
    plt.subplot(fg_no)
    length = end_yr - start_yr + 1.0
    start = max(0.0, ageSim - 3.0)
    x = np.arange(start, ageSim + length, 0.1)
    y = ageToVol(x)
    gr_age = np.arange(ageSim, ageSim + length, 0.1)
    gr = np.gradient(ageToVol(gr_age))
    gr_up = ageToVol(gr_age) + np.cumsum(gr) * (1.0 + gr_limit)
    gr_low = ageToVol(gr_age) - np.cumsum(gr) * (1.0 - gr_limit)
    plt.plot(x, y)
    plt.plot(ageSim, ageToVol(ageSim), "ro")
    plt.plot(gr_age, gr_up, "g-")
    plt.plot(gr_age, gr_low, "g-")
    plt.fill_between(gr_age, gr_low, gr_up, color="gray", alpha=0.3)
    # nodes = np.shape(nppvol)[1]
    for n in nppvol.T:
        plt.plot(agearray, n, "m-")
    for n in nutvol.T:
        plt.plot(agearray, n, "r-")
    plt.title(name)
    plt.ylabel("Stand volume $m^3 ha^{-1}$")
    endvol = np.round(min(np.mean(nppvol[-1, :]), np.mean(nutvol[-1, :])), 3)
    plt.text(ageSim, y[-1], str(endvol))
    plt.show()


def fig_stand_growth_node(
    _rounds, ageSim, start_yr, end_yr, ageToVol, agearray, vols, name, dwts
):
    sns.set()
    # yrs, cols = np.shape(agearray)

    plt.figure(
        num="Susi-tulokset",
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(18.0, 9.0),
    )
    gr_limit = 0.15  # allowed difference from table growth

    plt.subplot(211)  # growth figure
    length = end_yr - start_yr + 1.0  #

    for column, agerange in enumerate(agearray.T):
        start = max(0.0, ageSim[column] - 3.0)
        x = np.arange(start, ageSim[column] + length, 0.1)
        y = ageToVol(x)
        gr_age = np.arange(ageSim[column], ageSim[column] + length + 0.1, 0.1)
        gr = np.gradient(ageToVol(gr_age))
        gr_up = ageToVol(gr_age) + np.cumsum(gr) * (1.0 + gr_limit)
        gr_low = ageToVol(gr_age) - np.cumsum(gr) * (1.0 - gr_limit)
        plt.plot(x, y, "k-")
        plt.plot(ageSim[column], ageToVol(ageSim[column]), "ro")
        plt.plot(gr_age, gr_up, "g-")
        plt.plot(gr_age, gr_low, "g-")
        plt.fill_between(gr_age, gr_low, gr_up, color="gray", alpha=0.3)

        colors = ["blue", "red", "green", "yellow", "cyan", "magenta"]
        rnds, _yrs, _nodes = np.shape(vols)
        for r in range(rnds):
            vtmp = vols[r, :, column]
            vtmp = np.insert(vtmp, 0, ageToVol(ageSim[column]))
            plt.plot(agerange, vtmp, color=colors[r], linewidth=3)
    # ---------------------------------------------------------

    plt.subplot(212)
    rnds, days, _nodes = np.shape(dwts)
    agedays = np.array(range(days))
    for r in range(rnds):
        dwttmp = dwts[r, :, 1:-1]
        up = np.max(dwttmp.T, axis=0)
        down = np.min(dwttmp.T, axis=0)
        mean = np.mean(dwttmp.T, axis=0)
        # plt.plot(age, up , color= colors[r], alpha=0.25)
        # plt.plot(age, down, color= colors[r], alpha=0.25)
        plt.plot(agedays, mean, color=colors[r], linewidth=1)
        plt.fill_between(agedays, up, down, color=colors[r], alpha=0.2, label=name[r])
    plt.legend(loc="lower left")
    plt.xlabel("Time, days", fontsize=18)
    plt.ylabel("Water level,m", fontsize=18)

    plt.show()


def fig_hydro(
    ele,
    hts,
    spara,
    wpara,
    _wlocation,
    ets,
    Prec,
    T,
    het,
    lai,
    h0_west,
    h0_east,
    runoff,
    scen,
):
    from matplotlib.lines import Line2D

    het = np.ravel(het)
    n = spara.n
    L = spara.L
    sim_yrs = len(het) / 365.0
    aa, _bb = np.shape(hts)
    x = np.linspace(0, L, n)
    float(L / n)
    fig = plt.figure(
        num="Striphy" + scen,
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(20.0, 11.0),
    )  # Figsize(w,h), tuple inches
    ax = fig.add_axes((0.05, 0.5, 0.55, 0.46))  # left, bottom, width, height
    low = min([ele[0] + h0_west, ele[n - 1] + h0_east]) * 0.4
    high = max(ele) * 1.2
    ax.set_ylim(bottom=low, top=high)
    ax.plot(x[1 : n - 1], ele[1 : n - 1], "k-", linewidth=2, label="Surface elevation")
    line1 = Line2D(
        [],
        [],
        color="blue",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        label="Water table",
    )
    limit = ele - 0.5
    limitGr = ele - 0.35
    plt.plot(x, ele, "k-", x, limit, "k--")
    mid = int(n / 2)
    west = int(n / 10)
    east = n - int(n / 10)
    plt.fill_between(x, limit, ele, color="green", alpha=0.3)
    plt.fill_between(x, limitGr, ele, color="red", alpha=0.3)
    plt.plot(x[mid], ele[mid], "ro", markersize=20)
    plt.plot(x[west], ele[west], "go", markersize=20)
    plt.plot(x[east], ele[east], "mo", markersize=20)
    plt.xlabel("Distance, m", fontsize=16)
    plt.ylabel("Elevation, m", fontsize=16)

    y = np.mean(hts, axis=0)
    yu = y + np.std(hts, axis=0)
    yl = y - np.std(hts, axis=0)
    yuu = np.max(hts, axis=0)
    yll = np.min(hts, axis=0)
    line1.set_data(x, y)
    plt.plot(x, yu, "b--")
    plt.plot(x, yl, "b--")
    plt.fill_between(x, yl, yu, color="blue", alpha=0.1)
    plt.plot(x, yuu, linestyle="dotted")
    plt.plot(x, yll, linestyle="dotted")
    plt.fill_between(x, yll, yuu, color="blue", alpha=0.1)

    t1 = "Ditch depth " + str(h0_west) + str(h0_east) + " m"
    ax.text(0.5, high * 0.95, str(t1), fontsize=16, color="0.25")
    t2 = "Slope " + str(spara.slope) + " %"
    ax.text(0.5, high * 0.9, str(t2), fontsize=16, color="0.25")
    t3 = "Peat type: " + spara.peat_type[0]
    ax.text(0.5, high * 0.85, t3, fontsize=16, color="0.25")
    if type(lai) is float:
        lai = [lai]
    t4 = "LAI " + str(lai)  # str(lai[0]) + '...' + str(lai[-1])
    ax.text(0.5, high * 0.8, t4, fontsize=16, color="0.25")

    ax.add_line(line1)
    ax.legend(loc=1)

    ax2 = fig.add_axes((0.05, 0.1, 0.55, 0.3))  # left, bottom, width, height)
    ax2.set_ylim(bottom=h0_west * 2.0, top=0.2)
    surf = np.zeros(aa)
    limit2 = -0.5 * np.ones(aa)
    limit3 = -0.35 * np.ones(aa)
    plt.plot(range(aa), limit2, "k--", range(aa), surf, "k-")
    plt.fill_between(range(aa), limit2, surf, color="green", alpha=0.3)
    plt.fill_between(range(aa), limit3, surf, color="red", alpha=0.3)
    range(np.shape(hts)[0])
    yy = hts[:, mid] - ele[mid]
    yywest = hts[:, west] - ele[west]
    yyeast = hts[:, east] - ele[east]
    line3 = Line2D(
        [],
        [],
        color="red",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        linewidth=1,
        label="Mid field wt",
    )
    ax2.add_line(line3)
    line4 = Line2D(
        [],
        [],
        color="green",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        linewidth=0.5,
        label="West wt",
    )
    ax2.add_line(line4)
    line5 = Line2D(
        [],
        [],
        color="m",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        linewidth=0.5,
        label="East wt",
    )
    ax2.add_line(line5)
    ax2.set_xlim(left=0, right=aa)
    plt.xlabel("Time, days", fontsize=16)
    plt.ylabel("wt depth, m", fontsize=16)
    ax2.legend(loc=1)
    line3.set_data(range(len(yy)), yy)
    line4.set_data(range(len(yywest)), yywest)
    line5.set_data(range(len(yyeast)), yyeast)

    ax22 = ax2.twinx()
    plt.plot(range(len(runoff)), runoff * 1000.0, color="blue")
    ax22.set_ylim(bottom=0.0, top=max(runoff) * 3.0 * 1000.0)
    plt.fill_between(range(len(runoff)), 0.0, runoff * 1000.0, color="blue", alpha=0.3)
    plt.ylabel("Runoff, mm $day^{-1}$")

    ax3 = fig.add_axes((0.68, 0.75, 0.3, 0.21))  # left, bottom, width, height
    ax3.set_ylim(bottom=0, top=sum(Prec) * 1.1)
    ax3.set_xlim(left=0, right=aa)
    x = range(len(runoff))
    z = np.zeros(len(runoff))
    y = np.cumsum(runoff) * 1000.0
    plt.plot(x, y, color="blue")
    plt.fill_between(x, z, y, color="blue", alpha=0.3, label="Cumul R")
    yy = y + np.cumsum(ets)
    plt.plot(x, yy, color="red")
    plt.fill_between(x, y, yy, color="red", alpha=0.3, label="Cumul ET")
    plt.plot(range(len(Prec)), np.cumsum(Prec), color="black", label="Cumul P")
    plt.xlabel("Time, days", fontsize=16)
    plt.ylabel("mm", fontsize=16)
    ax3.legend(loc=1)

    t1 = "Weather from " + wpara["description"]
    ax3.text(10, sum(Prec) * 1.0, str(t1), fontsize=14, color="0.25")
    t2 = "From " + str(wpara["start_yr"]) + " to " + str(wpara["end_yr"])
    ax3.text(10, sum(Prec) * 0.85, t2, fontsize=14, color="0.25")
    t3 = (
        "Rain "
        + str(np.round(sum(Prec) / sim_yrs))
        + ", ET "
        + str(np.round(sum(ets) / sim_yrs))
        + " mm yr-1"
    )
    ax3.text(10, sum(Prec) * 0.7, t3, fontsize=14, color="0.25")
    t4 = "Runoff " + str(np.round(sum(runoff) * 1000.0 / sim_yrs)) + " mm yr-1"
    ax3.text(10, sum(Prec) * 0.55, t4, fontsize=14, color="0.25")

    ax4 = fig.add_axes((0.68, 0.45, 0.3, 0.21))  # left, bottom, width, height
    ax4.set_ylim(bottom=-10, top=35)
    ax4.set_xlim(left=0, right=aa)
    line8 = Line2D(
        [],
        [],
        color="b",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        label="Air temperature, deg C",
    )
    ax4.add_line(line8)
    nolla = np.zeros(aa)
    lo = np.ones(aa) * -10
    plt.fill_between(range(aa), lo, nolla, color="yellow", alpha=0.2)
    line8.set_data(range(len(T)), T)
    plt.xlabel("Time, days", fontsize=16)
    plt.ylabel("deg C", fontsize=16)
    ax4.legend(loc=1)
    t1 = "Mean temperature " + str(np.round(np.mean(T)))
    ax4.text(10, 30, t1, fontsize=14, color="0.25")

    ax5 = fig.add_axes((0.68, 0.1, 0.3, 0.21))  # left, bottom, width, height
    line9 = Line2D(
        [],
        [],
        color="b",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        label="CO2 efflux",
    )
    ax5.set_ylim((-10, 100))
    ax5.set_xlim((0, aa))
    ax5.add_line(line9)
    ax5.legend(loc=1)
    plt.fill_between(range(aa), lo, nolla, color="yellow", alpha=0.2)
    line9.set_data(range(len(het)), het)
    plt.xlabel("Time, days", fontsize=16)
    plt.ylabel("kg/ha", fontsize=16)
    t1 = "Annual CO2 efflux " + str(np.round(np.sum(het) / sim_yrs))
    ax5.text(10, 85, t1, fontsize=14, color="0.25")
    plt.show()


def weather_fig(df):
    sns.set()
    # import string
    # printable = set(string.printable)
    fs = 12
    fig = plt.figure(
        num="Susi - weather data", figsize=[15.0, 8.0], facecolor="#C1ECEC"
    )  # see hex color codes from https://www.rapidtables.com/web/color/html-color-codes.html
    df["Kunta"][0]
    # fig.suptitle('Weather data, '+ filter(lambda x: x in printable, municipality), fontsize=18)
    ax1 = fig.add_axes((0.05, 0.55, 0.6, 0.35))  # left, bottom, width, height
    ax1.plot(df.index, df["Prec"].values, "b-", label="Rainfall")
    ax1.set_xlabel("Time", fontsize=fs)
    ax1.set_ylabel("Rainfall, mm", fontsize=12)
    ax1.legend(loc="upper left")
    ax11 = ax1.twinx()
    ax11.plot(
        df.index,
        np.cumsum(df["Prec"].values),
        "m-",
        linewidth=2.0,
        label="Cumulative rainfall",
    )
    ax11.set_ylabel("Cumulative rainfall [mm]", fontsize=fs)
    ax11.legend(loc="upper right")

    annual_prec = df["Prec"].resample("A").sum()
    ax2 = fig.add_axes((0.73, 0.55, 0.25, 0.35))

    t1 = "Mean annual rainfall " + str(np.round(np.mean(annual_prec.values))) + " mm"
    ax2.set_title(t1, fontsize=14)
    y_pos = np.arange((len(annual_prec)))
    plt.bar(y_pos, annual_prec.values, align="center", alpha=0.5)
    plt.xticks(y_pos, annual_prec.index.year, rotation=45)
    ax2.set_ylabel("mm")

    zeroline = np.zeros(len(df.index))
    ax3 = fig.add_axes((0.05, 0.08, 0.6, 0.35))
    ax3.plot(df.index, df["T"], "g", linewidth=0.5)
    ax3.plot(df.index, zeroline, "b-")
    ax3.fill_between(
        df.index, df["T"], 0, where=df["T"] < 0.0, facecolor="b", alpha=0.3
    )
    ax3.fill_between(
        df.index, df["T"], 0, where=df["T"] >= 0.0, facecolor="r", alpha=0.3
    )
    ax3.set_ylabel(r"Air temperature, $^\circ$ C", fontsize=fs)

    annual_temp = df["T"].resample("A").mean()
    t2 = (
        "Mean annual temperature "
        + str(np.round(np.mean(annual_temp.values), 2))
        + r"  $^\circ$ C"
    )

    ax4 = fig.add_axes((0.73, 0.08, 0.25, 0.35))
    ax4.set_title(t2, fontsize=14)
    y_pos = np.arange((len(annual_temp)))
    plt.bar(y_pos, annual_temp.values, align="center", alpha=0.5)
    plt.xticks(y_pos, annual_temp.index.year, rotation=45)
    ax4.set_ylabel(r" $^\circ$ C", fontsize=fs)
    plt.show()


def motti_fig(df, ini_age, sim_time):
    sns.set()

    fig = plt.figure(
        num="Susi, stand data", figsize=[15.0, 8.0], facecolor="#C1ECEC"
    )  # see hex color codes from https://www.rapidtables.com/web/color/html-color-codes.html
    fig.suptitle("Stand development (Motti simulation)", fontsize=18)

    ax = plt.subplot(231)
    ax.plot(df["age"], df["hdom"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Dominant height [m]")
    # plt.xlabel('Age, yrs')

    ax = plt.subplot(232)
    ax.plot(df["age"], df["N"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Stem count [$ha^{-1}$]")
    # plt.xlabel('Age, yrs')

    ax = plt.subplot(233)
    ax.plot(df["age"], df["BA"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Basal area [$m^{2}$ $ha^{-1}$]")
    # plt.xlabel('Age, yrs')

    ax = plt.subplot(234)
    ax.plot(df["age"], df["vol"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Tilavuus [$m^{3}$ $ha^{-1}$]")
    # plt.xlabel('Age, yrs')

    ax = plt.subplot(235)
    ax.plot(df["age"], df["Dg"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Mean diameter [cm]")
    plt.xlabel("Age, years")

    sla = 6.8
    ax = plt.subplot(236)
    ax.plot(df["age"], df["leaves"] / 10.0 * sla, color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Leaf area index [m2 $m^{-2}$]")
    plt.xlabel("Age, years")
    plt.show()


def print_scenario(r, co2release, deltas, h0ts, dwts, bmToYi, npps, bm, yi, ets):
    print(
        "  +Heterotrophic respiration", np.round(sum(co2release[r, :]), 2), "kg CO2/ha"
    )
    gwlev = [np.mean((dwts[r, d, :])[1:-1]) for d in range(len(deltas))]
    print("Scenario summary, ditch depth:", h0ts[0], "m")
    print("  +Mean dwt", np.round(np.mean(gwlev), 3), "m")
    print("  +Table growth", yi[-1] - yi[0])
    print(
        "  +Assimilation growth",
        np.round(bmToYi(bm[0] + sum(npps[r, :])) - bmToYi(bm[0]), 3),
    )
    print("  +Evapotranspiration", np.round(sum(ets) * 1000.0))
    print("  +Deltas ", np.round(sum(deltas) * 1000.0, 2))


def print_scenario_nodes(
    r, deltas, ets, h0ts_west, h0ts_east, dwts, g_nuts
):
    print(
        "Scenario summary, ditch depth:", h0ts_west[0], h0ts_east[0], "m", "round ", r
    )
    # print ('  +C balance',  'kg CO2/ha?',np.round(np.sum(c_bals[r,:]), 2))
    gwlev = [np.mean((dwts[r, d, :])[1:-1]) for d in range(len(deltas))]
    print("  +Mean dwt", np.round(np.mean(gwlev), 3), "m")
    print("  +Evapotranspiration", np.round(sum(ets) * 1000.0), "mm")
    print("  +Deltas ", np.round(sum(deltas) * 1000.0, 2), "mm")
    # print ('  +Table growth', np.round(yi[-1]-yi[0],2), 'm3')
    print(
        "  +Nutrient limited volume",
        np.round(np.mean(g_nuts[r, -1], axis=0), 3),
        np.round(np.std(g_nuts[r, -1], axis=0), 3),
        "(sd)",
    )
