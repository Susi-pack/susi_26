import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import susi.io.netcdf_utils as nc_utils


def _create_profile_line(
    ax,
    wt,
    wtmin,
    sd,
    cols,
    ylabel,
    label,
    fs,
    facecolor,
    colorin,
    hidex=False,
    hidey=False,
    elevation=None,
):
    if elevation is not None:
        ax.plot(elevation, color="brown", label="soil surface")
    ax.plot(wt, color=colorin, label=label)
    ax.fill_between(range(cols), wt + sd * 2, wt - sd, color=colorin, alpha=0.075)
    if elevation is not None:
        drainnorm = elevation - 0.35
    else:
        drainnorm = -0.35
    ax.hlines(y=drainnorm, xmin=0, xmax=cols, color="red", linestyles="--")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    if elevation is None:
        ax.set_ylim([wtmin, 0])
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.legend()
    ax.grid(visible=False)
    ax.set_facecolor(facecolor)
    if hidex:
        ax.get_xaxis().set_visible(False)
    else:
        ax.tick_params(axis="x", labelsize=fs)
    if hidey:
        ax.get_yaxis().set_visible(False)
    else:
        ax.get_yaxis().set_visible(True)

    return ax


def _create_profile_boxplot(
    ax,
    datain,
    cols,
    colorin,
    title,
    label,
    fs,
    facecolor,
    zero=False,
    hidex=True,
    hidey=False,
):
    df = pd.DataFrame(data=datain, columns=list(range(cols)))
    df.boxplot(
        ax=ax,
        color=dict(boxes=colorin, whiskers=colorin, medians=colorin, caps=colorin),
        boxprops=dict(linestyle="-", linewidth=1.5, color=colorin, alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color=colorin, alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color=colorin, alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    if zero:
        ax.hlines(y=0, xmin=0, xmax=cols, color="red", linestyles="--")
    meanval = df.mean(axis=0)
    meanval = np.round(meanval.mean(), 2)
    title = title + ": mean " + str(meanval)
    ax.set_title(title, fontsize=fs)
    ax.set_ylabel(label, fontsize=fs)
    ax.set_facecolor(facecolor)
    # ax.get_xaxis().set_visible(False)
    if hidex:
        ax.get_xaxis().set_visible(False)
    else:
        ax.tick_params(axis="x", labelsize=fs)
    if hidey:
        ax.get_yaxis().set_visible(False)
    else:
        ax.get_yaxis().set_visible(True)
        ax.set_ylabel(label, fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)

    return ax


def spatial_bars(data: np.ndarray) -> matplotlib.figure.Figure:
    """
    data is a 2D np.ndarray with spatial and temporal values.
    No scenarios
    """
    n_time, n_space = data.shape

    fig, ax = plt.subplots(figsize=(8, 5))

    # Grouped bars - each time step has bars for each space point
    x = np.arange(n_time)
    width = 0.8 / n_space

    for space_idx in range(n_space):
        offset = (space_idx - n_space / 2 + 0.5) * width
        ax.bar(
            x + offset,
            data[:, space_idx],
            width,
            label=f"Space {space_idx}" if n_space <= 10 else None,
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Temporal Evolution")
    if n_space <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    return fig


def temporal_stats(data: np.ndarray) -> matplotlib.figure.Figure:
    n_time, n_space = data.shape
    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate over space
    mean_over_space = np.mean(data, axis=1)
    std_over_space = np.std(data, axis=1)

    x_time = np.arange(n_time)
    ax.plot(x_time, mean_over_space, "b-", label="Mean", linewidth=2)
    ax.fill_between(
        x_time,
        mean_over_space - std_over_space,
        mean_over_space + std_over_space,
        alpha=0.3,
        color="blue",
        label="±1 Std",
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Spatial Mean and Variability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# %% Annamari plots


def _get_var(vars: list[nc_utils.NetcdfVariableValue], path: str) -> np.ndarray:
    return next(v for v in vars if v.path == path).value


def stand(
    vars: list[nc_utils.NetcdfVariableValue], scen: int = 0
) -> matplotlib.figure.Figure:
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(ncols=12, nrows=12, figure=fig, wspace=0.25, hspace=0.25)

    wt = np.mean(_get_var(vars, "/strip/dwtyr")[scen, :, :], axis=0)
    cols = np.shape(wt)[0]
    wtls = np.mean(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    sdls = np.std(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    wtmin = min(wtls) - 0.2

    ax = fig.add_subplot(gs[10:, :4])
    ax.plot(wtls, color="orange", label="late summer")
    ax.fill_between(
        range(cols), wtls + sdls * 2, wtls - sdls * 2, color="orange", alpha=0.075
    )
    ax.hlines(y=-0.35, xmin=0, xmax=cols, color="red", linestyles="--")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_ylim([wtmin, 0])
    ax.set_ylabel("WT m", fontsize=fs)
    ax.legend()
    ax.grid(visible=False)
    ax.set_facecolor(facecolor)

    vol = _get_var(vars, "/stand/volume")[scen, :, :]
    growth = np.diff(vol, axis=0)
    dfgrowth = pd.DataFrame(data=growth, columns=list(range(cols)))
    axgrowth = fig.add_subplot(gs[8:10, :4])
    dfgrowth.boxplot(
        ax=axgrowth,
        color=dict(boxes="blue", whiskers="blue", medians="blue", caps="blue"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    axgrowth.set_title("Stand growth")
    axgrowth.set_ylabel("$m^3 ha^{-1} yr^{-1}$", fontsize=fs)

    axgrowth.get_xaxis().set_visible(False)
    axgrowth.tick_params(axis="y", labelsize=fs)
    axgrowth.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[10:, 4:8])
    totvol = vol[-1, :]
    domvol = _get_var(vars, "/stand/dominant/volume")[scen, -1, :]
    subdomvol = _get_var(vars, "/stand/subdominant/volume")[scen, -1, :]
    undervol = _get_var(vars, "/stand/under/volume")[scen, -1, :]
    df = pd.DataFrame(
        {
            "total": totvol,
            "dominant": domvol,
            "subdominant": subdomvol,
            "under": undervol,
        },
        index=range(cols),
    )
    df.plot(kind="bar", width=1.1, ax=ax)
    ax.set_title("Stand volume")

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)
    ax.legend(loc="upper right", fontsize=7)

    ax = fig.add_subplot(gs[10:, 8:])
    totvol = vol[:, :]
    domvol = _get_var(vars, "/stand/dominant/volume")[scen, :, :]
    subdomvol = _get_var(vars, "/stand/subdominant/volume")[scen, :, :]
    undervol = _get_var(vars, "/stand/under/volume")[scen, :, :]
    for c in range(cols):
        ax.plot(totvol[:, c], alpha=0.2)
    for c in range(cols):
        ax.plot(domvol[:, c], alpha=0.2)
    for c in range(cols):
        ax.plot(subdomvol[:, c], alpha=0.2)
    for c in range(cols):
        ax.plot(undervol[:, c], alpha=0.2)

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Stand volume increment")

    ax = fig.add_subplot(gs[8:10, 4:8])
    vol_data = _get_var(vars, "/stand/volume")[scen, 0:, :]
    for c in range(cols):
        ax.plot(np.diff(vol_data[:, c]), alpha=0.2)

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Volume")
    ax.get_xaxis().set_visible(False)

    ax = fig.add_subplot(gs[8:10, 8:])
    logvol = _get_var(vars, "/stand/logvolume")[scen, :, :]
    pulpvol = _get_var(vars, "/stand/pulpvolume")[scen, :, :]
    for c in range(cols):
        yrs = len(logvol[:, c])
        ax.plot(range(1, yrs), logvol[1:, c], alpha=0.2)
    for c in range(cols):
        yrs = len(pulpvol[:, c])
        ax.plot(range(1, yrs), pulpvol[1:, c], alpha=0.2)

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Log and pulp volume")
    ax.get_xaxis().set_visible(False)

    lmass = _get_var(vars, "/stand/leafmass")[scen, :, :]
    df = pd.DataFrame(data=lmass, columns=list(range(cols)))
    ax = fig.add_subplot(gs[6:8, :4])
    df.boxplot(
        ax=ax,
        color=dict(boxes="green", whiskers="green", medians="green", caps="green"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Leaf mass")
    ax.set_ylabel("$kg \ ha^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[6:8, 4:8])
    for c in range(cols):
        ax.plot(lmass[:, c], alpha=0.2)

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Leaf mass")
    ax.get_xaxis().set_visible(False)

    ax = fig.add_subplot(gs[6:8, 8:])
    dlmass = _get_var(vars, "/stand/dominant/leafmass")[scen, :, :]
    upperlim = _get_var(vars, "/stand/dominant/leafmax")[scen, :, :]
    lowerlim = _get_var(vars, "/stand/dominant/leafmin")[scen, :, :]
    for c in range(cols):
        yrs = len(dlmass[:, c])
        ax.fill_between(
            range(1, yrs), upperlim[1:, c], lowerlim[1:, c], color="green", alpha=0.01
        )
        ax.plot(dlmass[:, c], alpha=0.5, color="green")
    ax.plot(dlmass[:, c], alpha=0.5, color="green", label="dominant")

    dlmass = _get_var(vars, "/stand/subdominant/leafmass")[scen, :, :]
    upperlim = _get_var(vars, "/stand/subdominant/leafmax")[scen, :, :]
    lowerlim = _get_var(vars, "/stand/subdominant/leafmin")[scen, :, :]
    for c in range(cols):
        yrs = len(dlmass[:, c])
        ax.fill_between(
            range(1, yrs), upperlim[1:, c], lowerlim[1:, c], color="cyan", alpha=0.01
        )
        ax.plot(dlmass[:, c], alpha=0.5, color="cyan")
    ax.plot(dlmass[:, c], alpha=0.5, color="cyan", label="subdominant")

    dlmass = _get_var(vars, "/stand/under/leafmass")[scen, :, :]
    upperlim = _get_var(vars, "/stand/under/leafmax")[scen, :, :]
    lowerlim = _get_var(vars, "/stand/under/leafmin")[scen, :, :]
    for c in range(cols):
        yrs = len(dlmass[:, c])
        ax.fill_between(
            range(1, yrs), upperlim[1:, c], lowerlim[1:, c], color="orange", alpha=0.01
        )
        ax.plot(dlmass[:, c], alpha=0.5, color="orange")
    ax.plot(dlmass[:, c], alpha=0.5, color="orange", label="under")

    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Leaf mass in canopy layers")
    ax.get_xaxis().set_visible(False)

    ns = _get_var(vars, "/stand/nut_stat")[scen, :, :]
    ax = fig.add_subplot(gs[4:6, :4])

    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(ns[:, c], alpha=0.5, color="orange")
    ax.plot(ns[:, c], alpha=0.5, color="orange", label="nutrient status")

    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.get_xaxis().set_visible(False)

    ax = fig.add_subplot(gs[4:6, 4:8])
    dom_phys_r = (
        _get_var(vars, "/stand/dominant/NPP")[scen, :, :]
        / _get_var(vars, "/stand/dominant/NPP_pot")[scen, :, :]
    )
    df = pd.DataFrame(data=dom_phys_r, columns=list(range(cols)))
    df.boxplot(
        ax=ax,
        color=dict(boxes="blue", whiskers="blue", medians="blue", caps="blue"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Physical restrictions")

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[4:6, 8:])

    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(ns[:-1, c], growth[:, c], "go", alpha=0.5)

    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_ylabel("volume growth")
    ax.set_xlabel("nutrient status")

    ndemand = _get_var(vars, "/stand/n_demand")[scen, :, :]
    df = pd.DataFrame(data=ndemand, columns=list(range(cols)))
    ax = fig.add_subplot(gs[2:4, :4])
    df.boxplot(
        ax=ax,
        color=dict(boxes="blue", whiskers="blue", medians="blue", caps="blue"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("N demand")
    ax.set_ylabel("$kg \ ha^{-1} \ yr^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    pdemand = _get_var(vars, "/stand/p_demand")[scen, :, :]
    df = pd.DataFrame(data=pdemand, columns=list(range(cols)))
    ax = fig.add_subplot(gs[2:4, 4:8])
    df.boxplot(
        ax=ax,
        color=dict(boxes="green", whiskers="green", medians="green", caps="green"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("P demand")

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    kdemand = _get_var(vars, "/stand/k_demand")[scen, :, :]
    df = pd.DataFrame(data=kdemand, columns=list(range(cols)))
    ax = fig.add_subplot(gs[2:4, 8:])
    df.boxplot(
        ax=ax,
        color=dict(boxes="orange", whiskers="orange", medians="orange", caps="orange"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("K demand")

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[:2, :4])

    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(range(1, yrs), ndemand[1:, c], alpha=0.5, color="blue")
    ax.plot(range(1, yrs), ndemand[1:, c], alpha=0.5, color="blue", label="N demand")
    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_ylabel("kg $ha^{-1} yr^{-1}$", fontsize=fs)

    ax = fig.add_subplot(gs[:2, 4:8])

    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(range(1, yrs), pdemand[1:, c], alpha=0.5, color="green")
    ax.plot(range(1, yrs), pdemand[1:, c], alpha=0.5, color="green", label="P demand")
    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax = fig.add_subplot(gs[:2, 8:])

    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(range(1, yrs), kdemand[1:, c], alpha=0.5, color="orange")
    ax.plot(range(1, yrs), kdemand[1:, c], alpha=0.5, color="orange", label="K demand")
    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)

    return fig


def hydrology(
    vars: list[nc_utils.NetcdfVariableValue], scen: int = 0
) -> matplotlib.figure.Figure:
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(num="hydro", figsize=(15, 18))
    gs = gridspec.GridSpec(ncols=12, nrows=12, figure=fig, wspace=0.25, hspace=0.25)

    wt = np.mean(_get_var(vars, "/strip/dwtyr")[scen, :, :], axis=0)
    cols = np.shape(wt)[0]
    sd = np.std(_get_var(vars, "/strip/dwtyr")[scen, :, :], axis=0)
    wtgs = np.mean(_get_var(vars, "/strip/dwtyr_growingseason")[scen, :, :], axis=0)
    sdgs = np.std(_get_var(vars, "/strip/dwtyr_growingseason")[scen, :, :], axis=0)
    wtls = np.mean(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    sdls = np.std(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    wtmin = np.min(wtls) - 0.2

    ax = fig.add_subplot(gs[10:, :4])
    ax = _create_profile_line(
        ax,
        wt,
        wtmin,
        sd,
        cols,
        "WT m",
        "annual",
        fs,
        facecolor,
        "blue",
        hidex=True,
        hidey=False,
    )

    ax = fig.add_subplot(gs[10:, 4:8])
    ax = _create_profile_line(
        ax,
        wtgs,
        wtmin,
        sdgs,
        cols,
        None,
        "growing season",
        fs,
        facecolor,
        "green",
        hidex=True,
        hidey=True,
    )

    ax = fig.add_subplot(gs[10:, 8:])
    ax = _create_profile_line(
        ax,
        wtls,
        wtmin,
        sdls,
        cols,
        None,
        "late summer",
        fs,
        facecolor,
        "orange",
        hidex=True,
        hidey=True,
    )

    wt = np.mean(_get_var(vars, "/strip/dwt")[scen, :, :], axis=1)
    days = np.shape(wt)[0]
    sd = np.std(_get_var(vars, "/strip/dwt")[scen, :, :], axis=1)

    axwtts = fig.add_subplot(gs[8:10, :])
    axwtts.plot(wt, color="green", label="WT")
    axwtts.hlines(y=-0.35, xmin=0, xmax=days, color="red", linestyles="--")
    for c in range(1, cols - 1):
        axwtts.plot(range(days), _get_var(vars, "/strip/dwt")[scen, :, c], alpha=0.2)

    axwtts.tick_params(axis="y", labelsize=fs)
    axwtts.set_ylim(bottom=wtmin, top=0)
    axwtts.set_ylabel("WT m", fontsize=fs)
    axwtts.legend(loc="upper left")
    axwtts.grid(visible=False)
    axwtts.set_facecolor(facecolor)

    runoff = np.cumsum(_get_var(vars, "/strip/roff")[scen, :])
    ulimruno = max(runoff) * 1.1 * 1000.0
    axruno = fig.add_subplot(gs[7, :])
    axruno.plot(range(len(runoff)), runoff * 1000.0, color="blue", label="total runoff")
    axruno.set_ylim(bottom=0.0, top=ulimruno)
    axruno.fill_between(
        range(len(runoff)), 0.0, runoff * 1000.0, color="blue", alpha=0.3
    )
    axruno.grid(visible=False)
    axruno.set_ylabel("mm", fontsize=fs)
    axruno.tick_params(axis="y", labelsize=fs)
    axruno.legend(loc="upper left")
    axruno.set_facecolor(facecolor)
    axruno.get_xaxis().set_visible(False)

    runoff = np.cumsum(_get_var(vars, "/strip/roffwest")[scen, :])
    axruno = fig.add_subplot(gs[6, :])
    axruno.plot(range(len(runoff)), runoff * 1000.0, color="green", label="west runoff")
    axruno.set_ylim(bottom=0.0, top=ulimruno)
    axruno.fill_between(
        range(len(runoff)), 0.0, runoff * 1000.0, color="green", alpha=0.3
    )
    axruno.grid(visible=False)
    axruno.set_ylabel("mm", fontsize=fs)
    axruno.tick_params(axis="y", labelsize=fs)
    axruno.legend(loc="upper left")
    axruno.set_facecolor(facecolor)
    axruno.get_xaxis().set_visible(False)

    runoff = np.cumsum(_get_var(vars, "/strip/roffeast")[scen, :])
    axruno = fig.add_subplot(gs[5, :])
    axruno.plot(range(len(runoff)), runoff * 1000.0, color="red", label="east runoff")
    axruno.set_ylim(bottom=0.0, top=ulimruno)
    axruno.fill_between(
        range(len(runoff)), 0.0, runoff * 1000.0, color="red", alpha=0.3
    )
    axruno.grid(visible=False)
    axruno.set_ylabel("mm", fontsize=fs)
    axruno.tick_params(axis="y", labelsize=fs)
    axruno.legend(loc="upper left")
    axruno.set_facecolor(facecolor)
    axruno.get_xaxis().set_visible(False)

    runoff = np.cumsum(
        np.mean(_get_var(vars, "/strip/surfacerunoff")[scen, :, :], axis=1)
    )
    axruno = fig.add_subplot(gs[4, :])
    axruno.plot(
        range(len(runoff)), runoff * 1000.0, color="orange", label="surface runoff"
    )
    axruno.set_ylim(bottom=0.0, top=ulimruno)
    axruno.fill_between(
        range(len(runoff)), 0.0, runoff * 1000.0, color="orange", alpha=0.3
    )
    axruno.grid(visible=False)
    axruno.set_ylabel("mm ", fontsize=fs)
    axruno.tick_params(axis="y", labelsize=fs)
    axruno.legend(loc="upper left")
    axruno.set_facecolor(facecolor)
    axruno.get_xaxis().set_visible(False)

    deltas = _get_var(vars, "/strip/deltas")[scen, 1:, :] * 1000.0
    dfdeltas = pd.DataFrame(data=deltas, columns=list(range(cols)))

    ax = fig.add_subplot(gs[2:4, :4])
    ax = _create_profile_boxplot(
        ax,
        dfdeltas,
        cols,
        "blue",
        "Through soil surface",
        "Water flux mm $yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    ET = _get_var(vars, "/cpy/ET_yr")[scen, 1:, :] * 1000.0
    dfET = pd.DataFrame(data=ET, columns=list(range(cols)))

    ax = fig.add_subplot(gs[2:4, 4:8])
    ax = _create_profile_boxplot(
        ax, dfET, cols, "green", "ET", "", fs, facecolor, zero=False, hidex=True
    )

    transpi = _get_var(vars, "/cpy/transpi_yr")[scen, 1:, :] * 1000.0
    dftranspi = pd.DataFrame(data=transpi, columns=list(range(cols)))
    ax = fig.add_subplot(gs[2:4, 8:])
    ax = _create_profile_boxplot(
        ax,
        dftranspi,
        cols,
        "orange",
        "Transpiration",
        "",
        fs,
        facecolor,
        zero=False,
        hidex=True,
    )

    efloor = _get_var(vars, "/cpy/efloor_yr")[scen, 1:, :] * 1000.0
    dfefloor = pd.DataFrame(data=efloor, columns=list(range(cols)))

    ax = fig.add_subplot(gs[:2, :4])
    ax = _create_profile_boxplot(
        ax,
        dfefloor,
        cols,
        "blue",
        "Soil evaporation",
        "Water flux mm $yr^{-1}$",
        fs,
        facecolor,
        zero=False,
        hidex=True,
    )

    swe = _get_var(vars, "/cpy/SWEmax")[scen, 1:, :]
    dfswe = pd.DataFrame(data=swe, columns=list(range(cols)))

    ax = fig.add_subplot(gs[:2, 4:8])
    ax = _create_profile_boxplot(
        ax,
        dfswe,
        cols,
        "green",
        "Max snow water equivalent",
        "",
        fs,
        facecolor,
        zero=False,
        hidex=True,
    )

    interc = _get_var(vars, "/cpy/interc_yr")[scen, 1:, :] * 1000.0
    dfinterc = pd.DataFrame(data=interc, columns=list(range(cols)))

    ax = fig.add_subplot(gs[:2, 8:])
    ax = _create_profile_boxplot(
        ax,
        dfinterc,
        cols,
        "orange",
        "Mean interception storage",
        "",
        fs,
        facecolor,
        zero=False,
        hidex=True,
    )

    return fig


def mass(
    vars: list[nc_utils.NetcdfVariableValue], scen: int = 0
) -> matplotlib.figure.Figure:
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(ncols=12, nrows=12, figure=fig, wspace=0.25, hspace=0.25)

    wt = np.mean(_get_var(vars, "/strip/dwtyr")[scen, :, :], axis=0)
    cols = np.shape(wt)[0]
    wtls = np.mean(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    sdls = np.std(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    wtmin = min(wtls) - 0.2

    ax = fig.add_subplot(gs[10:, :4])
    ax.plot(wtls, color="orange", label="late summer")
    ax.fill_between(
        range(cols), wtls + sdls * 2, wtls - sdls * 2, color="orange", alpha=0.075
    )
    ax.hlines(y=-0.35, xmin=0, xmax=cols, color="red", linestyles="--")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_ylim(bottom=wtmin, top=0)
    ax.set_ylabel("WT m", fontsize=fs)
    ax.legend()
    ax.grid(visible=False)
    ax.set_facecolor(facecolor)

    litter = (
        _get_var(vars, "/groundvegetation/ds_litterfall")[scen, :, :] / 10000.0
        + _get_var(vars, "/groundvegetation/h_litterfall")[scen, :, :] / 10000.0
        + _get_var(vars, "/groundvegetation/s_litterfall")[scen, :, :] / 10000.0
        + _get_var(vars, "/stand/nonwoodylitter")[scen, :, :] / 10000.0
        + _get_var(vars, "/stand/woodylitter")[scen, :, :] / 10000.0
    )

    soil = _get_var(vars, "/esom/Mass/out")[scen, :, :] / 10000.0 * -1 + litter

    soilout = _get_var(vars, "/esom/Mass/out")[scen, :, :] / 10000.0 * -1

    df = pd.DataFrame(data=soil, columns=list(range(cols)))
    ax = fig.add_subplot(gs[8:10, :4])
    df.boxplot(
        ax=ax,
        color=dict(boxes="blue", whiskers="blue", medians="blue", caps="blue"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Soil mass change")
    ax.set_ylabel("$kg m^{-2} yr^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[10:, 4:8])
    esoms = ["L0L", "L0W", "LL", "LW", "FL", "FW", "H", "P1", "P2", "P3"]
    inipeat = np.zeros(cols)
    for sto in esoms[7:]:
        inipeat += _get_var(vars, f"/esom/Mass/{sto}")[scen, 0, :] / 10000.0
    endpeat = np.zeros(cols)
    for sto in esoms[7:]:
        endpeat += _get_var(vars, f"/esom/Mass/{sto}")[scen, -1, :] / 10000.0
    inimor = np.zeros(cols)
    for sto in esoms[:7]:
        inimor += _get_var(vars, f"/esom/Mass/{sto}")[scen, 0, :] / 10000.0
    endmor = np.zeros(cols)
    for sto in esoms[:7]:
        endmor += _get_var(vars, f"/esom/Mass/{sto}")[scen, -1, :] / 10000.0

    maxval = np.max(np.vstack((inipeat + inimor, endpeat + endmor)))
    minval = np.min(np.vstack((inipeat + inimor, endpeat + endmor)))

    df = pd.DataFrame(
        {
            "peat initial": inipeat,
            "mor initial": inimor,
            "peat end": endpeat,
            "mor end": endmor,
        },
        index=range(cols),
    )
    df[["peat initial", "mor initial"]].plot.bar(
        stacked=True,
        position=-0.35,
        width=0.25,
        ax=ax,
        colormap="copper",
        edgecolor="k",
        alpha=0.6,
    )
    df[["peat end", "mor end"]].plot.bar(
        stacked=True,
        position=-1.95,
        width=0.25,
        ax=ax,
        colormap="copper",
        edgecolor="k",
        alpha=0.6,
    )

    ax.set_title("Organic soil mass, kg $m^{-2}$")
    ax.set_ylim(bottom=minval * 0.95, top=maxval * 1.025)
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)
    ax.legend(loc="lower right", fontsize=7)

    df = pd.DataFrame(data=litter, columns=list(range(cols)))
    ax = fig.add_subplot(gs[6:8, :4])
    df.boxplot(
        ax=ax,
        color=dict(boxes="orange", whiskers="orange", medians="orange", caps="orange"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Litter input")
    ax.set_ylabel("$kg m^{-2} yr^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    gv = _get_var(vars, "/groundvegetation/gv_tot")[scen, :, :] / 10000.0
    grgv = np.diff(gv, axis=0)
    df = pd.DataFrame(data=grgv, columns=list(range(cols)))
    ax = fig.add_subplot(gs[4:6, :4])
    df.boxplot(
        ax=ax,
        color=dict(boxes="orange", whiskers="orange", medians="orange", caps="orange"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Ground vegetation biomass change")
    ax.set_ylabel("$kg m^{-2} yr^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    stand = _get_var(vars, "/stand/biomass")[scen, :, :] / 10000.0
    gr = np.diff(stand, axis=0)
    df = pd.DataFrame(data=gr, columns=list(range(cols)))
    ax = fig.add_subplot(gs[2:4, :4])
    df.boxplot(
        ax=ax,
        color=dict(boxes="green", whiskers="green", medians="green", caps="green"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Stand biomass change")
    ax.set_ylabel("$kg m^{-2} yr^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    site = gr + grgv + soilout[1:, :] + litter[1:, :]
    df = pd.DataFrame(data=site, columns=list(range(cols)))
    ax = fig.add_subplot(gs[:2, :4])
    df.boxplot(
        ax=ax,
        color=dict(boxes="green", whiskers="green", medians="green", caps="green"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="green", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Site mass change")
    ax.set_ylabel("$kg m^{-2} yr^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    leaflitter = (
        _get_var(vars, "/stand/nonwoodylitter")[scen, :, :]
        - _get_var(vars, "/stand/finerootlitter")[scen, :, :]
    )

    df = pd.DataFrame(data=leaflitter, columns=list(range(cols)))
    ax = fig.add_subplot(gs[:2, 4:8])
    df.boxplot(
        ax=ax,
        color=dict(boxes="blue", whiskers="blue", medians="blue", caps="blue"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="blue", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Leaf litter")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    finerootlitter = _get_var(vars, "/stand/finerootlitter")[scen, :, :]

    df = pd.DataFrame(data=finerootlitter, columns=list(range(cols)))
    ax = fig.add_subplot(gs[2:4, 4:8])
    df.boxplot(
        ax=ax,
        color=dict(boxes="orange", whiskers="orange", medians="orange", caps="orange"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="orange", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("fineroot litter")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    woodylitter = _get_var(vars, "/stand/woodylitter")[scen, :, :]

    df = pd.DataFrame(data=woodylitter, columns=list(range(cols)))
    ax = fig.add_subplot(gs[4:6, 4:8])
    df.boxplot(
        ax=ax,
        color=dict(boxes="brown", whiskers="brown", medians="brown", caps="brown"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="brown", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="brown", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="brown", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Woody litter")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    gvlitter = (
        _get_var(vars, "/groundvegetation/ds_litterfall")[scen, :, :]
        + _get_var(vars, "/groundvegetation/h_litterfall")[scen, :, :]
        + _get_var(vars, "/groundvegetation/s_litterfall")[scen, :, :]
    )

    df = pd.DataFrame(data=gvlitter, columns=list(range(cols)))
    ax = fig.add_subplot(gs[6:8, 4:8])
    df.boxplot(
        ax=ax,
        color=dict(boxes="red", whiskers="red", medians="red", caps="red"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="red", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="red", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="red", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Ground vegetation litter")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    out = _get_var(vars, "/esom/Mass/out")[scen, :, :] / 10000.0 * -1

    df = pd.DataFrame(data=out, columns=list(range(cols)))
    ax = fig.add_subplot(gs[8:10, 4:8])
    df.boxplot(
        ax=ax,
        color=dict(boxes="grey", whiskers="grey", medians="grey", caps="grey"),
        boxprops=dict(linestyle="-", linewidth=1.5, color="grey", alpha=0.6),
        flierprops=dict(linestyle="-", linewidth=1.5),
        medianprops=dict(linestyle="-", linewidth=1.5),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="grey", alpha=0.6),
        capprops=dict(linestyle="-", linewidth=1.5, color="grey", alpha=0.6),
        showfliers=False,
        grid=False,
        rot=0,
    )

    ax.set_title("Soil mass loss")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[10:, 8:])
    esoms = ["L0L", "L0W", "LL", "LW", "FL", "FW", "H", "P1", "P2", "P3"]
    yrs = np.shape(_get_var(vars, "/esom/Mass/L0L")[scen, :, :])[0]
    for c in range(cols):
        peat = np.zeros(yrs)
        for sto in esoms[7:]:
            peat += _get_var(vars, f"/esom/Mass/{sto}")[scen, :, c] / 10000.0
        mor = np.zeros(yrs)
        for sto in esoms[:7]:
            mor += _get_var(vars, f"/esom/Mass/{sto}")[scen, :, c] / 10000.0

        ax.plot(np.diff(peat), color="brown")
        ax.plot(np.diff(mor), color="orange")
    ax.plot(np.diff(peat), color="brown", label="peat")
    ax.plot(np.diff(mor), color="orange", label="mor")
    ax.legend()
    ax.set_title("Organic soil mass change, kg $m^{-2} yr^{-1}$")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)
    ax.legend(loc="lower right", fontsize=7)

    ax = fig.add_subplot(gs[8:10, 8:])
    for c in range(cols):
        ax.plot(out[1:, c], color="orange")

    ax.set_title("Soil mass loss")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    out_with_litter = (
        _get_var(vars, "/esom/Mass/out")[scen, :, :] / 10000.0 * -1 + litter
    )

    ax = fig.add_subplot(gs[6:8, 8:])
    for c in range(cols):
        ax.plot(out_with_litter[1:, c], color="red")

    ax.set_title("Soil mass change")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[4:6, 8:])
    for c in range(cols):
        ax.plot(litter[1:, c], color="orange")

    ax.set_title("Litterfall")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[2:4, 8:])
    for c in range(cols):
        ax.plot(finerootlitter[1:, c] / 10000.0, color="brown")

    ax.set_title("FinerootLitter")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    ax = fig.add_subplot(gs[:2, 8:])
    for c in range(cols):
        ax.plot(leaflitter[1:, c] / 10000.0, color="green")

    ax.set_title("LeafLitter")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    return fig


def carbon(
    vars: list[nc_utils.NetcdfVariableValue], scen: int = 0
) -> matplotlib.figure.Figure:
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(figsize=(15, 18))
    fig.suptitle("Carbon balance components", fontsize=fs + 2)
    gs = gridspec.GridSpec(ncols=12, nrows=14, figure=fig, wspace=0.5, hspace=0.5)
    mass_to_c = 0.5

    litter = (
        _get_var(vars, "/groundvegetation/ds_litterfall")[scen, :, :] / 10000.0
        + _get_var(vars, "/groundvegetation/h_litterfall")[scen, :, :] / 10000.0
        + _get_var(vars, "/groundvegetation/s_litterfall")[scen, :, :] / 10000.0
        + _get_var(vars, "/stand/nonwoodylitter")[scen, :, :] / 10000.0
        + _get_var(vars, "/stand/woodylitter")[scen, :, :] / 10000.0
    ) * mass_to_c

    soil = (
        _get_var(vars, "/esom/Mass/out")[scen, :, :] / 10000.0 * -1 * mass_to_c + litter
    )

    out = _get_var(vars, "/esom/Mass/out")[scen, :, :] / 10000.0 * -1 * mass_to_c

    wt = np.mean(_get_var(vars, "/strip/dwtyr")[scen, :, :], axis=0)
    cols = np.shape(wt)[0]
    sd = np.std(_get_var(vars, "/strip/dwtyr")[scen, :, :], axis=0)
    wtls = np.mean(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    sdls = np.std(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    wtmin = min(wtls) - 0.2

    ax = fig.add_subplot(gs[12:, :6])
    ax = _create_profile_line(
        ax,
        wt,
        wtmin,
        sd,
        cols,
        "WT m",
        "annual",
        fs,
        facecolor,
        "blue",
        hidex=True,
        hidey=False,
    )

    elevation = np.array(_get_var(vars, "/strip/elevation"))
    wtls = np.mean(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    sd = np.std(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    h = elevation + wtls

    ax = fig.add_subplot(gs[12:, 6:])
    ax = _create_profile_line(
        ax,
        h,
        wtmin,
        sd,
        cols,
        "WT m",
        "annual",
        fs,
        facecolor,
        "blue",
        hidex=True,
        hidey=False,
        elevation=elevation,
    )

    lmwtoditch = _get_var(vars, "/balance/C/LMWdoc_to_water")[scen, :, :] * -1
    ax = fig.add_subplot(gs[10:12, :6])
    df = pd.DataFrame(data=lmwtoditch, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "brown",
        "LMW to ditch",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    hmwtoditch = _get_var(vars, "/balance/C/HMW_to_water")[scen, :, :] * -1
    ax = fig.add_subplot(gs[10:12, 6:])
    df = pd.DataFrame(data=hmwtoditch, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "brown",
        "HMW to ditch",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    lmwtoatm = _get_var(vars, "/balance/C/LMWdoc_to_atm")[scen, :, :] * -1
    ax = fig.add_subplot(gs[8:10, :6])
    df = pd.DataFrame(data=lmwtoatm, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "orange",
        "LMW to atmosphere",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    hmwtoatm = _get_var(vars, "/balance/C/HMW_to_atm")[scen, :, :] * -1
    ax = fig.add_subplot(gs[8:10, 6:])
    df = pd.DataFrame(data=hmwtoatm, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "orange",
        "HMW to atmosphere",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    co2 = _get_var(vars, "/balance/C/co2c_release")[scen, :, :] * -1
    ax = fig.add_subplot(gs[6:8, :6])
    df = pd.DataFrame(data=co2, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "grey",
        "$CO_2C$ to atmosphere",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    ch4 = _get_var(vars, "/balance/C/ch4c_release")[scen, :, :] * -1
    ax = fig.add_subplot(gs[6:8, 6:])
    df = pd.DataFrame(data=ch4, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "grey",
        "$CH_4C$ to atmosphere",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    standl = _get_var(vars, "/balance/C/stand_litter_in")[scen, :, :]
    ax = fig.add_subplot(gs[4:6, :6])
    df = pd.DataFrame(data=standl, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "green",
        "Stand litter",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    gvl = _get_var(vars, "/balance/C/gv_litter_in")[scen, :, :]
    ax = fig.add_subplot(gs[4:6, 6:])
    df = pd.DataFrame(data=gvl, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "green",
        "Groundvegetation litter",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    soilc = _get_var(vars, "/balance/C/soil_c_balance_c")[scen, :, :]
    ax = fig.add_subplot(gs[2:4, :6])
    df = pd.DataFrame(data=soilc, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "black",
        "Soil C balance in C",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    soilco2 = _get_var(vars, "/balance/C/soil_c_balance_co2eq")[scen, :, :]
    ax = fig.add_subplot(gs[2:4, 6:])
    df = pd.DataFrame(data=soilco2, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "black",
        "Soil C balance in $CO_2$ eq",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    standc = _get_var(vars, "/balance/C/stand_c_balance_c")[scen, :, :]
    ax = fig.add_subplot(gs[:2, :6])
    df = pd.DataFrame(data=standc, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "green",
        "Stand C balance in C",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    standco2 = _get_var(vars, "/balance/C/stand_c_balance_co2eq")[scen, :, :]
    ax = fig.add_subplot(gs[:2, 6:])
    df = pd.DataFrame(data=standco2, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "green",
        "Stand C balance in $CO_2$ eq",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    return fig


def nutrient_balance(
    vars: list[nc_utils.NetcdfVariableValue],
    substance: str,
    scen: int = 0,
) -> matplotlib.figure.Figure:
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(figsize=(15, 18))
    tx = substance + " balance components"
    fig.suptitle(tx, fontsize=fs + 2)
    gs = gridspec.GridSpec(ncols=12, nrows=12, figure=fig, wspace=0.5, hspace=0.5)

    wt = np.mean(_get_var(vars, "/strip/dwtyr")[scen, :, :], axis=0)
    cols = np.shape(wt)[0]
    sd = np.std(_get_var(vars, "/strip/dwtyr")[scen, :, :], axis=0)
    wtls = np.mean(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    sdls = np.std(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    wtmin = min(wtls) - 0.2

    ax = fig.add_subplot(gs[10:, :6])
    ax = _create_profile_line(
        ax,
        wt,
        wtmin,
        sd,
        cols,
        "WT m",
        "annual",
        fs,
        facecolor,
        "blue",
        hidex=True,
        hidey=False,
    )

    elevation = np.array(_get_var(vars, "/strip/elevation"))
    wtls = np.mean(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    sd = np.std(_get_var(vars, "/strip/dwtyr_latesummer")[scen, :, :], axis=0)
    h = elevation + wtls

    ax = fig.add_subplot(gs[10:, 6:])
    ax = _create_profile_line(
        ax,
        h,
        wtmin,
        sd,
        cols,
        "WT m",
        "late summer",
        fs,
        facecolor,
        "blue",
        hidex=True,
        hidey=False,
        elevation=elevation,
    )

    towater = _get_var(vars, f"/balance/{substance}/to_water")[scen, :, :]
    ax = fig.add_subplot(gs[8:10, :6])
    df = pd.DataFrame(data=towater, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "brown",
        substance + " to water",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    brl = _get_var(vars, f"/balance/{substance}/decomposition_below_root_lyr")[
        scen, :, :
    ]
    ax = fig.add_subplot(gs[8:10, 6:])
    df = pd.DataFrame(data=brl, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "brown",
        substance + " below root layer",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    de = _get_var(vars, f"/balance/{substance}/decomposition_tot")[scen, :, :]
    ax = fig.add_subplot(gs[6:8, :6])
    df = pd.DataFrame(data=de, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "black",
        substance + " release in decomposition",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    dert = _get_var(vars, f"/balance/{substance}/decomposition_root_lyr")[scen, :, :]
    ax = fig.add_subplot(gs[6:8, 6:])
    df = pd.DataFrame(data=dert, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "black",
        substance + " release in root layer",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    supply = (
        _get_var(vars, f"/balance/{substance}/decomposition_root_lyr")[scen, :, :]
        + _get_var(vars, f"/balance/{substance}/deposition")[scen, :, :]
        + _get_var(vars, f"/balance/{substance}/fertilization_release")[scen, :, :]
    )

    ax = fig.add_subplot(gs[4:6, :6])
    df = pd.DataFrame(data=supply, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "blue",
        substance + " supply",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    fert = _get_var(vars, f"/balance/{substance}/fertilization_release")[scen, :, :]
    ax = fig.add_subplot(gs[4:6, 6:])
    df = pd.DataFrame(data=fert, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "black",
        substance + " release in fertilizers",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    dem = _get_var(vars, f"/balance/{substance}/stand_demand")[scen, :, :]
    ax = fig.add_subplot(gs[2:4, :6])
    df = pd.DataFrame(data=dem, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "green",
        substance + " stand uptake",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    dem = _get_var(vars, f"/balance/{substance}/gv_demand")[scen, :, :]
    ax = fig.add_subplot(gs[2:4, 6:])
    df = pd.DataFrame(data=dem, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "green",
        substance + " groundvegetation uptake",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    dem = _get_var(vars, f"/balance/{substance}/balance_root_lyr")[scen, :, :]
    ax = fig.add_subplot(gs[:2, :6])
    df = pd.DataFrame(data=dem, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "green",
        substance + " balance",
        "kg $ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    vg = _get_var(vars, "/stand/volumegrowth")[scen, :, :]
    ax = fig.add_subplot(gs[:2, 6:])
    df = pd.DataFrame(data=vg, columns=list(range(cols)))
    ax = _create_profile_boxplot(
        ax,
        df,
        cols,
        "green",
        "volume growth",
        "$m^{3} ha^{-1} yr^{-1}$",
        fs,
        facecolor,
        zero=False,
    )

    return fig
