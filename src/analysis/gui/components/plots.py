import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from susi.io.load_output_data import NetcdfVariablePath, NetcdfVariableArray


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
    df = pd.DataFrame(data=datain, columns=np.arange(cols))
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
    n_time, _n_space = data.shape
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


def _get_var(
    data: dict[NetcdfVariablePath, NetcdfVariableArray],
    var_path: str,
) -> NetcdfVariableArray:
    return data[NetcdfVariablePath(var_path)]


def stand(
    data: dict[NetcdfVariablePath, NetcdfVariableArray],
):
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(figsize=(15, 18))  # width, height
    gs = gridspec.GridSpec(ncols=12, nrows=12, figure=fig, wspace=0.25, hspace=0.25)

    wt = _get_var(data, "/strip/dwtyr").mean_over_time()
    cols = np.shape(wt)[0]
    sd = np.std(_get_var(data, "/strip/dwtyr").processed, axis=0)
    wtls = _get_var(data, "/strip/dwtyr_latesummer").mean_over_time()
    sdls = np.std(_get_var(data, "/strip/dwtyr_latesummer").processed, axis=0)
    wtmin = min(wtls) - 0.2

    # -------water table as a reference-----------------------------
    ax = fig.add_subplot(gs[10:, :4])
    ax.plot(wtls, color="orange", label="late summer")
    ax.fill_between(
        range(cols), wtls + sdls * 2, wtls - sd * 2, color="orange", alpha=0.075
    )
    ax.hlines(y=-0.35, xmin=0, xmax=cols, color="red", linestyles="--")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_ylim(bottom=wtmin, top=0)
    ax.set_ylabel("WT m", fontsize=fs)
    ax.legend()
    ax.grid(visible=False)
    ax.set_facecolor(facecolor)

    # ------------stand growth--------------------
    vol = _get_var(data, "/stand/volume").processed
    growth = np.diff(vol, axis=0)
    dfgrowth = pd.DataFrame(data=growth, columns=np.arange(cols))
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

    # -------------end volume-------------------
    ax = fig.add_subplot(gs[10:, 4:8])
    totvol = vol[-1, :]
    domvol = _get_var(data, "/stand/dominant/volume")
    subdomvol = _get_var(data, "/stand/subdominant/volume")
    undervol = _get_var(data, "/stand/under/volume")

    domvol_last = domvol.last_timestep()
    subdomvol_last = subdomvol.last_timestep()
    undervol_last = undervol.last_timestep()
    df = pd.DataFrame(
        {
            "total": totvol,
            "dominant": domvol_last,
            "subdominant": subdomvol_last,
            "under": undervol_last,
        },
        index=range(cols),
    )
    df.plot(kind="bar", width=1.1, ax=ax)
    ax.set_title("Stand volume")
    # ax.set_ylabel('$m^3 ha^{-1}$', fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)
    ax.legend(loc="upper right", fontsize=7)

    # ----- volume growth------------------------
    ax = fig.add_subplot(gs[10:, 8:])
    totvol = vol[:, :]
    domvol = domvol.processed
    subdomvol = subdomvol.processed
    undervol = undervol.processed
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

    # ----- volume growth ------------------------
    ax = fig.add_subplot(gs[8:10, 4:8])

    for c in range(cols):
        ax.plot(np.diff(vol[:, c]), alpha=0.2)

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Volume")
    ax.get_xaxis().set_visible(False)

    # ----- log volume growth------------------------
    ax = fig.add_subplot(gs[8:10, 8:])
    logvol = _get_var(data, "/stand/logvolume").processed
    pulpvol = _get_var(data, "/stand/pulpvolume").processed
    for c in range(cols):
        yrs = len(logvol[:, c])
        ax.plot(range(1, yrs), logvol[1:, c], alpha=0.2)
    for c in range(cols):
        ax.plot(range(1, yrs), pulpvol[1:, c], alpha=0.2)

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Log and pulp volume")
    ax.get_xaxis().set_visible(False)

    # -----------------leaf mass------------------
    lmass = _get_var(data, "/stand/leafmass").processed
    df = pd.DataFrame(data=lmass, columns=np.arange(cols))
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
    ax.set_ylabel(r"$kg \ ha^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    # ----- leaf mass time series------------------------
    ax = fig.add_subplot(gs[6:8, 4:8])
    for c in range(cols):
        ax.plot(lmass[:, c], alpha=0.2)

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Leaf mass")
    ax.get_xaxis().set_visible(False)

    # ----- leaf mass time series------------------------
    dlmass = _get_var(data, "/stand/dominant/leafmass").processed
    upperlim = _get_var(data, "/stand/dominant/leafmax").processed
    lowerlim = _get_var(data, "/stand/dominant/leafmin").processed
    ax = fig.add_subplot(gs[6:8, 8:])
    for c in range(cols):
        yrs = len(dlmass[:, c])
        ax.fill_between(
            range(1, yrs), upperlim[1:, c], lowerlim[1:, c], color="green", alpha=0.01
        )
        ax.plot(dlmass[:, c], alpha=0.5, color="green")
    ax.plot(dlmass[:, c], alpha=0.5, color="green", label="dominant")

    # ax.plot(range(1,yrs), upperlim[1:,c],  color='blue', alpha=0.01)
    sdlmass = _get_var(data, "/stand/subdominant/leafmass").processed
    sdupperlim = _get_var(data, "/stand/subdominant/leafmax").processed
    sdlowerlim = _get_var(data, "/stand/subdominant/leafmin").processed
    for c in range(cols):
        yrs = len(sdlmass[:, c])
        ax.fill_between(
            range(1, yrs),
            sdupperlim[1:, c],
            sdlowerlim[1:, c],
            color="cyan",
            alpha=0.01,
        )
        ax.plot(sdlmass[:, c], alpha=0.5, color="cyan")
    ax.plot(sdlmass[:, c], alpha=0.5, color="cyan", label="subdominant")

    ulmass = _get_var(data, "/stand/under/leafmass").processed
    uupperlim = _get_var(data, "/stand/under/leafmax").processed
    ulowerlim = _get_var(data, "/stand/under/leafmin").processed
    for c in range(cols):
        yrs = len(ulmass[:, c])
        ax.fill_between(
            range(1, yrs),
            uupperlim[1:, c],
            ulowerlim[1:, c],
            color="orange",
            alpha=0.01,
        )
        ax.plot(ulmass[:, c], alpha=0.5, color="orange")
    ax.plot(ulmass[:, c], alpha=0.5, color="orange", label="under")

    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_title("Leaf mass in canopy layers")
    ax.get_xaxis().set_visible(False)

    # -----------nutrient status-------------------------
    ax = fig.add_subplot(gs[4:6, :4])

    ns = _get_var(data, "/stand/nut_stat").processed
    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(ns[:, c], alpha=0.5, color="orange")
    ax.plot(ns[:, c], alpha=0.5, color="orange", label="nutrient status")

    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    # ax.set_title('Nutrient status')
    ax.get_xaxis().set_visible(False)

    # -----------physcal growth restrictions-------------------------
    ax = fig.add_subplot(gs[4:6, 4:8])
    dom_phys_r = (
        _get_var(data, "/stand/dominant/NPP").processed
        / _get_var(data, "/stand/dominant/NPP_pot").processed
    )
    df = pd.DataFrame(data=dom_phys_r, columns=np.arange(cols))
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

    # -----------nutrient status-------------------------
    ax = fig.add_subplot(gs[4:6, 8:])

    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(ns[:-1, c], growth[:, c], "go", alpha=0.5)

    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    # ax.set_title('Nutrient status')
    ax.set_ylabel("volume growth")
    ax.set_xlabel("nutrient status")

    ndemand = _get_var(data, "/stand/n_demand").processed
    df = pd.DataFrame(data=ndemand, columns=np.arange(cols))
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
    ax.set_ylabel(r"$kg \ ha^{-1} \ yr^{-1}$", fontsize=fs)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    pdemand = _get_var(data, "/stand/p_demand").processed
    df = pd.DataFrame(data=pdemand, columns=np.arange(cols))
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

    kdemand = _get_var(data, "/stand/k_demand").processed
    df = pd.DataFrame(data=kdemand, columns=np.arange(cols))
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
    # ax.set_title('Nutrient status')
    ax.set_ylabel("kg $ha^{-1} yr^{-1}$", fontsize=fs)

    # -----------------------------------
    ax = fig.add_subplot(gs[:2, 4:8])

    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(range(1, yrs), pdemand[1:, c], alpha=0.5, color="green")
    ax.plot(range(1, yrs), pdemand[1:, c], alpha=0.5, color="green", label="P demand")
    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)
    # -----------------------------------
    ax = fig.add_subplot(gs[:2, 8:])

    for c in range(cols):
        yrs = len(ns[:, c])
        ax.plot(range(1, yrs), kdemand[1:, c], alpha=0.5, color="orange")
    ax.plot(range(1, yrs), kdemand[1:, c], alpha=0.5, color="orange", label="K demand")
    ax.legend(loc="upper left")

    ax.set_facecolor(facecolor)
    ax.tick_params(axis="y", labelsize=fs)

    return fig


def hydrology(data: dict[NetcdfVariablePath, NetcdfVariableArray]):
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(num="hydro", figsize=(15, 18))
    gs = gridspec.GridSpec(ncols=12, nrows=12, figure=fig, wspace=0.25, hspace=0.25)

    wt = _get_var(data, "/strip/dwtyr").mean_over_time()
    cols = np.shape(wt)[0]
    sd = np.std(_get_var(data, "/strip/dwtyr").processed, axis=0)
    wtgs = _get_var(data, "/strip/dwtyr_growingseason").mean_over_time()
    sdgs = np.std(_get_var(data, "/strip/dwtyr_growingseason").processed, axis=0)
    wtls = _get_var(data, "/strip/dwtyr_latesummer").mean_over_time()
    sdls = np.std(_get_var(data, "/strip/dwtyr_latesummer").processed, axis=0)
    wtmin = min(wtls) - 0.2

    ax = fig.add_subplot(gs[10:, :4])
    _ax = _create_profile_line(
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
    _ax = _create_profile_line(
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
    _ax = _create_profile_line(
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

    dwt = _get_var(data, "/strip/dwt").processed
    wt_ts = np.mean(dwt, axis=1)
    days = np.shape(wt_ts)[0]

    axwtts = fig.add_subplot(gs[8:10, :])
    axwtts.plot(wt_ts, color="green", label="WT")
    axwtts.hlines(y=-0.35, xmin=0, xmax=days, color="red", linestyles="--")
    for c in range(1, cols - 1):
        axwtts.plot(range(days), dwt[:, c], alpha=0.2)

    axwtts.tick_params(axis="y", labelsize=fs)
    axwtts.set_ylim(bottom=wtmin, top=0)
    axwtts.set_ylabel("WT m", fontsize=fs)
    axwtts.legend(loc="upper left")
    axwtts.grid(visible=False)
    axwtts.set_facecolor(facecolor)

    roff = _get_var(data, "/strip/roff").processed
    runoff = np.cumsum(roff)
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

    roffwest = _get_var(data, "/strip/roffwest").processed
    runoff = np.cumsum(roffwest)
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

    roffeast = _get_var(data, "/strip/roffeast").processed
    runoff = np.cumsum(roffeast)
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

    surfacerunoff = _get_var(data, "/strip/surfacerunoff").processed
    runoff = np.cumsum(np.mean(surfacerunoff, axis=1))
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

    deltas = _get_var(data, "/strip/deltas").processed[1:, :] * 1000.0
    dfdeltas = pd.DataFrame(data=deltas, columns=np.arange(cols))

    ax = fig.add_subplot(gs[2:4, :4])
    _ax = _create_profile_boxplot(
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

    ET = _get_var(data, "/cpy/ET_yr").processed[1:, :] * 1000.0
    dfET = pd.DataFrame(data=ET, columns=np.arange(cols))

    ax = fig.add_subplot(gs[2:4, 4:8])
    _ax = _create_profile_boxplot(
        ax, dfET, cols, "green", "ET", "", fs, facecolor, zero=False, hidex=True
    )

    transpi = _get_var(data, "/cpy/transpi_yr").processed[1:, :] * 1000.0
    dftranspi = pd.DataFrame(data=transpi, columns=np.arange(cols))
    ax = fig.add_subplot(gs[2:4, 8:])
    _ax = _create_profile_boxplot(
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

    efloor = _get_var(data, "/cpy/efloor_yr").processed[1:, :] * 1000.0
    dfefloor = pd.DataFrame(data=efloor, columns=np.arange(cols))

    ax = fig.add_subplot(gs[:2, :4])
    _ax = _create_profile_boxplot(
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

    swe = _get_var(data, "/cpy/SWEmax").processed[1:, :]
    dfswe = pd.DataFrame(data=swe, columns=np.arange(cols))

    ax = fig.add_subplot(gs[:2, 4:8])
    _ax = _create_profile_boxplot(
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

    interc = _get_var(data, "/cpy/interc_yr").processed[1:, :] * 1000.0
    dfinterc = pd.DataFrame(data=interc, columns=np.arange(cols))

    ax = fig.add_subplot(gs[:2, 8:])
    _ax = _create_profile_boxplot(
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


def mass(data: dict[NetcdfVariablePath, NetcdfVariableArray]):
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(ncols=12, nrows=12, figure=fig, wspace=0.25, hspace=0.25)

    wt = _get_var(data, "/strip/dwtyr").mean_over_time()
    cols = np.shape(wt)[0]
    sd = np.std(_get_var(data, "/strip/dwtyr").processed, axis=0)
    wtls = _get_var(data, "/strip/dwtyr_latesummer").mean_over_time()
    sdls = np.std(_get_var(data, "/strip/dwtyr_latesummer").processed, axis=0)
    wtmin = min(wtls) - 0.2

    ax = fig.add_subplot(gs[10:, :4])
    ax.plot(wtls, color="orange", label="late summer")
    ax.fill_between(
        range(cols), wtls + sdls * 2, wtls - sd * 2, color="orange", alpha=0.075
    )
    ax.hlines(y=-0.35, xmin=0, xmax=cols, color="red", linestyles="--")
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_ylim(bottom=wtmin, top=0)
    ax.set_ylabel("WT m", fontsize=fs)
    ax.legend()
    ax.grid(visible=False)
    ax.set_facecolor(facecolor)

    ds_litterfall = (
        _get_var(data, "/groundvegetation/ds_litterfall").processed / 10000.0
    )
    h_litterfall = _get_var(data, "/groundvegetation/h_litterfall").processed / 10000.0
    s_litterfall = _get_var(data, "/groundvegetation/s_litterfall").processed / 10000.0
    nonwoodylitter = _get_var(data, "/stand/nonwoodylitter").processed / 10000.0
    woodylitter = _get_var(data, "/stand/woodylitter").processed / 10000.0

    litter = ds_litterfall + h_litterfall + s_litterfall + nonwoodylitter + woodylitter

    esom_mass_out = _get_var(data, "/esom/Mass/out").processed / 10000.0 * -1
    soil = esom_mass_out + litter
    soilout = esom_mass_out

    df = pd.DataFrame(data=soil, columns=np.arange(cols))
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
        inipeat += _get_var(data, f"/esom/Mass/{sto}").processed[0, :] / 10000.0
    endpeat = np.zeros(cols)
    for sto in esoms[7:]:
        endpeat += _get_var(data, f"/esom/Mass/{sto}").processed[-1, :] / 10000.0
    inimor = np.zeros(cols)
    for sto in esoms[:7]:
        inimor += _get_var(data, f"/esom/Mass/{sto}").processed[0, :] / 10000.0
    endmor = np.zeros(cols)
    for sto in esoms[:7]:
        endmor += _get_var(data, f"/esom/Mass/{sto}").processed[-1, :] / 10000.0

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

    df = pd.DataFrame(data=litter, columns=np.arange(cols))
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

    gv = _get_var(data, "/groundvegetation/gv_tot").processed / 10000.0
    grgv = np.diff(gv, axis=0)
    df = pd.DataFrame(data=grgv, columns=np.arange(cols))
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

    stand = _get_var(data, "/stand/biomass").processed / 10000.0
    gr = np.diff(stand, axis=0)
    df = pd.DataFrame(data=gr, columns=np.arange(cols))
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
    df = pd.DataFrame(data=site, columns=np.arange(cols))
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

    leaflitter = nonwoodylitter - _get_var(data, "/stand/finerootlitter").processed

    df = pd.DataFrame(data=leaflitter, columns=np.arange(cols))
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

    finerootlitter = _get_var(data, "/stand/finerootlitter").processed

    df = pd.DataFrame(data=finerootlitter, columns=np.arange(cols))
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

    df = pd.DataFrame(data=woodylitter, columns=np.arange(cols))
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

    gvlitter = ds_litterfall + h_litterfall + s_litterfall

    df = pd.DataFrame(data=gvlitter, columns=np.arange(cols))
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

    out = esom_mass_out

    df = pd.DataFrame(data=out, columns=np.arange(cols))
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

    yrs = np.shape(_get_var(data, "/esom/Mass/L0L").processed)[0]
    ax = fig.add_subplot(gs[10:, 8:])
    for c in range(cols):
        peat = np.zeros(yrs)
        for sto in esoms[7:]:
            peat += _get_var(data, f"/esom/Mass/{sto}").processed[:, c] / 10000.0
        mor = np.zeros(yrs)
        for sto in esoms[:7]:
            mor += _get_var(data, f"/esom/Mass/{sto}").processed[:, c] / 10000.0

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

    out = esom_mass_out

    ax = fig.add_subplot(gs[8:10, 8:])
    for c in range(cols):
        ax.plot(out[1:, c], color="orange")

    ax.set_title("Soil mass loss")

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_facecolor(facecolor)

    out = esom_mass_out + litter

    ax = fig.add_subplot(gs[6:8, 8:])
    for c in range(cols):
        ax.plot(out[1:, c], color="red")

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


def carbon(data: dict[NetcdfVariablePath, NetcdfVariableArray]):
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(figsize=(15, 18))
    fig.suptitle("Carbon balance components", fontsize=fs + 2)
    gs = gridspec.GridSpec(ncols=12, nrows=14, figure=fig, wspace=0.5, hspace=0.5)
    # mass_to_c = 0.5
    #
    # ds_litterfall = (
    #     _get_var(data, "/groundvegetation/ds_litterfall").processed / 10000.0
    # )
    # h_litterfall = _get_var(data, "/groundvegetation/h_litterfall").processed / 10000.0
    # s_litterfall = _get_var(data, "/groundvegetation/s_litterfall").processed / 10000.0
    # nonwoodylitter = _get_var(data, "/stand/nonwoodylitter").processed / 10000.0
    # woodylitter = _get_var(data, "/stand/woodylitter").processed / 10000.0

    # litter = (
    #     ds_litterfall + h_litterfall + s_litterfall + nonwoodylitter + woodylitter
    # ) * mass_to_c
    #
    # esom_mass_out = (
    #     _get_var(data, "/esom/Mass/out").processed / 10000.0 * -1 * mass_to_c
    # )
    # soil = esom_mass_out + litter
    # out = esom_mass_out

    wt = _get_var(data, "/strip/dwtyr").mean_over_time()
    cols = np.shape(wt)[0]
    sd = np.std(_get_var(data, "/strip/dwtyr").processed, axis=0)
    wtls = _get_var(data, "/strip/dwtyr_latesummer").mean_over_time()
    # sdls = np.std(_get_var(data, "/strip/dwtyr_latesummer").processed, axis=0)
    wtmin = min(wtls) - 0.2

    ax = fig.add_subplot(gs[12:, :6])
    _ax = _create_profile_line(
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

    elevation = _get_var(data, "/strip/elevation").processed[1:-1]
    wtls = _get_var(data, "/strip/dwtyr_latesummer").mean_over_time()
    sd = np.std(_get_var(data, "/strip/dwtyr_latesummer").processed, axis=0)
    h = elevation + wtls

    ax = fig.add_subplot(gs[12:, 6:])
    _ax = _create_profile_line(
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

    lmwtoditch = _get_var(data, "/balance/C/LMWdoc_to_water").processed * -1
    ax = fig.add_subplot(gs[10:12, :6])
    df = pd.DataFrame(data=lmwtoditch, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    hmwtoditch = _get_var(data, "/balance/C/HMW_to_water").processed * -1
    ax = fig.add_subplot(gs[10:12, 6:])
    df = pd.DataFrame(data=hmwtoditch, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    lmwtoatm = _get_var(data, "/balance/C/LMWdoc_to_atm").processed * -1
    ax = fig.add_subplot(gs[8:10, :6])
    df = pd.DataFrame(data=lmwtoatm, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    hmwtoatm = _get_var(data, "/balance/C/HMW_to_atm").processed * -1
    ax = fig.add_subplot(gs[8:10, 6:])
    df = pd.DataFrame(data=hmwtoatm, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    co2 = _get_var(data, "/balance/C/co2c_release").processed * -1
    ax = fig.add_subplot(gs[6:8, :6])
    df = pd.DataFrame(data=co2, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    co2 = _get_var(data, "/balance/C/ch4c_release").processed * -1
    ax = fig.add_subplot(gs[6:8, 6:])
    df = pd.DataFrame(data=co2, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    standl = _get_var(data, "/balance/C/stand_litter_in").processed
    ax = fig.add_subplot(gs[4:6, :6])
    df = pd.DataFrame(data=standl, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    gvl = _get_var(data, "/balance/C/gv_litter_in").processed
    ax = fig.add_subplot(gs[4:6, 6:])
    df = pd.DataFrame(data=gvl, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    soilc = _get_var(data, "/balance/C/soil_c_balance_c").processed
    ax = fig.add_subplot(gs[2:4, :6])
    df = pd.DataFrame(data=soilc, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    soilco2 = _get_var(data, "/balance/C/soil_c_balance_co2eq").processed
    ax = fig.add_subplot(gs[2:4, 6:])
    df = pd.DataFrame(data=soilco2, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    standc = _get_var(data, "/balance/C/stand_c_balance_c").processed
    ax = fig.add_subplot(gs[:2, :6])
    df = pd.DataFrame(data=standc, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    standco2 = _get_var(data, "/balance/C/stand_c_balance_co2eq").processed
    ax = fig.add_subplot(gs[:2, 6:])
    df = pd.DataFrame(data=standco2, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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
    data: dict[NetcdfVariablePath, NetcdfVariableArray], substance: str
):
    facecolor = "#f2f5eb"
    fs = 15
    fig = plt.figure(figsize=(15, 18))
    tx = substance + " balance components"
    fig.suptitle(tx, fontsize=fs + 2)
    gs = gridspec.GridSpec(ncols=12, nrows=12, figure=fig, wspace=0.5, hspace=0.5)

    wt = _get_var(data, "/strip/dwtyr").mean_over_time()
    cols = np.shape(wt)[0]
    sd = np.std(_get_var(data, "/strip/dwtyr").processed, axis=0)
    wtls = _get_var(data, "/strip/dwtyr_latesummer").mean_over_time()
    # sdls = np.std(_get_var(data, "/strip/dwtyr_latesummer").processed, axis=0)
    wtmin = min(wtls) - 0.2

    ax = fig.add_subplot(gs[10:, :6])
    _ax = _create_profile_line(
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

    elevation = _get_var(data, "/strip/elevation").processed[1:-1]
    wtls = _get_var(data, "/strip/dwtyr_latesummer").mean_over_time()
    sd = np.std(_get_var(data, "/strip/dwtyr_latesummer").processed, axis=0)
    h = elevation + wtls

    ax = fig.add_subplot(gs[10:, 6:])
    _ax = _create_profile_line(
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

    towater = _get_var(data, f"/balance/{substance}/to_water").processed
    ax = fig.add_subplot(gs[8:10, :6])
    df = pd.DataFrame(data=towater, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    brl = _get_var(data, f"/balance/{substance}/decomposition_below_root_lyr").processed
    ax = fig.add_subplot(gs[8:10, 6:])
    df = pd.DataFrame(data=brl, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    de = _get_var(data, f"/balance/{substance}/decomposition_tot").processed
    ax = fig.add_subplot(gs[6:8, :6])
    df = pd.DataFrame(data=de, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    dert = _get_var(data, f"/balance/{substance}/decomposition_root_lyr").processed
    ax = fig.add_subplot(gs[6:8, 6:])
    df = pd.DataFrame(data=dert, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    decomposition_root_lyr = _get_var(
        data, f"/balance/{substance}/decomposition_root_lyr"
    ).processed
    deposition = _get_var(data, f"/balance/{substance}/deposition").processed
    fertilization_release = _get_var(
        data, f"/balance/{substance}/fertilization_release"
    ).processed
    supply = decomposition_root_lyr + deposition + fertilization_release

    ax = fig.add_subplot(gs[4:6, :6])
    df = pd.DataFrame(data=supply, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    fert = _get_var(data, f"/balance/{substance}/fertilization_release").processed
    ax = fig.add_subplot(gs[4:6, 6:])
    df = pd.DataFrame(data=fert, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    dem = _get_var(data, f"/balance/{substance}/stand_demand").processed
    ax = fig.add_subplot(gs[2:4, :6])
    df = pd.DataFrame(data=dem, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    dem = _get_var(data, f"/balance/{substance}/gv_demand").processed
    ax = fig.add_subplot(gs[2:4, 6:])
    df = pd.DataFrame(data=dem, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    dem = _get_var(data, f"/balance/{substance}/balance_root_lyr").processed
    ax = fig.add_subplot(gs[:2, :6])
    df = pd.DataFrame(data=dem, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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

    vg = _get_var(data, "/stand/volumegrowth").processed
    ax = fig.add_subplot(gs[:2, 6:])
    df = pd.DataFrame(data=vg, columns=np.arange(cols))
    _ax = _create_profile_boxplot(
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
