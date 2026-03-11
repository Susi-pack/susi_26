import altair as alt
import numpy as np
import pandas as pd

import susi.io.load_output_data as load_output


def _get_var(
    data_store: load_output.OutputDataStore,
    path: str,
    stand_id: load_output.StandID,
    scenario_id: load_output.ScenarioID,
) -> np.ndarray:
    return data_store.get_variable_value_for_scenario_and_stand(
        load_output.NetcdfVariablePath(path), stand_id, scenario_id
    ).processed


def _create_line_chart(
    df: pd.DataFrame,
    title: str,
    y_label: str,
    color: str = "steelblue",
    show_legend: bool = True,
) -> alt.Chart:
    df_melted = df.reset_index().melt(
        id_vars="index", var_name="space", value_name="value"
    )
    df_melted = df_melted.rename(columns={"index": "time"})

    chart = (
        alt.Chart(df_melted)
        .mark_line(opacity=0.3)
        .encode(
            x=alt.X("time", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("value", title=y_label),
            color=alt.ColorValue(color),
            detail="space",
        )
    )

    mean_line = (
        alt.Chart(df_melted)
        .mark_line(opacity=0.8, strokeWidth=2)
        .encode(
            x="time",
            y=alt.Y("mean(value)", title=y_label),
            color=alt.ColorValue(color),
        )
    )

    combined = chart + mean_line
    combined = combined.properties(title=title)
    if not show_legend:
        combined = combined.configure_legend(disable=True)
    return combined


def _create_boxplot(
    df: pd.DataFrame,
    title: str,
    y_label: str,
    color: str = "steelblue",
) -> alt.Chart:
    df_melted = df.reset_index().melt(
        id_vars="index", var_name="space", value_name="value"
    )
    df_melted = df_melted.rename(columns={"index": "time"})

    chart = (
        alt.Chart(df_melted)
        .mark_boxplot(opacity=0.6, color=color)
        .encode(
            x=alt.X("time", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("value", title=y_label),
        )
        .properties(title=title)
    )
    return chart


def _create_profile_line(
    data: np.ndarray,
    title: str,
    y_label: str,
    color: str = "steelblue",
    show_drain_line: bool = True,
) -> alt.Chart:
    df = pd.DataFrame({"x": range(len(data)), "value": data})

    line = (
        alt.Chart(df)
        .mark_line(color=color)
        .encode(
            x=alt.X("x", title="Space", axis=alt.Axis(labels=True)),
            y=alt.Y("value", title=y_label, scale=alt.Scale(zero=False)),
        )
        .properties(title=title)
    )

    if show_drain_line:
        drain_line = (
            alt.Chart(pd.DataFrame({"y": [-0.35]}))
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(y="y")
        )
        line = line + drain_line

    return line


def _create_stacked_bar(
    df: pd.DataFrame,
    title: str,
    y_label: str,
) -> alt.Chart:
    df_melted = df.reset_index().melt(
        id_vars="index", var_name="category", value_name="value"
    )
    df_melted = df_melted.rename(columns={"index": "space"})

    chart = (
        alt.Chart(df_melted)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("space", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("value", title=y_label),
            color=alt.Color("category", scale=alt.Scale(scheme="browns")),
        )
        .properties(title=title)
    )
    return chart


def _create_scatter(
    x_data: np.ndarray,
    y_data: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "green",
) -> alt.Chart:
    df = pd.DataFrame({"x": x_data, "y": y_data})

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.5, color=color)
        .encode(
            x=alt.X("x", title=x_label),
            y=alt.Y("y", title=y_label),
        )
        .properties(title=title)
    )
    return chart


def stand(
    data_store: load_output.OutputDataStore,
    stand_id: load_output.StandID,
    scenario_id: load_output.ScenarioID,
):
    cols = np.shape(_get_var(data_store, "/strip/dwtyr", stand_id, scenario_id))[1]

    wtls = np.mean(
        _get_var(data_store, "/strip/dwtyr_latesummer", stand_id, scenario_id), axis=0
    )
    sdls = np.std(
        _get_var(data_store, "/strip/dwtyr_latesummer", stand_id, scenario_id), axis=0
    )

    vol = _get_var(data_store, "/stand/volume", stand_id, scenario_id)
    growth = np.diff(vol, axis=0)
    dfgrowth = pd.DataFrame(data=growth, columns=np.arange(cols))

    totvol = vol[-1, :]
    domvol = _get_var(data_store, "/stand/dominant/volume", stand_id, scenario_id)[-1, :]
    subdomvol = _get_var(data_store, "/stand/subdominant/volume", stand_id, scenario_id)[-1, :]
    undervol = _get_var(data_store, "/stand/under/volume", stand_id, scenario_id)[-1, :]
    dfvol = pd.DataFrame(
        {
            "total": totvol,
            "dominant": domvol,
            "subdominant": subdomvol,
            "under": undervol,
        },
        index=range(cols),
    )

    logvol = _get_var(data_store, "/stand/logvolume"), stand_id, scenario_id)
    pulpvol = _get_var(data_store, "/stand/pulpvolume"), stand_id, scenario_id)

    lmass = _get_var(data_store, "/stand/leafmass"), stand_id, scenario_id)
    dflmass = pd.DataFrame(data=lmass, columns=np.arange(cols))

    dlmass_dom = _get_var(data_store, "/stand/dominant/leafmass"), stand_id, scenario_id)
    upperlim_dom = _get_var(data_store, "/stand/dominant/leafmax"), stand_id, scenario_id)
    lowerlim_dom = _get_var(data_store, "/stand/dominant/leafmin"), stand_id, scenario_id)

    dlmass_subdom = _get_var(data_store, "/stand/subdominant/leafmass"), stand_id, scenario_id)
    upperlim_subdom = _get_var(data_store, "/stand/subdominant/leafmax"), stand_id, scenario_id)
    lowerlim_subdom = _get_var(data_store, "/stand/subdominant/leafmin"), stand_id, scenario_id)

    dlmass_under = _get_var(data_store, "/stand/under/leafmass"), stand_id, scenario_id)
    upperlim_under = _get_var(data_store, "/stand/under/leafmax"), stand_id, scenario_id)
    lowerlim_under = _get_var(data_store, "/stand/under/leafmin"), stand_id, scenario_id)

    ns = _get_var(data_store, "/stand/nut_stat"), stand_id, scenario_id)
    dfns = pd.DataFrame(data=ns, columns=np.arange(cols))

    dom_phys_r = (
        _get_var(data_store, "/stand/dominant/NPP"), stand_id, scenario_id)
        / _get_var(data_store, "/stand/dominant/NPP_pot"), stand_id, scenario_id)
    )
    df_phys_r = pd.DataFrame(data=dom_phys_r, columns=np.arange(cols))

    ndemand = _get_var(data_store, "/stand/n_demand"), stand_id, scenario_id)
    df_ndemand = pd.DataFrame(data=ndemand, columns=np.arange(cols))

    pdemand = _get_var(data_store, "/stand/p_demand"), stand_id, scenario_id)
    df_pdemand = pd.DataFrame(data=pdemand, columns=np.arange(cols))

    kdemand = _get_var(data_store, "/stand/k_demand"), stand_id, scenario_id)
    df_kdemand = pd.DataFrame(data=kdemand, columns=np.arange(cols))

    chart_wt = _create_profile_line(
        wtls, "Water Table - Late Summer", "WT (m)", "orange"
    )

    chart_growth = _create_boxplot(
        dfgrowth, "Stand Growth", "$m^3 ha^{-1} yr^{-1}$", "blue"
    )

    chart_vol_bar = _create_stacked_bar(dfvol, "Stand Volume", "$m^3 ha^{-1}$")

    chart_vol_inc = _create_line_chart(
        pd.DataFrame(vol), "Stand Volume Increment", "$m^3 ha^{-1}$", "blue"
    )

    vol_data = _get_var(data_store, "/stand/volume", stand_id, scenario_id)[0:, :]
    chart_volume = _create_line_chart(
        pd.DataFrame(vol_data), "Volume", "$m^3 ha^{-1}$", "blue"
    )

    logpulp_data = pd.DataFrame(
        {
            "logvol": logvol[1:, :].mean(axis=1),
            "pulpvol": pulpvol[1:, :].mean(axis=1),
        }
    )
    chart_logpulp = (
        alt.Chart(
            logpulp_data.reset_index().melt(
                id_vars="index", var_name="type", value_name="value"
            )
        )
        .mark_line(opacity=0.7)
        .encode(
            x="index",
            y="value",
            color=alt.Color(
                "type",
                scale=alt.Scale(
                    domain=["logvol", "pulpvol"], range=["brown", "orange"]
                ),
            ),
        )
        .properties(title="Log and Pulp Volume")
    )

    chart_volume = _create_line_chart(
        pd.DataFrame(vol_data), "Volume", "$m^3 ha^{-1}$", "blue"
    )

    chart_logpulp = _create_line_chart(
        pd.DataFrame(logvol), "Log and Pulp Volume", "$m^3 ha^{-1}$", "brown"
    ) + _create_line_chart(pd.DataFrame(pulpvol), "", "$m^3 ha^{-1}$", "orange")

    chart_lmass_box = _create_boxplot(dflmass, "Leaf Mass", "$kg ha^{-1}$", "green")

    chart_lmass_line = _create_line_chart(dflmass, "Leaf Mass", "$kg ha^{-1}$", "green")

    leaf_data = []
    for c in range(cols):
        for t in range(1, len(dlmass_dom[:, c])):
            leaf_data.append(
                {
                    "time": t,
                    "space": c,
                    "value": dlmass_dom[t, c],
                    "layer": "dominant",
                    "upper": upperlim_dom[t, c],
                    "lower": lowerlim_dom[t, c],
                }
            )
            leaf_data.append(
                {
                    "time": t,
                    "space": c,
                    "value": dlmass_subdom[t, c],
                    "layer": "subdominant",
                    "upper": upperlim_subdom[t, c],
                    "lower": lowerlim_subdom[t, c],
                }
            )
            leaf_data.append(
                {
                    "time": t,
                    "space": c,
                    "value": dlmass_under[t, c],
                    "layer": "under",
                    "upper": upperlim_under[t, c],
                    "lower": lowerlim_under[t, c],
                }
            )

    df_leaf = pd.DataFrame(leaf_data)
    chart_leaf_canopy = (
        alt.Chart(df_leaf)
        .mark_line(opacity=0.5)
        .encode(
            x="time",
            y="value",
            color=alt.Color(
                "layer",
                scale=alt.Scale(
                    domain=["dominant", "subdominant", "under"],
                    range=["green", "cyan", "orange"],
                ),
            ),
            detail="space",
        )
        .properties(title="Leaf Mass in Canopy Layers")
    )

    chart_ns = _create_line_chart(dfns, "Nutrient Status", "NS", "orange")

    chart_phys_r = _create_boxplot(df_phys_r, "Physical Restrictions", "", "blue")

    chart_ns_growth = _create_scatter(
        ns[:-1, :].flatten(),
        growth.flatten(),
        "Nutrient Status vs Volume Growth",
        "nutrient status",
        "volume growth",
    )

    chart_ndemand = _create_boxplot(
        df_ndemand, "N Demand", "$kg ha^{-1} yr^{-1}$", "blue"
    )
    chart_pdemand = _create_boxplot(
        df_pdemand, "P Demand", "$kg ha^{-1} yr^{-1}$", "green"
    )
    chart_kdemand = _create_boxplot(
        df_kdemand, "K Demand", "$kg ha^{-1} yr^{-1}$", "orange"
    )

    chart_ndemand_line = _create_line_chart(
        df_ndemand, "N Demand", "$kg ha^{-1} yr^{-1}$", "blue"
    )
    chart_pdemand_line = _create_line_chart(
        df_pdemand, "P Demand", "$kg ha^{-1} yr^{-1}$", "green"
    )
    chart_kdemand_line = _create_line_chart(
        df_kdemand, "K Demand", "$kg ha^{-1} yr^{-1}$", "orange"
    )

    return (
        chart_wt,
        chart_growth,
        chart_vol_bar,
        chart_vol_inc,
        chart_volume,
        chart_logpulp,
        chart_lmass_box,
        chart_lmass_line,
        chart_leaf_canopy,
        chart_ns,
        chart_phys_r,
        chart_ns_growth,
        chart_ndemand,
        chart_pdemand,
        chart_kdemand,
        chart_ndemand_line,
        chart_pdemand_line,
        chart_kdemand_line,
    )


def hydrology(
    data_store: load_output.OutputDataStore,
    stand_id: load_output.StandID,
    scenario_id: load_output.ScenarioID,
) -> tuple:
    cols = np.shape(_get_var(data_store, "/strip/dwtyr", stand_id, scenario_id))[1]

    wt = np.mean(_get_var(data_store, "/strip/dwtyr", stand_id, scenario_id), axis=0)
    sd = np.std(_get_var(data_store, "/strip/dwtyr", stand_id, scenario_id), axis=0)
    wtgs = np.mean(
        _get_var(data_store, "/strip/dwtyr_growingseason", stand_id, scenario_id), axis=0
    )
    sdgs = np.std(
        _get_var(data_store, "/strip/dwtyr_growingseason"), stand_id, scenario_id), axis=0
    )
    wtls = np.mean(_get_var(data_store, "/strip/dwtyr_latesummer"), stand_id, scenario_id), axis=0)
    sdls = np.std(_get_var(data_store, "/strip/dwtyr_latesummer"), stand_id, scenario_id), axis=0)

    wt_ts = np.mean(_get_var(data_store, "/strip/dwt"), stand_id, scenario_id), axis=1)
    days = len(wt_ts)

    runoff_total = np.cumsum(_get_var(data_store, "/strip/roff", stand_id, scenario_id))
    runoff_west = np.cumsum(_get_var(data_store, "/strip/roffwest", stand_id, scenario_id))
    runoff_east = np.cumsum(_get_var(data_store, "/strip/roffeast", stand_id, scenario_id))
    runoff_surface = np.cumsum(
        np.mean(_get_var(data_store, "/strip/surfacerunoff"), stand_id, scenario_id), axis=1)
    )

    deltas = _get_var(data_store, "/strip/deltas", stand_id, scenario_id)[1:, :] * 1000.0
    df_deltas = pd.DataFrame(data=deltas, columns=np.arange(cols))

    ET = _get_var(data_store, "/cpy/ET_yr", stand_id, scenario_id)[1:, :] * 1000.0
    df_ET = pd.DataFrame(data=ET, columns=np.arange(cols))

    transpi = _get_var(data_store, "/cpy/transpi_yr", stand_id, scenario_id)[1:, :] * 1000.0
    df_transpi = pd.DataFrame(data=transpi, columns=np.arange(cols))

    efloor = _get_var(data_store, "/cpy/efloor_yr", stand_id, scenario_id)[1:, :] * 1000.0
    df_efloor = pd.DataFrame(data=efloor, columns=np.arange(cols))

    swe = _get_var(data_store, "/cpy/SWEmax", stand_id, scenario_id)[1:, :]
    df_swe = pd.DataFrame(data=swe, columns=np.arange(cols))

    interc = _get_var(data_store, "/cpy/interc_yr", stand_id, scenario_id)[1:, :] * 1000.0
    df_interc = pd.DataFrame(data=interc, columns=np.arange(cols))

    chart_wt_annual = _create_profile_line(wt, "Water Table - Annual", "WT (m)", "blue")
    chart_wt_gs = _create_profile_line(
        wtgs, "Water Table - Growing Season", "WT (m)", "green"
    )
    chart_wt_ls = _create_profile_line(
        wtls, "Water Table - Late Summer", "WT (m)", "orange"
    )

    df_wt_ts = pd.DataFrame({"time": range(days), "wt": wt_ts})
    chart_wt_ts = (
        alt.Chart(df_wt_ts)
        .mark_line(color="green")
        .encode(
            x="time",
            y=alt.Y("wt", scale=alt.Scale(zero=False), title="WT (m)"),
        )
        .properties(title="Water Table Time Series")
    ) + (
        alt.Chart(pd.DataFrame({"y": [-0.35]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(y="y")
    )

    df_runoff = pd.DataFrame(
        {
            "time": range(len(runoff_total)),
            "total": runoff_total * 1000.0,
            "west": runoff_west * 1000.0,
            "east": runoff_east * 1000.0,
            "surface": runoff_surface * 1000.0,
        }
    )

    chart_runoff = (
        alt.Chart(df_runoff.melt(id_vars="time", var_name="type", value_name="runoff"))
        .mark_line(opacity=0.7)
        .encode(
            x="time",
            y=alt.Y("runoff", title="mm"),
            color=alt.Color(
                "type",
                scale=alt.Scale(
                    domain=["total", "west", "east", "surface"],
                    range=["blue", "green", "red", "orange"],
                ),
            ),
        )
        .properties(title="Cumulative Runoff")
    )

    chart_deltas = _create_boxplot(
        df_deltas, "Through Soil Surface", "Water flux mm $yr^{-1}$", "blue"
    )
    chart_et = _create_boxplot(df_ET, "ET", "", "green")
    chart_transpi = _create_boxplot(df_transpi, "Transpiration", "", "orange")
    chart_efloor = _create_boxplot(
        df_efloor, "Soil Evaporation", "Water flux mm $yr^{-1}$", "blue"
    )
    chart_swe = _create_boxplot(df_swe, "Max Snow Water Equivalent", "", "green")
    chart_interc = _create_boxplot(df_interc, "Mean Interception Storage", "", "orange")

    return (
        chart_wt_annual,
        chart_wt_gs,
        chart_wt_ls,
        chart_wt_ts,
        chart_runoff,
        chart_deltas,
        chart_et,
        chart_transpi,
        chart_efloor,
        chart_swe,
        chart_interc,
    )


def mass(vars: list[nc_utils.NetcdfVariableValue], scen: int = 0) -> tuple:
    cols = np.shape(_get_var(data_store, "/strip/dwtyr")[scen])[1]

    wtls = np.mean(_get_var(data_store, "/strip/dwtyr_latesummer"), stand_id, scenario_id), axis=0)
    sdls = np.std(_get_var(data_store, "/strip/dwtyr_latesummer"), stand_id, scenario_id), axis=0)

    litter = (
        _get_var(data_store, "/groundvegetation/ds_litterfall"), stand_id, scenario_id) / 10000.0
        + _get_var(data_store, "/groundvegetation/h_litterfall"), stand_id, scenario_id) / 10000.0
        + _get_var(data_store, "/groundvegetation/s_litterfall"), stand_id, scenario_id) / 10000.0
        + _get_var(data_store, "/stand/nonwoodylitter"), stand_id, scenario_id) / 10000.0
        + _get_var(data_store, "/stand/woodylitter"), stand_id, scenario_id) / 10000.0
    )

    soil = _get_var(data_store, "/esom/Mass/out"), stand_id, scenario_id) / 10000.0 * -1 + litter
    soilout = _get_var(data_store, "/esom/Mass/out"), stand_id, scenario_id) / 10000.0 * -1

    df_soil = pd.DataFrame(data=soil, columns=np.arange(cols))

    esoms = ["L0L", "L0W", "LL", "LW", "FL", "FW", "H", "P1", "P2", "P3"]
    inipeat = np.zeros(cols)
    for sto in esoms[7:]:
        inipeat += _get_var(data_store, f"/esom/Mass/{sto}", stand_id, scenario_id)[0, :] / 10000.0
    endpeat = np.zeros(cols)
    for sto in esoms[7:]:
        endpeat += _get_var(data_store, f"/esom/Mass/{sto}", stand_id, scenario_id)[-1, :] / 10000.0
    inimor = np.zeros(cols)
    for sto in esoms[:7]:
        inimor += _get_var(data_store, f"/esom/Mass/{sto}", stand_id, scenario_id)[0, :] / 10000.0
    endmor = np.zeros(cols)
    for sto in esoms[:7]:
        endmor += _get_var(data_store, f"/esom/Mass/{sto}", stand_id, scenario_id)[-1, :] / 10000.0

    df_peat_mor = pd.DataFrame(
        {
            "peat initial": inipeat,
            "mor initial": inimor,
            "peat end": endpeat,
            "mor end": endmor,
        },
        index=range(cols),
    )

    df_litter = pd.DataFrame(data=litter, columns=np.arange(cols))

    gv = _get_var(data_store, "/groundvegetation/gv_tot"), stand_id, scenario_id) / 10000.0
    grgv = np.diff(gv, axis=0)
    df_grgv = pd.DataFrame(data=grgv, columns=np.arange(cols))

    stand = _get_var(data_store, "/stand/biomass"), stand_id, scenario_id) / 10000.0
    gr_stand = np.diff(stand, axis=0)
    df_gr_stand = pd.DataFrame(data=gr_stand, columns=np.arange(cols))

    site = gr_stand + grgv + soilout[1:, :] + litter[1:, :]
    df_site = pd.DataFrame(data=site, columns=np.arange(cols))

    leaflitter = (
        _get_var(data_store, "/stand/nonwoodylitter"), stand_id, scenario_id)
        - _get_var(data_store, "/stand/finerootlitter"), stand_id, scenario_id)
    )
    df_leaflitter = pd.DataFrame(data=leaflitter, columns=np.arange(cols))

    finerootlitter = _get_var(data_store, "/stand/finerootlitter"), stand_id, scenario_id)
    df_finerootlitter = pd.DataFrame(data=finerootlitter, columns=np.arange(cols))

    woodylitter = _get_var(data_store, "/stand/woodylitter"), stand_id, scenario_id)
    df_woodylitter = pd.DataFrame(data=woodylitter, columns=np.arange(cols))

    gvlitter = (
        _get_var(data_store, "/groundvegetation/ds_litterfall"), stand_id, scenario_id)
        + _get_var(data_store, "/groundvegetation/h_litterfall"), stand_id, scenario_id)
        + _get_var(data_store, "/groundvegetation/s_litterfall"), stand_id, scenario_id)
    )
    df_gvlitter = pd.DataFrame(data=gvlitter, columns=np.arange(cols))

    out = _get_var(data_store, "/esom/Mass/out"), stand_id, scenario_id) / 10000.0 * -1
    df_out = pd.DataFrame(data=out, columns=np.arange(cols))

    chart_wt = _create_profile_line(
        wtls, "Water Table - Late Summer", "WT (m)", "orange"
    )

    chart_soil = _create_boxplot(
        df_soil, "Soil Mass Change", "$kg m^{-2} yr^{-1}$", "blue"
    )

    chart_peat_mor = _create_stacked_bar(
        df_peat_mor, "Organic Soil Mass", "$kg m^{-2}$"
    )

    chart_litter = _create_boxplot(
        df_litter, "Litter Input", "$kg m^{-2} yr^{-1}$", "orange"
    )

    chart_gv = _create_boxplot(
        df_grgv, "Ground Vegetation Biomass Change", "$kg m^{-2} yr^{-1}$", "orange"
    )

    chart_stand = _create_boxplot(
        df_gr_stand, "Stand Biomass Change", "$kg m^{-2} yr^{-1}$", "green"
    )

    chart_site = _create_boxplot(
        df_site, "Site Mass Change", "$kg m^{-2} yr^{-1}$", "green"
    )

    chart_leaflitter = _create_boxplot(df_leaflitter, "Leaf Litter", "", "blue")

    chart_finerootlitter = _create_boxplot(
        df_finerootlitter, "Fineroot Litter", "", "orange"
    )

    chart_woodylitter = _create_boxplot(df_woodylitter, "Woody Litter", "", "brown")

    chart_gvlitter = _create_boxplot(df_gvlitter, "Ground Vegetation Litter", "", "red")

    chart_out = _create_boxplot(df_out, "Soil Mass Loss", "", "grey")

    chart_peat_mor_change = _create_line_chart(
        pd.DataFrame(
            {"peat": np.diff(inipeat + inimor), "mor": np.diff(endpeat + endmor)}
        ),
        "Organic Soil Mass Change",
        "$kg m^{-2} yr^{-1}$",
    )

    chart_out_line = _create_line_chart(
        df_out, "Soil Mass Loss", "$kg m^{-2}$", "orange"
    )

    chart_litter_line = _create_line_chart(
        df_litter, "Litterfall", "$kg m^{-2}$", "orange"
    )

    return (
        chart_wt,
        chart_soil,
        chart_peat_mor,
        chart_litter,
        chart_gv,
        chart_stand,
        chart_site,
        chart_leaflitter,
        chart_finerootlitter,
        chart_woodylitter,
        chart_gvlitter,
        chart_out,
        chart_peat_mor_change,
        chart_out_line,
        chart_litter_line,
    )


def carbon(vars: list[nc_utils.NetcdfVariableValue], scen: int = 0) -> tuple:
    cols = np.shape(_get_var(data_store, "/strip/dwtyr")[scen])[1]
    mass_to_c = 0.5

    litter = (
        _get_var(data_store, "/groundvegetation/ds_litterfall"), stand_id, scenario_id) / 10000.0
        + _get_var(data_store, "/groundvegetation/h_litterfall"), stand_id, scenario_id) / 10000.0
        + _get_var(data_store, "/groundvegetation/s_litterfall"), stand_id, scenario_id) / 10000.0
        + _get_var(data_store, "/stand/nonwoodylitter"), stand_id, scenario_id) / 10000.0
        + _get_var(data_store, "/stand/woodylitter"), stand_id, scenario_id) / 10000.0
    ) * mass_to_c

    wt = np.mean(_get_var(data_store, "/strip/dwtyr"), stand_id, scenario_id), axis=0)
    wtls = np.mean(_get_var(data_store, "/strip/dwtyr_latesummer"), stand_id, scenario_id), axis=0)
    sdls = np.std(_get_var(data_store, "/strip/dwtyr_latesummer"), stand_id, scenario_id), axis=0)

    elevation = np.array(_get_var(data_store, "/strip/elevation"))
    h = elevation + wtls

    lmwtoditch = _get_var(data_store, "/balance/C/LMWdoc_to_water"), stand_id, scenario_id) * -1
    df_lmwtoditch = pd.DataFrame(data=lmwtoditch, columns=np.arange(cols))

    hmwtoditch = _get_var(data_store, "/balance/C/HMW_to_water"), stand_id, scenario_id) * -1
    df_hmwtoditch = pd.DataFrame(data=hmwtoditch, columns=np.arange(cols))

    lmwtoatm = _get_var(data_store, "/balance/C/LMWdoc_to_atm"), stand_id, scenario_id) * -1
    df_lmwtoatm = pd.DataFrame(data=lmwtoatm, columns=np.arange(cols))

    hmwtoatm = _get_var(data_store, "/balance/C/HMW_to_atm"), stand_id, scenario_id) * -1
    df_hmwtoatm = pd.DataFrame(data=hmwtoatm, columns=np.arange(cols))

    co2 = _get_var(data_store, "/balance/C/co2c_release"), stand_id, scenario_id) * -1
    df_co2 = pd.DataFrame(data=co2, columns=np.arange(cols))

    ch4 = _get_var(data_store, "/balance/C/ch4c_release"), stand_id, scenario_id) * -1
    df_ch4 = pd.DataFrame(data=ch4, columns=np.arange(cols))

    standl = _get_var(data_store, "/balance/C/stand_litter_in"), stand_id, scenario_id)
    df_standl = pd.DataFrame(data=standl, columns=np.arange(cols))

    gvl = _get_var(data_store, "/balance/C/gv_litter_in"), stand_id, scenario_id)
    df_gvl = pd.DataFrame(data=gvl, columns=np.arange(cols))

    soilc = _get_var(data_store, "/balance/C/soil_c_balance_c"), stand_id, scenario_id)
    df_soilc = pd.DataFrame(data=soilc, columns=np.arange(cols))

    soilco2 = _get_var(data_store, "/balance/C/soil_c_balance_co2eq"), stand_id, scenario_id)
    df_soilco2 = pd.DataFrame(data=soilco2, columns=np.arange(cols))

    standc = _get_var(data_store, "/balance/C/stand_c_balance_c"), stand_id, scenario_id)
    df_standc = pd.DataFrame(data=standc, columns=np.arange(cols))

    standco2 = _get_var(data_store, "/balance/C/stand_c_balance_co2eq"), stand_id, scenario_id)
    df_standco2 = pd.DataFrame(data=standco2, columns=np.arange(cols))

    chart_wt = _create_profile_line(wt, "Water Table - Annual", "WT (m)", "blue")

    chart_wt_elev = _create_profile_line(
        h, "Water Table with Elevation", "WT (m)", "blue", show_drain_line=False
    )

    chart_lmwtoditch = _create_boxplot(
        df_lmwtoditch, "LMW to Ditch", "kg $ha^{-1} yr^{-1}$", "brown"
    )
    chart_hmwtoditch = _create_boxplot(
        df_hmwtoditch, "HMW to Ditch", "kg $ha^{-1} yr^{-1}$", "brown"
    )
    chart_lmwtoatm = _create_boxplot(
        df_lmwtoatm, "LMW to Atmosphere", "kg $ha^{-1} yr^{-1}$", "orange"
    )
    chart_hmwtoatm = _create_boxplot(
        df_hmwtoatm, "HMW to Atmosphere", "kg $ha^{-1} yr^{-1}$", "orange"
    )
    chart_co2 = _create_boxplot(
        df_co2, "$CO_2C$ to Atmosphere", "kg $ha^{-1} yr^{-1}$", "grey"
    )
    chart_ch4 = _create_boxplot(
        df_ch4, "$CH_4C$ to Atmosphere", "kg $ha^{-1} yr^{-1}$", "grey"
    )
    chart_standl = _create_boxplot(
        df_standl, "Stand Litter", "kg $ha^{-1} yr^{-1}$", "green"
    )
    chart_gvl = _create_boxplot(
        df_gvl, "Ground Vegetation Litter", "kg $ha^{-1} yr^{-1}$", "green"
    )
    chart_soilc = _create_boxplot(
        df_soilc, "Soil C Balance in C", "kg $ha^{-1} yr^{-1}$", "black"
    )
    chart_soilco2 = _create_boxplot(
        df_soilco2, "Soil C Balance in $CO_2$ eq", "kg $ha^{-1} yr^{-1}$", "black"
    )
    chart_standc = _create_boxplot(
        df_standc, "Stand C Balance in C", "kg $ha^{-1} yr^{-1}$", "green"
    )
    chart_standco2 = _create_boxplot(
        df_standco2, "Stand C Balance in $CO_2$ eq", "kg $ha^{-1} yr^{-1}$", "green"
    )

    return (
        chart_wt,
        chart_wt_elev,
        chart_lmwtoditch,
        chart_hmwtoditch,
        chart_lmwtoatm,
        chart_hmwtoatm,
        chart_co2,
        chart_ch4,
        chart_standl,
        chart_gvl,
        chart_soilc,
        chart_soilco2,
        chart_standc,
        chart_standco2,
    )


def nutrient_balance(
    vars: list[nc_utils.NetcdfVariableValue],
    substance: str,
    scen: int = 0,
) -> tuple:
    cols = np.shape(_get_var(data_store, "/strip/dwtyr")[scen])[1]

    wt = np.mean(_get_var(data_store, "/strip/dwtyr"), stand_id, scenario_id), axis=0)
    wtls = np.mean(_get_var(data_store, "/strip/dwtyr_latesummer"), stand_id, scenario_id), axis=0)
    sdls = np.std(_get_var(data_store, "/strip/dwtyr_latesummer"), stand_id, scenario_id), axis=0)

    elevation = np.array(_get_var(data_store, "/strip/elevation"))
    h = elevation + wtls

    towater = _get_var(data_store, f"/balance/{substance}/to_water"), stand_id, scenario_id)
    df_towater = pd.DataFrame(data=towater, columns=np.arange(cols))

    brl = _get_var(data_store, f"/balance/{substance}/decomposition_below_root_lyr")[
        scen, :, :
    ]
    df_brl = pd.DataFrame(data=brl, columns=np.arange(cols))

    de = _get_var(data_store, f"/balance/{substance}/decomposition_tot"), stand_id, scenario_id)
    df_de = pd.DataFrame(data=de, columns=np.arange(cols))

    dert = _get_var(data_store, f"/balance/{substance}/decomposition_root_lyr")[
        scen, :, :
    ]
    df_dert = pd.DataFrame(data=dert, columns=np.arange(cols))

    supply = (
        _get_var(data_store, f"/balance/{substance}/decomposition_root_lyr"), stand_id, scenario_id)
        + _get_var(data_store, f"/balance/{substance}/deposition"), stand_id, scenario_id)
        + _get_var(data_store, f"/balance/{substance}/fertilization_release")[
            scen, :, :
        ]
    )
    df_supply = pd.DataFrame(data=supply, columns=np.arange(cols))

    fert = _get_var(data_store, f"/balance/{substance}/fertilization_release")[
        scen, :, :
    ]
    df_fert = pd.DataFrame(data=fert, columns=np.arange(cols))

    dem = _get_var(data_store, f"/balance/{substance}/stand_demand"), stand_id, scenario_id)
    df_dem = pd.DataFrame(data=dem, columns=np.arange(cols))

    dem_gv = _get_var(data_store, f"/balance/{substance}/gv_demand"), stand_id, scenario_id)
    df_dem_gv = pd.DataFrame(data=dem_gv, columns=np.arange(cols))

    bal = _get_var(data_store, f"/balance/{substance}/balance_root_lyr"), stand_id, scenario_id)
    df_bal = pd.DataFrame(data=bal, columns=np.arange(cols))

    vg = _get_var(data_store, "/stand/volumegrowth"), stand_id, scenario_id)
    df_vg = pd.DataFrame(data=vg, columns=np.arange(cols))

    chart_wt = _create_profile_line(wt, "Water Table - Annual", "WT (m)", "blue")

    chart_wt_elev = _create_profile_line(
        h,
        f"Water Table with Elevation - {substance}",
        "WT (m)",
        "blue",
        show_drain_line=False,
    )

    chart_towater = _create_boxplot(
        df_towater, f"{substance} to Water", "kg $ha^{-1} yr^{-1}$", "brown"
    )
    chart_brl = _create_boxplot(
        df_brl, f"{substance} Below Root Layer", "kg $ha^{-1} yr^{-1}$", "brown"
    )
    chart_de = _create_boxplot(
        df_de, f"{substance} Release in Decomposition", "kg $ha^{-1} yr^{-1}$", "black"
    )
    chart_dert = _create_boxplot(
        df_dert, f"{substance} Release in Root Layer", "kg $ha^{-1} yr^{-1}$", "black"
    )
    chart_supply = _create_boxplot(
        df_supply, f"{substance} Supply", "kg $ha^{-1} yr^{-1}$", "blue"
    )
    chart_fert = _create_boxplot(
        df_fert, f"{substance} Release in Fertilizers", "kg $ha^{-1} yr^{-1}$", "black"
    )
    chart_dem = _create_boxplot(
        df_dem, f"{substance} Stand Uptake", "kg $ha^{-1} yr^{-1}$", "green"
    )
    chart_dem_gv = _create_boxplot(
        df_dem_gv,
        f"{substance} Ground Vegetation Uptake",
        "kg $ha^{-1} yr^{-1}$",
        "green",
    )
    chart_bal = _create_boxplot(
        df_bal, f"{substance} Balance", "kg $ha^{-1} yr^{-1}$", "green"
    )
    chart_vg = _create_boxplot(df_vg, "Volume Growth", "$m^3 ha^{-1} yr^{-1}$", "green")

    return (
        chart_wt,
        chart_wt_elev,
        chart_towater,
        chart_brl,
        chart_de,
        chart_dert,
        chart_supply,
        chart_fert,
        chart_dem,
        chart_dem_gv,
        chart_bal,
        chart_vg,
    )
