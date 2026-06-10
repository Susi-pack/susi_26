from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float
from pyproj import CRS, Transformer


# --- Sub-dataclasses ---


@dataclass(frozen=True)
class NPK:
    N: float
    P: float
    K: float


@dataclass(frozen=True)
class FieldLayerShare:
    dwarf_shrub: float
    herb: float


# --- Main dataclasses ---


@dataclass(frozen=True)
class Params:
    num_nodes: int = field(doc="Number of computation nodes")
    site_fertility_class: Float[np.ndarray, " n"] = field(
        doc="Site fertility class (1-5)"
    )
    drainage_status: int = field(default=4, doc="Drainage status (1-4)")
    total_turnover_multiplier: float = field(
        default=1.5,
        doc="Multiplier: above-ground turnover -> total turnover (incl. roots)",
    )
    biomass_conversion_factor: float = field(
        default=1.9,
        doc="Multiplier: above-ground biomass -> total biomass",
    )
    clearcut_reduction_factor: float = field(
        default=0.33, doc="Clear-cut reduction factor"
    )

    # Field layer shares per site type
    pine_upland_field_layer_share: FieldLayerShare = field(
        default=FieldLayerShare(dwarf_shrub=0.91, herb=0.09),
        doc="Pine upland: share of dwarf shrubs and herbs in field layer",
    )
    spruce_upland_field_layer_share: FieldLayerShare = field(
        default=FieldLayerShare(dwarf_shrub=0.71, herb=0.29),
        doc="Spruce upland: share of dwarf shrubs and herbs in field layer",
    )
    broadleaved_upland_field_layer_share: FieldLayerShare = field(
        default=FieldLayerShare(dwarf_shrub=0.38, herb=0.62),
        doc="Broadleaved upland: share of dwarf shrubs and herbs in field layer",
    )
    spruce_mire_field_layer_share: FieldLayerShare = field(
        default=FieldLayerShare(dwarf_shrub=0.50, herb=0.50),
        doc="Spruce mire: share of dwarf shrubs and herbs in field layer",
    )
    pine_bog_field_layer_share: FieldLayerShare = field(
        default=FieldLayerShare(dwarf_shrub=0.90, herb=0.10),
        doc="Pine bog: share of dwarf shrubs and herbs in field layer",
    )

    # Nutrient concentrations per vegetation type [mg/g]
    dwarf_shrub_nutrient_concentration: NPK = field(
        default=NPK(N=12.0, P=1.0, K=4.7),
        doc="Dwarf shrub nutrient concentration [mg/g]",
    )
    herb_nutrient_concentration: NPK = field(
        default=NPK(N=18.0, P=2.0, K=15.1),
        doc="Herb nutrient concentration [mg/g]",
    )
    upland_moss_nutrient_concentration: NPK = field(
        default=NPK(N=12.5, P=1.4, K=4.3),
        doc="Upland moss nutrient concentration [mg/g]",
    )
    sphagnum_nutrient_concentration: NPK = field(
        default=NPK(N=6.0, P=1.4, K=4.3),
        doc="Sphagnum nutrient concentration [mg/g]",
    )

    # Annual litterfall rates (fraction of living biomass lost as litter)
    dwarf_shrub_litterfall_rate: float = field(
        default=0.33, doc="Dwarf shrub litterfall rate"
    )
    herb_litterfall_rate: float = field(default=1.0, doc="Herb litterfall rate")
    upland_moss_litterfall_rate: float = field(
        default=0.3, doc="Upland moss litterfall rate"
    )
    sphagnum_litterfall_rate: float = field(default=0.3, doc="Sphagnum litterfall rate")

    # Green mass fraction of total biomass
    dwarf_shrub_green_mass_fraction: float = field(
        default=0.2, doc="Dwarf shrub green mass fraction"
    )
    herb_green_mass_fraction: float = field(default=0.5, doc="Herb green mass fraction")
    upland_moss_green_mass_fraction: float = field(
        default=0.3, doc="Upland moss green mass fraction"
    )
    sphagnum_green_mass_fraction: float = field(
        default=0.3, doc="Sphagnum green mass fraction"
    )

    # Retranslocation fractions (nutrients retracted before litterfall), [kg/kg, dimensionless]
    dwarf_shrub_retranslocation: NPK = field(
        default=NPK(N=0.69, P=0.73, K=0.87),
        doc="Dwarf shrub retranslocation fractions",
    )
    herb_retranslocation: NPK = field(
        default=NPK(N=0.69, P=0.73, K=0.87),
        doc="Herb retranslocation fractions",
    )
    upland_moss_retranslocation: NPK = field(
        default=NPK(N=0.5, P=0.5, K=0.5),
        doc="Upland moss retranslocation fractions",
    )
    sphagnum_retranslocation: NPK = field(
        default=NPK(N=0.5, P=0.5, K=0.5),
        doc="Sphagnum retranslocation fractions",
    )


@dataclass(frozen=True)
class ComputedConstants:
    dem: Float[np.ndarray, " n"] = field(doc="Surface elevation [m asl]")
    latitude_wgs84: float = field(doc="Latitude in WGS84 [degrees]")
    longitude_wgs84: float = field(doc="Longitude in WGS84 [degrees]")
    ix_spruce_mire: tuple = field(
        doc="Boolean/indices where species == 2 (spruce mire)"
    )
    ix_pine_bog: tuple = field(doc="Boolean/indices where species == 1 (pine bog)")
    ix_open_peat: tuple = field(
        doc="Boolean/indices where species == 4 (open peatland)"
    )
    dominant_tree_species: Float[np.ndarray, " n"]


@dataclass(frozen=True)
class State:
    gv_tot: Float[np.ndarray, " n"] = field(doc="Total ground vegetation mass [kg/ha]")
    n_gv: Float[np.ndarray, " n"] = field(
        doc="N in ground vegetation [kg/ha] — persisted for delta computation"
    )
    p_gv: Float[np.ndarray, " n"] = field(
        doc="P in ground vegetation [kg/ha] — persisted for delta computation"
    )
    k_gv: Float[np.ndarray, " n"] = field(
        doc="K in ground vegetation [kg/ha] — persisted for delta computation"
    )


@dataclass(frozen=True)
class Outputs:
    gv_change: Float[np.ndarray, " n"] = field(
        doc="Biomass change during timestep [kg/ha/yr]"
    )
    gv_field: Float[np.ndarray, " n"] = field(doc="Field layer vegetation mass [kg/ha]")
    gv_bot: Float[np.ndarray, " n"] = field(doc="Bottom layer vegetation mass [kg/ha]")
    gv_leafmass: Float[np.ndarray, " n"] = field(
        doc="Leaf mass in ground vegetation [kg/ha]"
    )
    ds_litterfall: Float[np.ndarray, " n"] = field(
        doc="Dwarf shrub litterfall [kg/ha/yr]"
    )
    h_litterfall: Float[np.ndarray, " n"] = field(doc="Herb litterfall [kg/ha/yr]")
    s_litterfall: Float[np.ndarray, " n"] = field(doc="Sphagnum litterfall [kg/ha/yr]")
    n_litter_nw: Float[np.ndarray, " n"] = field(doc="N in non-woody litter [kg/ha/yr]")
    p_litter_nw: Float[np.ndarray, " n"] = field(doc="P in non-woody litter [kg/ha/yr]")
    k_litter_nw: Float[np.ndarray, " n"] = field(doc="K in non-woody litter [kg/ha/yr]")
    n_litter_w: Float[np.ndarray, " n"] = field(doc="N in woody litter [kg/ha/yr]")
    p_litter_w: Float[np.ndarray, " n"] = field(doc="P in woody litter [kg/ha/yr]")
    k_litter_w: Float[np.ndarray, " n"] = field(doc="K in woody litter [kg/ha/yr]")
    nup: Float[np.ndarray, " n"] = field(doc="Total N uptake [kg/ha]")
    pup: Float[np.ndarray, " n"] = field(doc="Total P uptake [kg/ha]")
    kup: Float[np.ndarray, " n"] = field(doc="Total K uptake [kg/ha]")
    nonwoodylitter: Float[np.ndarray, " n"] = field(
        doc="Non-woody litterfall [kg/ha/yr]"
    )
    woodylitter: Float[np.ndarray, " n"] = field(doc="Woody litterfall [kg/ha/yr]")


@dataclass(frozen=True)
class Inputs:
    ts: Float[np.ndarray, " n"] = field(doc="Temperature sum [degree days]")
    vol: Float[np.ndarray, " n"] = field(doc="Stem volume [m3/ha]")
    stems: Float[np.ndarray, " n"] = field(doc="Number of stems [stems/ha]")
    ba: Float[np.ndarray, " n"] = field(doc="Basal area [m2/ha]")
    age: Float[np.ndarray, " n"] = field(doc="Stand age [years]")


def compute_constants(
    params: Params,
    lat: float,
    lon: float,
    dominant_tree_species: Float[np.ndarray, " n"],
) -> ComputedConstants:
    dem = np.ones(params.num_nodes) * 80.0

    inProj = CRS("epsg:3067")
    outProj = CRS("epsg:4326")
    transformer = Transformer.from_crs(inProj, outProj)

    # Original (lat,lon) is in EPSG:3067 [m]
    latitude_wgs84, longitude_wgs84 = transformer.transform(lon, lat)

    return ComputedConstants(
        dem=dem,
        latitude_wgs84=latitude_wgs84,
        longitude_wgs84=longitude_wgs84,
        ix_spruce_mire=np.where(np.equal(dominant_tree_species, 2)),
        ix_pine_bog=np.where(np.equal(dominant_tree_species, 1)),
        ix_open_peat=np.where(np.equal(dominant_tree_species, 4)),
        dominant_tree_species=dominant_tree_species,
    )


def assemble_inputs(
    ts: Float[np.ndarray, " n"],
    vol: Float[np.ndarray, " n"],
    stems: Float[np.ndarray, " n"],
    ba: Float[np.ndarray, " n"],
    age: Float[np.ndarray, " n"],
) -> Inputs:
    return Inputs(ts=ts, vol=vol, stems=stems, ba=ba, age=age)


def compute_initial_state(
    params: Params, computed_constants: ComputedConstants
) -> State:
    n = params.num_nodes
    return State(
        gv_tot=np.zeros(n),
        n_gv=np.zeros(n),
        p_gv=np.zeros(n),
        k_gv=np.zeros(n),
    )


def _fill_site_nutrients(
    ix,
    gv_tot,
    gv_field,
    gv_bot,
    ds_litterfall,
    h_litterfall,
    s_litterfall,
    gv_leafmass,
    n_gv,
    p_gv,
    k_gv,
    n_litter_nw,
    p_litter_nw,
    k_litter_nw,
    n_litter_w,
    p_litter_w,
    k_litter_w,
    params,
    fl_ds: float,
    fl_h: float,
):
    gv_field[ix] = np.minimum(gv_tot[ix], gv_field[ix])
    gv_bot[ix] = np.minimum(gv_tot[ix], gv_bot[ix])
    gv_field[ix] = np.maximum(gv_field[ix], gv_tot[ix] - gv_bot[ix])

    field_layer = gv_field[ix]
    bottom = gv_bot[ix]
    tot_minus_bot = gv_tot[ix] - gv_bot[ix]

    nut_ds = params.dwarf_shrub_nutrient_concentration
    nut_h = params.herb_nutrient_concentration
    nut_s = params.sphagnum_nutrient_concentration
    retrans_ds = params.dwarf_shrub_retranslocation
    retrans_h = params.herb_retranslocation
    retrans_s = params.sphagnum_retranslocation

    ds_lf = (
        fl_ds
        * tot_minus_bot
        * params.dwarf_shrub_litterfall_rate
        * params.total_turnover_multiplier
    )
    h_lf = (
        fl_h
        * tot_minus_bot
        * params.herb_litterfall_rate
        * params.total_turnover_multiplier
    )
    s_lf = bottom * params.sphagnum_litterfall_rate

    ds_litterfall[ix] = ds_lf
    h_litterfall[ix] = h_lf
    s_litterfall[ix] = s_lf

    gv_leafmass[ix] = (
        fl_ds * tot_minus_bot * params.dwarf_shrub_green_mass_fraction
        + fl_h * tot_minus_bot * params.herb_green_mass_fraction
        + bottom * params.sphagnum_green_mass_fraction
    )

    conv = params.biomass_conversion_factor

    n_gv[ix] = (
        field_layer * (fl_ds * nut_ds.N + fl_h * nut_h.N) * 1e-3 * conv
        + bottom * nut_s.N * 1e-3
    )
    p_gv[ix] = (
        field_layer * (fl_ds * nut_ds.P + fl_h * nut_h.P) * 1e-3 * conv
        + bottom * nut_s.P * 1e-3
    )
    k_gv[ix] = (
        field_layer * (fl_ds * nut_ds.K + fl_h * nut_h.K) * 1e-3 * conv
        + bottom * nut_s.K * 1e-3
    )

    n_litter_nw[ix] = (
        ds_lf * nut_ds.N * 1e-3 * (1.0 - retrans_ds.N) * 0.25
        + h_lf * nut_h.N * 1e-3 * (1.0 - retrans_h.N)
        + s_lf * nut_s.N * 1e-3 * (1.0 - retrans_s.N)
    )
    p_litter_nw[ix] = (
        ds_lf * nut_ds.P * 1e-3 * (1.0 - retrans_ds.P) * 0.25
        + h_lf * nut_h.P * 1e-3 * (1.0 - retrans_h.P)
        + s_lf * nut_s.P * 1e-3 * (1.0 - retrans_s.P)
    )
    k_litter_nw[ix] = (
        ds_lf * nut_ds.K * 1e-3 * (1.0 - retrans_ds.K) * 0.25
        + h_lf * nut_h.K * 1e-3 * (1.0 - retrans_h.K)
        + s_lf * nut_s.K * 1e-3 * (1.0 - retrans_s.K)
    )

    n_litter_w[ix] = ds_lf * nut_ds.N * 1e-3 * (1.0 - retrans_ds.N) * 0.75
    p_litter_w[ix] = ds_lf * nut_ds.P * 1e-3 * (1.0 - retrans_ds.P) * 0.75
    k_litter_w[ix] = ds_lf * nut_ds.K * 1e-3 * (1.0 - retrans_ds.K) * 0.75


def run_timestep(
    params: Params,
    computed_constants: ComputedConstants,
    input: Inputs,
    state: State,
) -> tuple[State, Outputs]:
    n = params.num_nodes
    cc = computed_constants

    gv_tot = np.zeros(n)
    gv_field = np.zeros(n)
    gv_bot = np.zeros(n)
    gv_leafmass = np.zeros(n)
    ds_litterfall = np.zeros(n)
    h_litterfall = np.zeros(n)
    s_litterfall = np.zeros(n)

    n_litter_nw = np.zeros(n)
    p_litter_nw = np.zeros(n)
    k_litter_nw = np.zeros(n)

    n_litter_w = np.zeros(n)
    p_litter_w = np.zeros(n)
    k_litter_w = np.zeros(n)

    n_gv = np.zeros(n)
    p_gv = np.zeros(n)
    k_gv = np.zeros(n)

    lon = cc.longitude_wgs84
    lat = cc.latitude_wgs84
    drain = params.drainage_status

    # --- Spruce mire ---
    ix = cc.ix_spruce_mire
    dem_ix = cc.dem[ix]
    vol_ix = input.vol[ix]
    stems_ix = input.stems[ix]
    ba_ix = input.ba[ix]
    age_ix = input.age[ix]
    sfc_ix = params.site_fertility_class[ix]
    ts_ix = input.ts[ix]

    gv_tot[ix] = (
        np.square(
            35.52
            + 0.001 * lon * dem_ix
            - 1.1 * drain**2
            - 2e-5 * vol_ix * stems_ix
            + 4e-5 * stems_ix * age_ix
            + 0.139 * lon * drain
        )
        - 0.5
        + 116.54
    )
    gv_bot[ix] = (
        np.square(
            -3.182
            + 0.022 * lat * lon
            + 2e-4 * dem_ix * age_ix
            - 0.077 * sfc_ix * lon
            - 0.003 * lon * vol_ix
            + 2e-4 * np.square(vol_ix)
        )
        - 0.5
        + 98.10
    )
    gv_field[ix] = (
        np.square(
            23.24
            - 1.163 * drain**2
            + 1.515 * sfc_ix * drain
            - 2e-5 * vol_ix * stems_ix
            + 8e-5 * ts_ix * age_ix
            + 1e-5 * stems_ix * dem_ix
        )
        - 0.5
        + 162.58
    )

    _fill_site_nutrients(
        ix,
        gv_tot,
        gv_field,
        gv_bot,
        ds_litterfall,
        h_litterfall,
        s_litterfall,
        gv_leafmass,
        n_gv,
        p_gv,
        k_gv,
        n_litter_nw,
        p_litter_nw,
        k_litter_nw,
        n_litter_w,
        p_litter_w,
        k_litter_w,
        params,
        params.spruce_mire_field_layer_share.dwarf_shrub,
        params.spruce_mire_field_layer_share.herb,
    )

    # --- Pine bog ---
    ix = cc.ix_pine_bog
    dem_ix = cc.dem[ix]
    vol_ix = input.vol[ix]
    stems_ix = input.stems[ix]
    ba_ix = input.ba[ix]
    age_ix = input.age[ix]
    sfc_ix = params.site_fertility_class[ix]
    ts_ix = input.ts[ix]

    gv_tot[ix] = (
        np.square(
            50.098
            + 0.005 * lon * dem_ix
            - 1e-5 * vol_ix * stems_ix
            + 0.026 * sfc_ix * age_ix
            - 1e-4 * dem_ix * ts_ix
            - 0.014 * vol_ix * drain
        )
        - 0.5
        + 167.40
    )
    gv_bot[ix] = (
        np.square(
            31.809
            + 0.008 * lon * dem_ix
            - 3e-4 * stems_ix * ba_ix
            + 6e-5 * stems_ix * age_ix
            - 0.188 * dem_ix
        )
        - 0.5
        + 222.22
    )
    gv_field[ix] = (
        np.square(
            48.12
            - 1e-5 * ts_ix**2
            + 0.013 * sfc_ix * age_ix
            - 0.04 * vol_ix * drain
            + 0.026 * sfc_ix * vol_ix
        )
        - 0.5
        + 133.26
    )

    _fill_site_nutrients(
        ix,
        gv_tot,
        gv_field,
        gv_bot,
        ds_litterfall,
        h_litterfall,
        s_litterfall,
        gv_leafmass,
        n_gv,
        p_gv,
        k_gv,
        n_litter_nw,
        p_litter_nw,
        k_litter_nw,
        n_litter_w,
        p_litter_w,
        k_litter_w,
        params,
        params.pine_bog_field_layer_share.dwarf_shrub,
        params.pine_bog_field_layer_share.herb,
    )

    # --- Clear-cut reduction ---
    ix_cc = np.where(input.age < 5.0)
    cc_factor = params.clearcut_reduction_factor
    gv_tot[ix_cc] = gv_tot[ix_cc] * cc_factor
    n_gv[ix_cc] = n_gv[ix_cc] * cc_factor
    p_gv[ix_cc] = p_gv[ix_cc] * cc_factor
    k_gv[ix_cc] = k_gv[ix_cc] * cc_factor
    n_litter_nw[ix_cc] = n_litter_nw[ix_cc] * cc_factor
    p_litter_nw[ix_cc] = p_litter_nw[ix_cc] * cc_factor
    k_litter_nw[ix_cc] = k_litter_nw[ix_cc] * cc_factor
    n_litter_w[ix_cc] = n_litter_w[ix_cc] * cc_factor
    p_litter_w[ix_cc] = p_litter_w[ix_cc] * cc_factor
    k_litter_w[ix_cc] = k_litter_w[ix_cc] * cc_factor

    # --- Litterfall totals ---
    nonwoody = ds_litterfall * 0.25 + h_litterfall + s_litterfall
    woody = ds_litterfall * 0.75

    # --- Nutrient uptake ---
    nup_net = np.where(n_gv - state.n_gv > 0.0, n_gv - state.n_gv, 0.0)
    pup_net = np.where(p_gv - state.p_gv > 0.0, p_gv - state.p_gv, 0.0)
    kup_net = np.where(k_gv - state.k_gv > 0.0, k_gv - state.k_gv, 0.0)

    nup = nup_net + n_litter_nw + n_litter_w
    pup = pup_net + p_litter_nw + p_litter_w
    kup = kup_net + k_litter_nw + k_litter_w

    gv_change = state.gv_tot - gv_tot

    return State(
        gv_tot=gv_tot,
        n_gv=n_gv,
        p_gv=p_gv,
        k_gv=k_gv,
    ), Outputs(
        gv_change=gv_change,
        gv_field=gv_field,
        gv_bot=gv_bot,
        gv_leafmass=gv_leafmass,
        ds_litterfall=ds_litterfall,
        h_litterfall=h_litterfall,
        s_litterfall=s_litterfall,
        n_litter_nw=n_litter_nw,
        p_litter_nw=p_litter_nw,
        k_litter_nw=k_litter_nw,
        n_litter_w=n_litter_w,
        p_litter_w=p_litter_w,
        k_litter_w=k_litter_w,
        nup=nup,
        pup=pup,
        kup=kup,
        nonwoodylitter=nonwoody,
        woodylitter=woody,
    )
