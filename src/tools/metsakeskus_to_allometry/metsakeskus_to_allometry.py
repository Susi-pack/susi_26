"""
Reads Metsäkeskus forest inventory data (.gpkg) in Metsäkeskus's 2026 data
model (v2.1.x: site data in its own growthplacedata layer, tree-stand records
dated by treestand.treedatadate and classified by treestand.treestandclass)
--> Filters the data (more details below)
--> Generates SUSI allometry files

A file in the old, pre-2026 format is refused, with a pointer to
metsakeskus_to_allometry_old_format.py, which still reads it. The two tools are
deliberately independent copies, so the old one can be deleted cleanly once
Metsäkeskus stops distributing the old format.

Generalizes Sandeep's script into a tool, following the conventions
Unlike in that script, there is no representative sampling here, so
every stand that survives filtering is turned into an allometry CSV file.
"""

# %% Imports
import argparse
import dataclasses
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import geopandas as gpd
import pandas as pd
import pyogrio
from pydantic import model_validator
from shapely.geometry import Polygon

from susi.io.extra_pydantic_types import NonNegativeFloat
from susi.io.load_output_data import StandID
from susi.io.project_layout import (
    CONFIG_FILENAME,
    allometry_dir_for_project,
    stand_data_path_for_project,
)
from susi.io.stand_data import (
    SOURCE_CRS,
    StandData,
    StandDataDocument,
    centroid_to_ykj,
    dump_stand_data_document,
)
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerName,
)
from tools.shared_allometry_tool_utils.allometry_generation_defaults import (
    AllometryGenerationDefaults,
)
from tools.shared_allometry_tool_utils.cli_paths import (
    finalize_cli_config,
    resolve_config_path,
)
from tools.shared_allometry_tool_utils.dense_young_stand_scaling import (
    DenseYoungStandScalingConfig,
    basal_area_weighted_mean_diameter,
    decide_scaling_factors,
    stands_the_default_rule_would_scale,
    strata_to_grow_from,
    total_basal_area,
    total_stem_count,
)
from tools.shared_allometry_tool_utils.growth_and_yield_table import (
    build_growth_and_yield_table as build_isolated_growth_and_yield_table,
)
from tools.shared_allometry_tool_utils.growth_and_yield_table import (
    initial_age,
)
from tools.shared_allometry_tool_utils.input_validation import (
    load_config_or_exit,
    load_toml_config,
    make_existing_file_validator,
    valid_existing_directory,
)
from tools.shared_allometry_tool_utils.print_formatting import (
    StandSkipped,
    print_dense_young_stand_warning,
    print_scaled_stands,
    print_section,
    print_skips,
)
from tools.shared_allometry_tool_utils.tree_stratum import (
    ZERO_STRATUM,
    PerSpecies,
    TreeStratum,
)

# %% Constants -- hard-coded, non-negotiable

# Metsäkeskus maingroup code: forest land (excludes agricultural/other land)
MAINGROUP_FOREST_LAND = 1

# 2: Korpi (spruce mire), 3: Räme (pine mire)
# Excluding 1: Kangas (mineral soil)
SUBGROUP_PEATLAND = (2, 3)

# Drained sites:
# 7: ojikko, 8: muuttuma 9: turvekangas
# Excludes code 6 (ojittamaton suo, undrained/pristine mire)
# Reason: SUSI only simulates drained peatlands
DRAINAGESTATE_DRAINED = (7, 8, 9)

# Metsäkeskus tree-species codes -> SUSI's fixed three growth-model slots.
PINE_SPECIES_CODES = frozenset({1})
SPRUCE_SPECIES_CODES = frozenset({2})
DECIDUOUS_SPECIES_CODES = frozenset({3, 4, 5, 6, 7, 8, 9, 15, 20, 29})

# Growth_and_Yield_Table.peat: every stand this tool processes is peatland by
# construction (SUBGROUP_PEATLAND above already restricts to Korpi/Räme).
PEAT = 1

# treestand.treestandclass: what kind of tree-stand record a row is.
# 1 ("inventointipuusto") is an inventory snapshot: the trees as an inventory
# found them on treedatadate, whatever the inventory's source
# (treestanddatasource: field-measured, remote-sensed, mixed, or reported by
# the owner or an authority). 2 and 3 are Metsäkeskus's own projections, grown
# forward to the current date and ten years ahead; 4 is dead trees.
# Only class 1 is an inventory.
TREESTANDCLASS_INVENTORY = 1

# Each layer this tool reads -> the columns it reads from it (the stand
# layer's geometry comes along on top). Reading only these keeps the
# multi-GB read fast, and skips the stand layer's timestamp columns, whose
# occasional leap-second value ("...T20:02:60Z") pandas warns about.
# Also what gpkg_format_problem checks a file for.
LAYER_COLUMNS = {
    "stand": ("standid", "area"),
    "growthplacedata": (
        "standid",
        "maingroup",
        "subgroup",
        "fertilityclass",
        "soiltype",
        "drainagestate",
    ),
    "treestand": (
        "standid",
        "treestandid",
        "treestandclass",
        "treestanddatasource",
        "treedatadate",
        "developmentclass",
    ),
    "treestratum": (
        "treestandid",
        "treespecies",
        "age",
        "basalarea",
        "stemcount",
        "meandiameter",
        "meanheight",
    ),
}

# What the CLI stops with on an old-format file: one with the site columns on
# the stand layer and no growthplacedata layer.
OLD_FORMAT_MESSAGE = (
    "This .gpkg is in Metsäkeskus's old (pre-2026) format, which this tool "
    "does not read. Use susi-metsakeskus-to-allometry-old-format for it, or "
    "download the data again in the current format."
)


# %% dataclasses

# PerSpecies[T] now lives in tools.shared_allometry_tool_utils.tree_stratum
# (imported above), shared with xml_to_allometry.py and
# new_growth_allometry.py -- it used to be defined here only.


@dataclass(frozen=True)
class StandCandidate:
    """
    Represents one stand after database merge + species aggregation, before the viability
    check (partition_viable_candidates) that decides whether it's actually
    processed.
    """

    id: StandID
    subgroup: int
    fertilityclass: int
    # The inventory snapshot's own development-class code ("02", "T1", ...),
    # None when the snapshot records none (remote-sensed ones never do).
    developmentclass: str | None
    drainagestate: int
    soiltype: int | None
    strata: PerSpecies[TreeStratum]
    geometry: Polygon
    area: float | None  # ha, the stand layer's own area column; None if not recorded


@dataclass(frozen=True)
class ParsedStand:
    """
    One stand ready for Growth_and_Yield_Table.
    Difference with StandData: StandData is what finally gets written to the
    JSON file. ParsedStand is temporary, and never exists outside this file.
    """

    id: StandID
    subgroup: int
    fertilityclass: int
    # The inventory snapshot's own development-class code ("02", "T1", ...),
    # None when the snapshot records none (remote-sensed ones never do).
    developmentclass: str | None
    drainagestate: int
    soiltype: int | None
    strata: PerSpecies[TreeStratum]
    stand_basalarea: float
    stand_meanheight: float
    stand_meandiameter: float
    x_ykj: int
    y_ykj: int
    area: float | None  # ha -- the stand layer's own area column, see build_stand
    geometry: Polygon
    dominant_species: int
    subdominant_species: int


@dataclass(frozen=True)
class GpkgLayers:
    """
    The four gpkg layers this tool reads, each with only its LAYER_COLUMNS.
    stand: one row per stand, its geometry and area.
    growthplacedata: one row per stand, its site (main group, subgroup,
    fertility class, soil type, drainage state).
    treestand: one row per tree-stand record (inventory snapshots and
    Metsäkeskus's projections) of a stand.
    treestratum: one row per stratum of a tree-stand record.
    treestandsummary is deliberately absent: the tool aggregates the strata
    itself, the same way for every stand.
    """

    stand: gpd.GeoDataFrame
    growthplacedata: pd.DataFrame
    treestand: pd.DataFrame
    treestratum: pd.DataFrame


class ExtractionConfig(AllometryGenerationDefaults):
    """
    Defaulted/required parameters, loaded from a TOML file. Hard-coded,
    non-negotiable parameters live as module constants above instead.

    Subclasses AllometryGenerationDefaults (not StrictFrozenModel directly)
    for the shared n_trees/start_year/end_year/step_years defaults --
    StrictFrozenModel (susi.io.extra_pydantic_types) still gives us presence
    checking for the required fields below (no default -> required),
    rejection of unknown fields (extra="forbid"), and immutability
    (frozen=True) for free -- replacing check_config_fields,
    REQUIRED_CONFIG_FIELDS, and the dataclass's own frozen=True.
    """

    # Required, no defaults
    target_year: int
    altitude: float
    ddy: float

    # (minimum, maximum) stand mean diameter, cm, both ends included -- see
    # filter_by_mean_diameter_range. No default: what counts as too young or
    # too old a stand depends on the region
    mean_diameter_range_filter: tuple[NonNegativeFloat, NonNegativeFloat]

    # Defaulted

    # peatland fertility classes of interest
    # restricting to 2-5 excludes the very richest and very poorest extremes.
    fertilityclass_filter: tuple[int, ...] = (2, 3, 4, 5)

    # The optional [dense_young_stand_scaling] table: off, with the default
    # numbers, when the config file leaves it out. The same table, defaults
    # and rule as xml_to_allometry.py's XmlConfig. It lives here rather than
    # on AllometryGenerationDefaults because new_growth_allometry.py's config
    # subclasses that too, and the option means nothing there.
    dense_young_stand_scaling: DenseYoungStandScalingConfig = (
        DenseYoungStandScalingConfig()
    )

    @model_validator(mode="after")
    def mean_diameter_range_is_a_range(self) -> Self:
        minimum, maximum = self.mean_diameter_range_filter
        if minimum >= maximum:
            raise ValueError(
                f"mean_diameter_range_filter's minimum ({minimum:g}) is not "
                f"below its maximum ({maximum:g}). Give it as [minimum, maximum]."
            )
        return self

    @model_validator(mode="after")
    def dense_young_stand_scaling_can_apply(self) -> Self:
        # Scaling only acts on stands thinner than max_mean_diameter, and the
        # range filter runs first: with its minimum at or above that, no stand
        # is left to scale, and turning the option on would silently do
        # nothing.
        scaling = self.dense_young_stand_scaling
        minimum = self.mean_diameter_range_filter[0]
        if scaling.enabled and minimum >= scaling.max_mean_diameter:
            raise ValueError(
                f"[dense_young_stand_scaling] is enabled, but it only scales "
                f"stands with a mean diameter below max_mean_diameter "
                f"({scaling.max_mean_diameter:g} cm), and "
                f"mean_diameter_range_filter's minimum ({minimum:g} cm) already "
                "leaves all of those out. Lower the minimum, or turn the "
                "scaling off."
            )
        return self


@dataclass(frozen=True)
class CLIArguments:
    input_gpkg: Path
    config: ExtractionConfig
    config_path: Path
    project_dir: Path
    allow_out_of_range_values: bool
    dry_run: bool


@dataclass(frozen=True)
class StandPlanned:
    """
    The files one stand would produce.
    This is a fact about the stand and the output folder,
    knowable before any growth table is computed.
    """

    stand_id: StandID
    dominant_csv: Path
    # None exactly when the second species carries no basal area of its own.
    subdominant_csv: Path | None


@dataclass(frozen=True)
class StandWritten:
    """
    A stand that survives the filters and the checks,
    and whose allometry file gets written.
    """

    stand_id: StandID
    stand_data: StandData
    dominant_csv: Path
    # When the second species carries zero basal area, no subdominant allometry file is written.
    subdominant_csv: Path | None


StandOutcome = StandWritten | StandSkipped


# %% Config parsing


def load_extraction_config(config_path: Path) -> ExtractionConfig:
    return load_toml_config(config_path, ExtractionConfig.model_validate)


# %% gpkg loading (I/O)


# The warning pyogrio gives every time it reads the stand layer's "Measured 3D
# Polygon" geometry (or lists the layers, or reads a layer's metadata): it
# drops the measure (M) value. This tool only uses x/y, so main() silences the
# warning once, for the whole run.
MEASURED_GEOMETRY_WARNING = r"Measured \(M\) geometry types are not supported"


def _read_layer(gpkg_path: Path, layer: str, columns: list[str]) -> pd.DataFrame:
    """
    One gpkg layer's given columns, every row, as gpd.read_file reads it: a
    GeoDataFrame for the stand layer, a plain DataFrame for the
    attribute-only ones.
    """
    return gpd.read_file(gpkg_path, layer=layer, columns=columns)


def gpkg_format_problem(gpkg_path: Path) -> str | None:
    """
    None when the file has every layer and column in LAYER_COLUMNS: the
    2026 data model this tool reads. Otherwise, what is wrong with it, as the
    message the CLI stops with.

    An old-format file (no growthplacedata layer, and the site columns still
    on the stand layer) gets OLD_FORMAT_MESSAGE, which points to the tool that
    reads it. Only the layer list and each layer's column names are read
    here, not the data.
    """
    layers = set(gpd.list_layers(gpkg_path)["name"])

    def layer_columns(layer: str) -> set[str]:
        # The layer's column names, from its metadata: no row is read.
        return set(pyogrio.read_info(gpkg_path, layer=layer)["fields"])

    if (
        "growthplacedata" not in layers
        and "stand" in layers
        and "maingroup" in layer_columns("stand")
    ):
        return OLD_FORMAT_MESSAGE

    missing: list[str] = []
    for layer, columns in LAYER_COLUMNS.items():
        if layer not in layers:
            missing.append(f"layer '{layer}'")
            continue
        present = layer_columns(layer)
        missing.extend(
            f"{layer}.{column}" for column in columns if column not in present
        )
    if missing:
        return (
            "This .gpkg is not in the Metsäkeskus data model this tool reads "
            f"(2026, v2.1.x). Missing: {', '.join(missing)}."
        )
    return None


def load_gpkg_layers(gpkg_path: Path) -> GpkgLayers:
    """Loads the four gpkg layers this tool needs, each with only its
    LAYER_COLUMNS. See GpkgLayers for what each one holds."""
    layers = {
        layer: _read_layer(gpkg_path, layer, columns=list(columns))
        for layer, columns in LAYER_COLUMNS.items()
    }
    stand = layers["stand"]
    if not isinstance(stand, gpd.GeoDataFrame):
        raise TypeError("The gpkg stand layer has no geometry")
    return GpkgLayers(
        stand=stand,
        growthplacedata=layers["growthplacedata"],
        treestand=layers["treestand"],
        treestratum=layers["treestratum"],
    )


def stand_layer_in_source_crs(stand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    The stand layer in SOURCE_CRS (EPSG:3067), where centroid_to_ykj and
    StandDataDocument.crs need it: returned as-is when it's already there,
    reprojected as a whole otherwise -- the layer-at-once counterpart of
    shared_utils.to_source_crs, which does one polygon at a time.

    Raises ValueError when the layer declares no CRS: there is nothing to
    reproject from, and that's a whole-file problem, so it aborts the run
    rather than skipping stands.
    """
    if stand.crs is None:
        raise ValueError(
            "The gpkg stand layer declares no CRS, so it can't be reprojected"
        )
    if stand.crs == SOURCE_CRS:
        return stand
    return stand.to_crs(SOURCE_CRS)


# %% Filtering


def filter_stands_by_site_attributes(
    growthplacedata: pd.DataFrame,
    fertilityclass_filter: tuple[int, ...],
) -> pd.DataFrame:
    """
    Restrict to forest land, on peatland, already drained.
    This is hard-coded per the global variables above.
    Also filters the configured fertility-class range.

    growthplacedata holds one row per stand; the codes are text in the gpkg,
    so they are made numbers here, and stay numbers in the returned rows.
    """
    numeric = growthplacedata.copy()
    for column in ("maingroup", "subgroup", "drainagestate", "fertilityclass"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    mask = (
        (numeric["maingroup"] == MAINGROUP_FOREST_LAND)
        & numeric["subgroup"].isin(SUBGROUP_PEATLAND)
        & numeric["drainagestate"].isin(DRAINAGESTATE_DRAINED)
        & numeric["fertilityclass"].isin(fertilityclass_filter)
    )
    return numeric[mask].copy()


def _inventory_snapshots(treestand: pd.DataFrame, stand_ids: set[int]) -> pd.DataFrame:
    """
    Shared for select_target_year_snapshot() and compute_year_distribution().
    The given stands' inventory snapshots (TREESTANDCLASS_INVENTORY), with
    their date and year parsed.
    """
    candidates = treestand[treestand["standid"].isin(stand_ids)].copy()
    candidates["date_dt"] = pd.to_datetime(candidates["treedatadate"], errors="coerce")
    candidates["class_num"] = pd.to_numeric(
        candidates["treestandclass"], errors="coerce"
    )
    candidates["year"] = candidates["date_dt"].dt.year
    return candidates[candidates["class_num"] == TREESTANDCLASS_INVENTORY]


def _source_code_order(source: str) -> tuple[bool, int, str]:
    """Sort key for treestanddatasource codes: numeric codes in numeric order
    (2 before 21), anything else after them."""
    return (not source.isdigit(), int(source) if source.isdigit() else 0, source)


def compute_year_distribution(
    treestand: pd.DataFrame, stand_ids: set[int]
) -> pd.DataFrame:
    """
    Per year: how many distinct stands have an inventory snapshot dated that
    year (n_stands), and how many of them from each inventory source (one
    "source <code>" column per treestanddatasource code; "source none" when
    the snapshot records no source). A stand with two snapshots from
    different sources in one year counts once in n_stands.
    """
    snapshots = _inventory_snapshots(treestand, stand_ids).dropna(subset=["year"])
    snapshots = snapshots.astype({"year": int})
    snapshots["source"] = snapshots["treestanddatasource"].fillna("none").astype(str)

    n_stands = snapshots.groupby("year").agg(n_stands=("standid", "nunique"))
    per_source = (
        snapshots.groupby(["year", "source"])["standid"].nunique().unstack(fill_value=0)
    )
    per_source = per_source[sorted(per_source.columns, key=_source_code_order)]
    per_source.columns = [f"source {source}" for source in per_source.columns]
    return (
        pd.concat([n_stands, per_source], axis=1)
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )


def select_target_year_snapshot(
    treestand: pd.DataFrame,
    stand_ids: set[int],
    target_year: int,
) -> pd.DataFrame:
    """
    One treestand row per stand: the inventory snapshot dated in
    target_year, the earliest one if the stand has several that year.
    """
    snapshots = _inventory_snapshots(treestand, stand_ids)
    exact_year = snapshots[snapshots["year"] == target_year]

    return (
        exact_year.sort_values(["standid", "date_dt"])
        .drop_duplicates(subset="standid", keep="first")
        .reset_index(drop=True)
    )


def drop_duplicated_stands(
    treestand_snapshot: pd.DataFrame,
    growthplacedata: pd.DataFrame,
    stand: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, list[StandSkipped]]:
    """
    attach_stand_attributes joins each snapshot to growthplacedata and to the
    stand layer by standid, and expects one row per stand in each. A stand
    with more than one would come out of that join twice: built twice, its
    files written twice. Such a stand is dropped from treestand_snapshot here
    instead, as a StandSkipped naming the layer and the count.

    Only the snapshot's stands are checked: a duplicate elsewhere in the
    region doesn't stop the rest. growthplacedata is the whole layer, not the
    site-filtered one, so a duplicate the site filter happened to halve is
    still caught.
    """
    snapshot_ids = set(pd.to_numeric(treestand_snapshot["standid"], errors="coerce"))
    reasons_per_stand: dict[int, list[str]] = {}
    for layer_name, layer in (("growthplacedata", growthplacedata), ("stand", stand)):
        layer_ids = pd.to_numeric(layer["standid"], errors="coerce")
        counts = layer_ids[layer_ids.isin(snapshot_ids)].value_counts()
        for stand_id, count in counts[counts > 1].items():
            reasons_per_stand.setdefault(int(stand_id), []).append(
                f"standid appears {count} times in the {layer_name} layer"
            )

    skipped = [
        StandSkipped(stand_id=StandID(str(stand_id)), reason="; ".join(reasons))
        for stand_id, reasons in sorted(reasons_per_stand.items())
    ]
    is_duplicated = pd.to_numeric(treestand_snapshot["standid"], errors="coerce").isin(
        set(reasons_per_stand)
    )
    return treestand_snapshot[~is_duplicated].reset_index(drop=True), skipped


def attach_stand_attributes(
    treestand_snapshot: pd.DataFrame,
    filtered_growthplacedata: pd.DataFrame,
    stand: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Attaches each selected treestand snapshot's site columns (from
    growthplacedata) and its geometry and area (from the stand layer), since
    treestand itself carries neither. The snapshot keeps its own
    developmentclass column.

    A stand missing from the stand layer gets a NaN geometry here, which
    build_stand_candidates reports as a clean skip.
    """
    site_columns = [
        "standid",
        "subgroup",
        "fertilityclass",
        "drainagestate",
        "soiltype",
    ]
    with_site = treestand_snapshot.merge(
        filtered_growthplacedata[site_columns], on="standid", how="left"
    )
    return with_site.merge(
        pd.DataFrame(stand[["standid", "area", "geometry"]]), on="standid", how="left"
    )


# %% Species aggregation


def basal_area_per_tree_m2(diameter_cm: float) -> float:
    radius_m = diameter_cm / 2 / 100
    return math.pi * radius_m**2


def estimate_stemcount(total_basal_area: float, mean_diameter_cm: float) -> int:
    """Backs out stem count from basal area and mean diameter when the
    inventory didn't record stemcount directly: N = BA / (per-tree basal
    area of a stem with diameter mean_diameter_cm)."""
    if total_basal_area <= 0 or mean_diameter_cm <= 0:
        return 0
    return max(1, round(total_basal_area / basal_area_per_tree_m2(mean_diameter_cm)))


class DegenerateSpeciesDataError(ValueError):
    """Raised by aggregate_species_group when a species group carries real,
    positive recorded basal area but no usable mean diameter and/or mean
    height to go with it. There is no safe placeholder to substitute here
    (unlike a species with zero basal area, which is provably inert
    downstream regardless of what its age/diameter/height say -- see
    aggregate_species_group's docstring): a fabricated diameter would also
    get compounded by estimate_stemcount into a fabricated stem count
    derived from a real basal area and a fake diameter. Caught by
    build_stand_candidates and turned into a StandSkipped for the whole
    stand."""


def aggregate_species_group(rows: pd.DataFrame, species_name: str) -> TreeStratum:
    """Collapses every treestratum row belonging to one species group (there
    can be several, e.g. different diameter cohorts) into one TreeStratum.
    Basal area being zero is never a reason to discard or rewrite what was
    actually recorded for a species -- e.g. a species can be recorded with a
    real, positive stem count and no basal-area figure, and that stem count
    is kept. Basal area only controls whether this tool ever *uses* the
    species for allometry: a zero-basal-area species can never become
    dominant (ranked by basal area, and partition_viable_candidates already
    guarantees the stand's total is positive) and is gated out even as
    subdominant (process_stand only builds a subdominant table when
    basal_area > 0) -- so whatever this function returns for it is
    informational only, never fed into Growth_and_Yield_Table (also see
    ZERO_STRATUM and isolate_species_layer, which enforces the same "not
    passed forward" rule one level down, per canopy layer).

    A species entirely absent from this stand (rows empty) gets the flat
    ZERO_STRATUM -- there is nothing recorded to preserve.

    A species present with rows summing to zero basal area keeps whatever
    was actually recorded: the real summed stem count, and age/diameter/
    height from a plain average of the recorded rows. Only a field with
    truly nothing to average (an all-NaN column) falls back to 0 -- not a
    fabricated placeholder, and deliberately not NaN either, since NaN would
    silently poison build_stand's basal-area-weighted stand-level
    age/height/diameter (NaN * 0 is NaN, not 0).

    A species present with real, POSITIVE basal area but degenerate
    diameter/height data raises DegenerateSpeciesDataError instead (see that
    class's docstring): unlike the zero-basal-area case, this species can
    actually reach Growth_and_Yield_Table, so there is no safe value to
    invent. species_name is only used to name it in that error message."""
    if rows.empty:
        return ZERO_STRATUM

    total_basal_area = float(rows["basalarea"].sum(skipna=True))
    total_stem_count = float(rows["stemcount"].sum(skipna=True))

    if total_basal_area > 0:
        # Basal-area-weighted means: strata with more BA (bigger/more trees)
        # dominate the aggregate age/diameter/height for the species group.
        weights = rows["basalarea"].fillna(0.0)
        weighted_age = (rows["age"].fillna(0.0) * weights).sum() / total_basal_area
        weighted_diameter = (
            rows["meandiameter"].fillna(0.0) * weights
        ).sum() / total_basal_area
        weighted_height = (
            rows["meanheight"].fillna(0.0) * weights
        ).sum() / total_basal_area

        # This species has real, recorded basal area -- it can end up as the
        # dominant or subdominant canopy layer and actually be fed into
        # Growth_and_Yield_Table, so a missing diameter/height here is real,
        # artificial data that must not flow forward -- reject the stand
        # rather than paper over it.
        if not (pd.notna(weighted_diameter) and weighted_diameter > 0):
            raise DegenerateSpeciesDataError(
                f"{species_name}: basal area {total_basal_area:.2f} m2/ha "
                "but no usable mean diameter"
            )
        if not (pd.notna(weighted_height) and weighted_height > 0):
            raise DegenerateSpeciesDataError(
                f"{species_name}: basal area {total_basal_area:.2f} m2/ha "
                "but no usable mean height"
            )
        age = max(1, round(weighted_age))
        diameter = float(weighted_diameter)
        height = float(weighted_height)

        if total_stem_count <= 0:
            total_stem_count = estimate_stemcount(total_basal_area, diameter)
    else:
        # Rows ARE present -- this species IS recorded in the stand -- but
        # they sum to zero basal area. Keep the real stem count (do not zero
        # it just because basal area is zero) and a plain average of
        # whatever age/diameter/height was recorded; only fall back to 0
        # where a column is entirely missing/NaN across these rows.
        weighted_age = rows["age"].mean()
        weighted_diameter = rows["meandiameter"].mean()
        weighted_height = rows["meanheight"].mean()
        age = round(weighted_age) if pd.notna(weighted_age) else 0
        diameter = float(weighted_diameter) if pd.notna(weighted_diameter) else 0.0
        height = float(weighted_height) if pd.notna(weighted_height) else 0.0

    return TreeStratum(
        age=age,
        basal_area=total_basal_area,
        stem_count=round(total_stem_count) if total_stem_count > 0 else 0,
        mean_diameter=diameter,
        mean_height=height,
    )


def build_species_strata(stand_strata: pd.DataFrame) -> PerSpecies[TreeStratum]:
    """Splits one stand's treestratum rows into the three SUSI species slots.
    May raise DegenerateSpeciesDataError (see aggregate_species_group) --
    callers should expect and handle that."""
    return PerSpecies(
        pine=aggregate_species_group(
            stand_strata[stand_strata["treespecies"].isin(PINE_SPECIES_CODES)],
            species_name="pine",
        ),
        spruce=aggregate_species_group(
            stand_strata[stand_strata["treespecies"].isin(SPRUCE_SPECIES_CODES)],
            species_name="spruce",
        ),
        deciduous=aggregate_species_group(
            stand_strata[stand_strata["treespecies"].isin(DECIDUOUS_SPECIES_CODES)],
            species_name="deciduous",
        ),
    )


# total_basal_area is now the shared
# tools.shared_allometry_tool_utils.dense_young_stand_scaling function,
# imported above -- it used to be a local copy.


# %% Candidate assembly and viability


def build_stand_candidates(
    merged_filtered: pd.DataFrame,
    treestratum: pd.DataFrame,
) -> tuple[list[StandCandidate], list[StandSkipped]]:
    """One StandCandidate per row of merged_filtered. A row with no usable
    geometry is excluded here, as a StandSkipped -- not carried downstream as
    a None geometry -- and so is one whose geometry isn't a single Polygon
    (StandData.polygon accepts nothing else). Same treatment for a species
    with real basal area but degenerate diameter/height data (build_species_strata's
    DegenerateSpeciesDataError, see aggregate_species_group) -- caught here
    and turned into a StandSkipped for the whole stand, rather than letting
    artificial data flow forward into a ParsedStand."""
    candidates: list[StandCandidate] = []
    skipped: list[StandSkipped] = []
    treestratum_treestandid = pd.to_numeric(treestratum["treestandid"], errors="coerce")
    # The species codes are text in the gpkg ("1", "2", ...), and the
    # *_SPECIES_CODES sets are numbers: converted once here, for every stand.
    treestratum = treestratum.assign(
        treespecies=pd.to_numeric(treestratum["treespecies"], errors="coerce")
    )

    for _, row in merged_filtered.iterrows():
        stand_id = StandID(str(int(row["standid"])))
        geometry = row.get("geometry")
        # `geometry is None` alone misses a NaN geometry cell (e.g. from an
        # unmatched merge key -- pandas fills those with float NaN, not
        # None), and a non-null but empty geometry, both of which would
        # otherwise reach centroid_to_ykj and raise there instead of being
        # reported as a clean skip.
        if geometry is None or pd.isna(geometry) or geometry.is_empty:
            skipped.append(StandSkipped(stand_id=stand_id, reason="no usable geometry"))
            continue
        if not isinstance(geometry, Polygon):
            skipped.append(
                StandSkipped(
                    stand_id=stand_id,
                    reason=f"geometry is a {geometry.geom_type}, not a Polygon",
                )
            )
            continue

        treestandid_value = pd.to_numeric(
            pd.Series([row.get("treestandid")]), errors="coerce"
        ).iloc[0]
        if pd.isna(treestandid_value):
            skipped.append(StandSkipped(stand_id=stand_id, reason="no treestandid"))
            continue

        stand_strata = treestratum[treestratum_treestandid == int(treestandid_value)]
        try:
            strata = build_species_strata(stand_strata)
        except DegenerateSpeciesDataError as error:
            skipped.append(StandSkipped(stand_id=stand_id, reason=str(error)))
            continue

        candidates.append(
            StandCandidate(
                id=stand_id,
                subgroup=int(row["subgroup"]),
                fertilityclass=int(row["fertilityclass"]),
                developmentclass=(
                    str(row["developmentclass"])
                    if pd.notna(row.get("developmentclass"))
                    else None
                ),
                drainagestate=(
                    int(row["drainagestate"])
                    if pd.notna(row.get("drainagestate"))
                    else DRAINAGESTATE_DRAINED[0]
                ),
                soiltype=(
                    int(row["soiltype"]) if pd.notna(row.get("soiltype")) else None
                ),
                strata=strata,
                geometry=geometry,
                # None when the cell is empty: "not recorded", like soiltype.
                area=(float(row["area"]) if pd.notna(row.get("area")) else None),
            )
        )

    return candidates, skipped


def partition_viable_candidates(
    candidates: list[StandCandidate],
) -> tuple[list[StandCandidate], list[StandSkipped]]:
    """Splits out stands with zero total basal area across all three species
    -- no usable growth data -- as StandSkipped, instead of a None sentinel
    threaded through the builder below."""
    viable: list[StandCandidate] = []
    skipped: list[StandSkipped] = []
    for candidate in candidates:
        if total_basal_area(candidate.strata) <= 0:
            skipped.append(
                StandSkipped(
                    stand_id=candidate.id, reason="zero basal area across all species"
                )
            )
        else:
            viable.append(candidate)
    return viable, skipped


def filter_by_mean_diameter_range(
    candidates: list[StandCandidate],
    mean_diameter_range: tuple[float, float],
) -> tuple[list[StandCandidate], list[StandSkipped]]:
    """
    Keeps a stand when its basal-area-weighted mean diameter lies in
    mean_diameter_range (cm, both ends included): this is how the tool leaves
    out stands too young or too old to grow into a useful allometry curve.
    It judges the very trees the growth model is given, on the inventory
    snapshot's own date, whatever the inventory's source.

    The diameter is the one dense young stand scaling judges a stand by and
    StandData.mean_diameter records. It is undefined for a stand with no
    basal area, so this takes viable candidates (partition_viable_candidates).
    """
    minimum, maximum = mean_diameter_range
    kept: list[StandCandidate] = []
    skipped: list[StandSkipped] = []
    for candidate in candidates:
        mean_diameter = basal_area_weighted_mean_diameter(candidate.strata)
        if minimum <= mean_diameter <= maximum:
            kept.append(candidate)
        else:
            skipped.append(
                StandSkipped(
                    stand_id=candidate.id,
                    reason=(
                        f"mean diameter {mean_diameter:.1f} cm, outside "
                        f"{minimum:g}-{maximum:g} cm"
                    ),
                )
            )
    return kept, skipped


# The row for stands whose inventory snapshot records no development class.
NO_DEVELOPMENTCLASS_LABEL = "(none)"


def summarize_mean_diameter_by_developmentclass(
    candidates: list[StandCandidate],
) -> pd.DataFrame:
    """
    Per development class: how many stands, and the 10th percentile, median
    and 90th percentile of their basal-area-weighted mean diameter (cm) --
    the figure filter_by_mean_diameter_range judges. Report only: it is what
    a user reads to choose mean_diameter_range_filter for their own region.
    Stands whose snapshot records no class share one NO_DEVELOPMENTCLASS_LABEL
    row. Takes viable candidates, like filter_by_mean_diameter_range.
    """
    if not candidates:
        return pd.DataFrame(
            {
                column: []
                for column in ("developmentclass", "n_stands", "p10", "median", "p90")
            }
        )
    diameters = pd.DataFrame(
        {
            "developmentclass": [
                candidate.developmentclass or NO_DEVELOPMENTCLASS_LABEL
                for candidate in candidates
            ],
            "mean_diameter": [
                basal_area_weighted_mean_diameter(candidate.strata)
                for candidate in candidates
            ],
        }
    )
    per_class = diameters.groupby("developmentclass")["mean_diameter"]
    return (
        pd.DataFrame(
            {
                "n_stands": per_class.size(),
                "p10": per_class.quantile(0.1),
                "median": per_class.quantile(0.5),
                "p90": per_class.quantile(0.9),
            }
        )
        .round(1)
        .reset_index()
    )


# %% Building the final stand record


def determine_dominant_and_subdominant_species(
    strata: PerSpecies[TreeStratum],
) -> tuple[int, int]:
    """Species codes (1=pine, 2=spruce, 3=deciduous) ranked by basal area.
    The subdominant is always the second-ranked species' own stratum, even
    when its basal area is zero: never a copy of the dominant species, whose
    basal area would then count twice in a monoculture stand.
    """
    basal_areas = {
        species_code: species_stratum(strata, species_code).basal_area
        for species_code in _SPECIES_SLOT
    }
    ranked = sorted(
        basal_areas, key=lambda species_code: basal_areas[species_code], reverse=True
    )
    return ranked[0], ranked[1]


def build_stand(candidate: StandCandidate) -> ParsedStand:
    """
    Builds the final stand record from an already-viable candidate
    Not final, though: centroid_to_ykj can still raise for a geometry that's
    present and non-empty but otherwise degenerate (e.g. all-coincident points)

    `area` is the stand layer's own area column, carried over from the
    candidate -- not recomputed from the geometry (CONTEXT.md, "Stand area").
    """
    strata = candidate.strata
    stand_total_ba = total_basal_area(strata)

    # Stand-level height/diameter: basal-area-weighted across the three
    # species slots (mirrors aggregate_species_group's weighting, one level up).
    stand_height = (
        strata.pine.mean_height * strata.pine.basal_area
        + strata.spruce.mean_height * strata.spruce.basal_area
        + strata.deciduous.mean_height * strata.deciduous.basal_area
    ) / stand_total_ba
    # The diameter comes from the function dense young stand scaling judges a
    # stand with, so the mean diameter recorded in StandData and the one the
    # rule compares with its limit are the same number by construction. It
    # raises for a stand with no basal area, which a viable candidate
    # (partition_viable_candidates) never is.
    stand_diameter = basal_area_weighted_mean_diameter(strata)

    dominant_species, subdominant_species = determine_dominant_and_subdominant_species(
        strata
    )
    x_ykj, y_ykj = centroid_to_ykj(candidate.geometry)

    return ParsedStand(
        id=candidate.id,
        subgroup=candidate.subgroup,
        fertilityclass=candidate.fertilityclass,
        developmentclass=candidate.developmentclass,
        drainagestate=candidate.drainagestate,
        soiltype=candidate.soiltype,
        strata=strata,
        stand_basalarea=float(stand_total_ba),
        stand_meanheight=float(stand_height),
        stand_meandiameter=float(stand_diameter),
        x_ykj=x_ykj,
        y_ykj=y_ykj,
        area=candidate.area,
        geometry=candidate.geometry,
        dominant_species=dominant_species,
        subdominant_species=subdominant_species,
    )


def build_stands(
    candidates: list[StandCandidate],
) -> tuple[list[ParsedStand], list[StandSkipped]]:
    """Batch build_stand, isolating one candidate's failure (see
    build_stand's docstring) as a StandSkipped instead of letting it
    abort every other stand in the run -- the same per-stand isolation
    principle process_stand already applies to the growth-table stage."""
    built: list[ParsedStand] = []
    skipped: list[StandSkipped] = []
    for candidate in candidates:
        try:
            built.append(build_stand(candidate))
        except Exception as error:  # noqa: BLE001 -- one bad stand's geometry math must not abort the batch
            skipped.append(StandSkipped(stand_id=candidate.id, reason=str(error)))
    return built, skipped


# %% Growth-and-yield table construction

# ZERO_STRATUM (used below by isolate_species_layer) is defined once, next
# to TreeStratum, and shared with aggregate_species_group -- see its
# docstring there.


# The PerSpecies slot each species code names. The one species-code -> slot
# mapping in this tool: species_stratum and isolate_species_layer both read it.
# Moving it into shared_allometry_tool_utils/ is create-input-structure
# ticket 12's job.
_SPECIES_SLOT = {1: "pine", 2: "spruce", 3: "deciduous"}


def species_stratum(strata: PerSpecies[TreeStratum], species_code: int) -> TreeStratum:
    """The one TreeStratum a species code (1=pine, 2=spruce, 3=deciduous) refers to."""
    return getattr(strata, _SPECIES_SLOT[species_code])


def isolate_species_layer(
    strata: PerSpecies[TreeStratum], active_species: int
) -> PerSpecies[TreeStratum]:
    """Zeroes every species slot except active_species -- this is how a
    single canopy layer (dominant or subdominant) is modeled as that one
    species growing alone"""
    all_zero = PerSpecies(
        pine=ZERO_STRATUM, spruce=ZERO_STRATUM, deciduous=ZERO_STRATUM
    )
    return dataclasses.replace(
        all_zero,
        **{_SPECIES_SLOT[active_species]: species_stratum(strata, active_species)},
    )


def build_growth_and_yield_table(
    strata: PerSpecies[TreeStratum],
    active_species: int,
    fertility_class: int,
    x_ykj: int,
    y_ykj: int,
    altitude: float,
    ddy: float,
    n_trees: int,
    start_year: int,
    end_year: int,
    step_years: int,
) -> pd.DataFrame:
    """One canopy layer's allometric growth trajectory (get_table's age-indexed
    rows from start_year to end_year), modeled as active_species growing alone.

    Isolates the layer locally (the only one of the three tools that needs
    per-layer isolation -- xml_to_allometry.py and new_growth_allometry.py
    pass their full, never-isolated PerSpecies straight through instead),
    then delegates the actual Growth_and_Yield_Table construction to the
    shared helper."""
    layer = isolate_species_layer(strata, active_species)
    return build_isolated_growth_and_yield_table(
        strata=layer,
        fertility_class=fertility_class,
        x_ykj=x_ykj,
        y_ykj=y_ykj,
        altitude=altitude,
        ddy=ddy,
        n_trees=n_trees,
        start_year=start_year,
        end_year=end_year,
        step_years=step_years,
        peat=PEAT,
    )


# %% Writing output (I/O)


def write_allometry_csv(table: pd.DataFrame, output_path: Path) -> None:
    """Writes one CanopyLayerAllometry-contract CSV -- readable directly by
    susi.io.susi_parameter_model.read_allometry_info_from_csv."""
    table_with_species = table.copy()
    table_with_species.to_csv(output_path, index=False)


# dump_stand_data_document is now the shared
# susi.io.stand_data function, imported above --
# it used to be a local copy.


# %% Per-stand orchestration


def plan_stand_outputs(stand: ParsedStand, output_dir: Path) -> StandPlanned:
    """Which files this stand produces, and where. Pure, total, and cheap: it
    reads only what partition_viable_candidates and build_stand have
    already settled, so it can run long before the growth model does.

    Both paths go through here -- process_stand for a real run, main() for a
    dry run -- so the dry run reports the real run's own decision rather than
    a parallel reimplementation of the same naming rules that could drift
    from it.

    The subdominant file exists exactly when the second-ranked species carries
    basal area of its own: a monoculture stand gets no subdominant file at
    all, rather than one duplicating the dominant."""
    subdominant_stratum = species_stratum(stand.strata, stand.subdominant_species)
    return StandPlanned(
        stand_id=stand.id,
        dominant_csv=output_dir / f"{stand.id}_dominant.csv",
        subdominant_csv=(
            output_dir / f"{stand.id}_subdominant.csv"
            if subdominant_stratum.basal_area > 0
            else None
        ),
    )


def build_stand_data(
    stand: ParsedStand,
    plan: StandPlanned,
    initial_age_per_layer: dict[CanopyLayerName, float],
    scaling_factor: float | None,
) -> StandData:
    """The stand's StandData record. Pure: it runs no growth model and writes
    nothing.
    """

    allometry_file_per_layer: dict[CanopyLayerName, AllometryFileAndSpecies] = {
        CanopyLayerName.dominant: AllometryFileAndSpecies(
            file_path=plan.dominant_csv, species_id=stand.dominant_species
        ),
    }
    if plan.subdominant_csv is not None:
        allometry_file_per_layer[CanopyLayerName.subdominant] = AllometryFileAndSpecies(
            file_path=plan.subdominant_csv, species_id=stand.subdominant_species
        )

    return StandData(
        site_fertility_class=stand.fertilityclass,
        allometry_file_per_layer=allometry_file_per_layer,
        initial_age_per_layer=initial_age_per_layer,
        x_ykj=stand.x_ykj,
        y_ykj=stand.y_ykj,
        polygon=stand.geometry,
        stand_area=stand.area,
        main_group=MAINGROUP_FOREST_LAND,
        sub_group=stand.subgroup,
        # The stand-level figures are raw, exactly as the inventory
        # recorded them, for a scaled stand too: the record keeps what
        # the inventory reported, and notes dense young stand scaling
        # separately, in the two fields at the end.
        basal_area=stand.stand_basalarea,
        mean_height=stand.stand_meanheight,
        mean_diameter=stand.stand_meandiameter,
        # The sum of the three species' stem counts, from the very
        # function the rule judges a stand's density with. Without it,
        # the scaling factor (target stem count over THIS number) would
        # have nothing in the record to refer to. A species' stem count
        # is an estimate when the inventory recorded none
        # (aggregate_species_group, estimate_stemcount); the sum takes
        # whatever the strata carry, estimated or not, as the rule does.
        stem_count=total_stem_count(stand.strata),
        developmentclass=stand.developmentclass,
        drainagestate=stand.drainagestate,
        soil_type=stand.soiltype,
        dense_young_stand_scaling_applied=scaling_factor is not None,
        dense_young_stand_scaling_factor=scaling_factor,
    )


def process_stand(
    stand: ParsedStand,
    config: ExtractionConfig,
    output_dir: Path,
    scaling_factor: float | None,
) -> StandOutcome:
    """Builds the dominant canopy layer's allometry CSV (always) and the
    subdominant's (only when a genuine second species is present -- see
    StandWritten.subdominant_csv), then the shared StandData record those
    files (plus this stand's other metadata) are wrapped into. Any failure
    here is this one stand's problem, not the run's: it's caught and turned
    into a StandSkipped rather than aborting the batch.

    scaling_factor is the stand's dense young stand scaling factor, decided
    once in main(): None for a stand that isn't scaled. With a factor, the
    whole stand is scaled first -- all three species' strata, by the one
    factor -- and only then are the two layers split off it, so both growth
    tables are built from the scaled-down strata. The StandData record is
    filled from the unscaled stand either way, and says whether the stand
    was scaled and by what factor.

    Both growth tables are computed in full, and the StandData record below
    is built (and pydantic-validated -- e.g. against a degenerate
    stand.area) BEFORE anything is written to disk: if any of that fails,
    we must not have already written the dominant CSV -- otherwise a
    StandSkipped outcome would leave a stray, unreferenced CSV behind,
    contradicting the reported result."""
    # plan_stand_outputs reads the unscaled strata, on purpose: a positive
    # factor can't change whether the subdominant species has basal area, so
    # the plan is the same scaled or not -- and it stays the one a dry run
    # reports, which never sees a scaled stratum.
    plan = plan_stand_outputs(stand, output_dir)
    try:
        # Two sets of strata on purpose. stand.strata is what the inventory
        # recorded, and is all that goes into the StandData record below. The
        # growth model gets strata_to_grow: the scaled strata when there is a
        # factor, the recorded ones otherwise. The scaling is applied here,
        # to the whole stand, BEFORE build_growth_and_yield_table isolates a
        # layer: the rule judged the stand on all three species, so all three
        # are multiplied by the one factor, including a third species that
        # neither layer keeps.
        strata_to_grow = strata_to_grow_from(stand.strata, scaling_factor)

        dominant_table = build_growth_and_yield_table(
            strata_to_grow,
            stand.dominant_species,
            stand.fertilityclass,
            stand.x_ykj,
            stand.y_ykj,
            config.altitude,
            config.ddy,
            config.n_trees,
            config.start_year,
            config.end_year,
            config.step_years,
        )

        subdominant_table: pd.DataFrame | None = None
        if plan.subdominant_csv is not None:
            subdominant_table = build_growth_and_yield_table(
                strata_to_grow,
                stand.subdominant_species,
                stand.fertilityclass,
                stand.x_ykj,
                stand.y_ykj,
                config.altitude,
                config.ddy,
                config.n_trees,
                config.start_year,
                config.end_year,
                config.step_years,
            )

        # Each layer starts at the age its own curve starts at: the growth
        # model's basal-area-weighted pooled age of the strata it was grown
        # from. Each layer is one species grown alone, so that's the species'
        # own stratum age -- never a pooled stand-level age -- as long as the
        # stratum age is already a whole number
        # (aggregate_species_group rounds it; the growth model rounds again).
        # It's read off the table either way, the same way
        # xml_to_allometry.py does, so it's always the curve's real start.
        initial_age_per_layer: dict[CanopyLayerName, float] = {
            CanopyLayerName.dominant: initial_age(dominant_table),
        }
        if subdominant_table is not None:
            initial_age_per_layer[CanopyLayerName.subdominant] = initial_age(
                subdominant_table
            )

        stand_data = build_stand_data(
            stand=stand,
            plan=plan,
            initial_age_per_layer=initial_age_per_layer,
            scaling_factor=scaling_factor,
        )

        # Both tables computed successfully (or there is no subdominant
        # layer to compute), and the StandData record above validated --
        # only now do we write anything to disk.
        write_allometry_csv(dominant_table, plan.dominant_csv)

        if subdominant_table is not None:
            assert plan.subdominant_csv is not None
            write_allometry_csv(subdominant_table, plan.subdominant_csv)

        return StandWritten(
            stand_id=plan.stand_id,
            stand_data=stand_data,
            dominant_csv=plan.dominant_csv,
            subdominant_csv=plan.subdominant_csv,
        )
    except Exception as error:  # noqa: BLE001 -- deliberately broad: any per-stand failure becomes a skip, not a run-aborting exception
        return StandSkipped(stand_id=stand.id, reason=str(error))


# %% CLI

# valid_existing_directory (for --project-dir) and
# make_existing_file_validator(".gpkg"/".toml") are now the shared
# tools.shared_allometry_tool_utils functions imported above -- these used
# to be local, hand-written copies (valid_project_dir_path, valid_gpkg_path,
# valid_config_path). Where the output lands is likewise not decided here:
# allometry_dir_for_project comes from susi.io.project_layout.


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Metsäkeskus .gpkg stand data (2026 data model) to SUSI "
            "allometry CSV inputs"
        )
    )
    parser.add_argument(
        "input_gpkg",
        type=make_existing_file_validator(".gpkg"),
        help=(
            "Path to the Metsäkeskus .gpkg file, in the 2026 data model. An "
            "old-format file is refused, with a pointer to the tool that "
            "reads it."
        ),
    )
    parser.add_argument(
        "--config",
        type=make_existing_file_validator(".toml"),
        default=None,
        help=(
            "Path to the TOML config file. Defaults to "
            f"{CONFIG_FILENAME} inside the project's inputs/ folder."
        ),
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        type=valid_existing_directory,
        help=(
            "Path to the project's folder -- the project root, the folder "
            "holding its inputs/ and outputs/. Decides the output directory, "
            "<project-dir>/inputs/allometry/, and -- unless --config is "
            "given -- where the config file is looked up: "
            f"<project-dir>/inputs/{CONFIG_FILENAME}."
        ),
    )
    parser.add_argument(
        "--allow-out-of-range-values",
        action="store_true",
        help=(
            "Allow the config file's altitude/ddy values outside their enforced range "
            "instead of blocking. Out-of-range values are still printed as a warning."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report what the run would produce, then exit having written "
            "nothing at all: no CSVs, no JSON, not even the output "
            "folder. Stops before the growth model, which is what makes a "
            "real run slow, so the filter report arrives in a fraction of "
            "the time."
        ),
    )

    args = parser.parse_args()

    # Checked first, from the layer list and column names alone: a file in
    # the wrong format makes everything below moot.
    format_problem = gpkg_format_problem(args.input_gpkg)
    if format_problem is not None:
        parser.error(format_problem)

    # --config defaults to CONFIG_FILENAME inside the project's inputs/
    # folder -- the layout the docs have the user set up beforehand.
    config_path = resolve_config_path(
        args.config, args.project_dir, parser, CONFIG_FILENAME
    )
    config = load_config_or_exit(parser, config_path, load_extraction_config)

    # Validates config.altitude/ddy and refuses to reuse an existing output
    # folder -- both checked here (dry-run mode included: "would this run
    # even start?" is exactly what a dry run is for) so a run that fails
    # while reading or filtering leaves no empty folder behind. Creating the
    # folder is main()'s job, and only on a real run.
    finalize_cli_config(
        parser,
        config.altitude,
        config.ddy,
        args.project_dir,
        args.allow_out_of_range_values,
    )

    return CLIArguments(
        input_gpkg=args.input_gpkg,
        config=config,
        config_path=config_path,
        project_dir=args.project_dir,
        allow_out_of_range_values=args.allow_out_of_range_values,
        dry_run=args.dry_run,
    )


# %% Progress-report printing (side-effecting; kept out of the pure layer above)

# print_section/print_skips are now the shared
# tools.shared_allometry_tool_utils.print_formatting functions, imported
# above -- these used to be local copies.


def print_year_distribution(year_distribution: pd.DataFrame) -> None:
    print(
        "Available inventory years (treestandclass = 1): stands per year, "
        "and per inventory source (treestanddatasource):"
    )
    print(year_distribution.to_string(index=False))
    print(
        "  Sources: 1 = field-measured, 2 = remote-sensed, 9 = combined from "
        "several sources, 21-23 = reported by the owner, an authority or an "
        "operator. Every code is listed in the .gpkg's own codeschema_mapping "
        "table."
    )
    print()


def print_mean_diameter_by_developmentclass(summary: pd.DataFrame) -> None:
    print(
        "   Mean diameter (cm) by development class, to help choose "
        "mean_diameter_range_filter for your region:"
    )
    if summary.empty:
        print("   (no stands)")
    else:
        print(
            "\n".join(
                f"   {line}" for line in summary.to_string(index=False).splitlines()
            )
        )
    print(
        "   (Development-class codes: the .gpkg's codeschema_mapping table, "
        "field developmentclass.)"
    )
    print()


def csv_counts(outputs: list[StandPlanned] | list[StandWritten]) -> tuple[int, int]:
    """(dominant, subdominant) CSV counts. Takes either record because the two
    carry the same pair of path fields, which is what lets the dry run's
    report and the real run's report be counted the same way instead of each
    doing its own arithmetic -- and therefore be compared to each other."""
    dominant = len(outputs)
    subdominant = sum(1 for output in outputs if output.subdominant_csv is not None)
    return dominant, subdominant


def print_dry_run_plan(
    plans: list[StandPlanned],
    output_dir: Path,
    project_dir: Path,
) -> None:
    """The dry run's stand-in for the real run's writing report: the same
    counts and the same file names, with nothing on disk. The folder is named
    as the one that WOULD be created -- parse_CLI_arguments has already
    refused the run if it exists, so this path is known to be free. The CSVs
    go in output_dir (<project-dir>/inputs/allometry/), the JSON next to that
    folder in <project-dir>/inputs/ -- same split as the real run's."""
    dominant, subdominant = csv_counts(plans)
    json_path = stand_data_path_for_project(project_dir)

    print(f"Destination folder: {output_dir.resolve()} (not created)")
    print()
    print(
        f"Would write: {len(plans):,} stand(s) -- {dominant:,} dominant + "
        f"{subdominant:,} subdominant = {dominant + subdominant:,} CSV(s)"
    )
    print(f"Would write informational JSON: {json_path}")
    print()
    # The dry run stops before build_growth_and_yield_table, so the failures
    # process_stand would catch (growth model or coordinate math raising for
    # one stand) cannot be known here. Every filter-level skip above IS real:
    # those stages all ran.
    print(
        "Note: a dry run stops before the growth model, so per-stand "
        "growth-model failures are not detected. The counts above are an "
        "upper bound."
    )


# %% main


def main() -> None:
    # Set before parse_CLI_arguments, whose format check already reads the
    # file. Process-wide, which is what a CLI's entry point wants.
    warnings.filterwarnings("ignore", message=MEASURED_GEOMETRY_WARNING)
    cli_args = parse_CLI_arguments()
    output_dir = allometry_dir_for_project(cli_args.project_dir)

    print_section("Reading")
    print("Tool initialized with:")
    print(f"    - input_gpkg  = {cli_args.input_gpkg.resolve()}")
    print(f"    - project_dir = {cli_args.project_dir.resolve()}")
    print(f"    - config      = {cli_args.config_path.resolve()}")
    print(f"    - output_dir  = {output_dir.resolve()}")
    print(f"    - target_year = {cli_args.config.target_year}")
    print(f"    - altitude    = {cli_args.config.altitude}")
    print(f"    - ddy         = {cli_args.config.ddy}")
    minimum, maximum = cli_args.config.mean_diameter_range_filter
    print(f"    - mean_diameter_range_filter = {minimum:g}-{maximum:g} cm")
    if cli_args.dry_run:
        print("    - DRY RUN -- nothing will be written")
    print()

    layers = load_gpkg_layers(cli_args.input_gpkg)
    stand_in_source_crs = stand_layer_in_source_crs(layers.stand)
    if stand_in_source_crs is not layers.stand:
        print(f"stand layer reprojected: {layers.stand.crs} -> {SOURCE_CRS}")
    layers = dataclasses.replace(layers, stand=stand_in_source_crs)
    print(f"stand          : {len(layers.stand):>9,} rows")
    print(f"growthplacedata: {len(layers.growthplacedata):>9,} rows")
    print(f"treestand      : {len(layers.treestand):>9,} rows")
    print(f"treestratum    : {len(layers.treestratum):>9,} rows")

    print_section("Filtering")

    print(
        "1. Site filter -- forest land, peatland (Korpi/Räme), already drained, "
        f"fertility class in {cli_args.config.fertilityclass_filter}:"
    )
    # growthplacedata holds one row per stand: it is the site filter's input.
    filtered_growthplacedata = filter_stands_by_site_attributes(
        layers.growthplacedata, cli_args.config.fertilityclass_filter
    )
    print(
        f"   -> {len(filtered_growthplacedata):,} / "
        f"{len(layers.growthplacedata):,} stands kept"
    )
    print()

    stand_ids = set(
        pd.to_numeric(filtered_growthplacedata["standid"], errors="coerce")
        .dropna()
        .astype(int)
    )
    print_year_distribution(compute_year_distribution(layers.treestand, stand_ids))

    print(
        f"2. Inventory-snapshot filter -- snapshot dated in "
        f"{cli_args.config.target_year}:"
    )
    snapshot = select_target_year_snapshot(
        layers.treestand, stand_ids, cli_args.config.target_year
    )
    n_excluded = len(stand_ids) - len(snapshot)
    print(f"   -> {len(snapshot):,} / {len(stand_ids):,} stands kept")
    if n_excluded > 0:
        print(
            f"      ({n_excluded:,} stand(s) had no inventory snapshot dated in "
            f"{cli_args.config.target_year} and are excluded)"
        )
    print()

    # A stand with more than one row in growthplacedata or the stand layer
    # would come out of the join below twice: dropped first, and reported
    # with step 3's skips.
    unique_snapshot, duplicate_skips = drop_duplicated_stands(
        snapshot, layers.growthplacedata, layers.stand
    )
    merged = attach_stand_attributes(
        unique_snapshot, filtered_growthplacedata, layers.stand
    )

    print(
        "3. Structural + species-data checks -- duplicated rows, geometry, "
        "treestandid, diameter/height:"
    )
    candidates, candidate_skips = build_stand_candidates(merged, layers.treestratum)
    print(f"   -> {len(candidates):,} / {len(snapshot):,} stands kept")
    # structural_skips covers a stand with duplicated rows
    # (drop_duplicated_stands), missing structural fields (no usable
    # geometry, no treestandid) and a species with real basal area but no
    # usable diameter/height (DegenerateSpeciesDataError) -- build_stand_candidates.
    structural_skips = duplicate_skips + candidate_skips
    print_skips(structural_skips, "   Skipped (invalid or missing data)")
    print()

    print("4. Viability check -- nonzero total basal area:")
    viable_candidates, ba_skips = partition_viable_candidates(candidates)
    print(f"   -> {len(viable_candidates):,} / {len(candidates):,} stands kept")
    print_skips(ba_skips, "   Skipped (zero basal area)")
    print()

    print(
        f"5. Mean-diameter filter -- {minimum:g}-{maximum:g} cm "
        "(mean_diameter_range_filter):"
    )
    # Printed before filtering, over every viable stand: the table shows where
    # the range cuts, not just what survived it.
    print_mean_diameter_by_developmentclass(
        summarize_mean_diameter_by_developmentclass(viable_candidates)
    )
    in_range_candidates, diameter_skips = filter_by_mean_diameter_range(
        viable_candidates, cli_args.config.mean_diameter_range_filter
    )
    print(
        f"   -> {len(in_range_candidates):,} / {len(viable_candidates):,} stands kept"
    )
    print_skips(diameter_skips, "   Skipped (mean diameter outside the range)")

    parsed_stands, build_skips = build_stands(in_range_candidates)
    print_skips(build_skips, "   Skipped (could not build stand record)")
    print()

    print(f"Stands ready for allometry: {len(parsed_stands):,}")

    # Dense young stand scaling. Every stand's scaling factor is decided
    # once, here: None for all of them unless the config file turns the
    # option on. It is decided before the dry-run return below, so a dry run
    # reports the scaled stands and the warning exactly as a real run does.
    #
    # The rule sees each stand's full strata -- all three species, before
    # process_stand splits the stand into its dominant and subdominant
    # layers -- so a stand is judged, and scaled, as a whole.
    strata_per_stand = {stand.id: stand.strata for stand in parsed_stands}
    scaling_factors = decide_scaling_factors(
        strata_per_stand, cli_args.config.dense_young_stand_scaling
    )
    print_scaled_stands(strata_per_stand, scaling_factors)
    # The warning: stands the DEFAULT numbers would scale, judged on the
    # strata each stand is about to be grown from. With the option off, those
    # are the dense young stands the user may not know about. It never blocks
    # the run.
    print_dense_young_stand_warning(
        stands_the_default_rule_would_scale(strata_per_stand, scaling_factors)
    )

    # Everything above this point is identical in a dry run: the filters and
    # the skip reports are precisely what a dry run exists to show. What it
    # skips is everything below -- the growth model (~all of a real run's
    # time) and every write, the output folder included.
    if cli_args.dry_run:
        print_section("Writing (dry run -- nothing is written)")
        plans = [plan_stand_outputs(stand, output_dir) for stand in parsed_stands]
        print_dry_run_plan(plans, output_dir, cli_args.project_dir)
        return

    print_section("Writing")
    print(f"Destination folder: {output_dir.resolve()}")
    # Created here rather than at argument-parsing time, so a run that fails
    # while reading or filtering leaves no empty folder behind to block the
    # next attempt.
    output_dir.mkdir(parents=True)
    print()

    outcomes = [
        process_stand(
            stand,
            cli_args.config,
            output_dir,
            scaling_factor=scaling_factors[stand.id],
        )
        for stand in parsed_stands
    ]
    written: list[StandWritten] = [o for o in outcomes if isinstance(o, StandWritten)]
    processing_skips: list[StandSkipped] = [
        o for o in outcomes if isinstance(o, StandSkipped)
    ]

    # Same breakdown, and the same csv_counts, as the dry run reports -- so a
    # dry run and the real run that follows it can be read against each other.
    dominant, subdominant = csv_counts(written)
    print(
        f"Allometry files written: {len(written):,} stand(s) -- {dominant:,} "
        f"dominant + {subdominant:,} subdominant = "
        f"{dominant + subdominant:,} CSV(s)"
    )
    print_skips(processing_skips, "Skipped (processing failure)")

    final_stands: dict[StandID, StandData] = {
        outcome.stand_id: outcome.stand_data for outcome in written
    }
    json_path = stand_data_path_for_project(cli_args.project_dir)
    dump_stand_data_document(
        output_path=json_path,
        document=StandDataDocument(
            # stand_layer_in_source_crs has already reprojected the layer into it.
            crs=SOURCE_CRS,
            altitude=cli_args.config.altitude,
            ddy=cli_args.config.ddy,
            stands=final_stands,
        ),
    )
    print(f"Informational JSON written: {json_path}")


if __name__ == "__main__":
    main()
