"""
Reads Metsäkeskus forest inventory data (.gpkg)
--> Filters the data (more details below)
--> Generates SUSI allometry files

Generalizes Sandeep's script into a tool, following the conventions
Unlike in that script, there is no representative sampling here, so
every stand that survives filtering is turned into an allometry CSV file.
"""

# %% Imports
import argparse
import dataclasses
import json
import math
import tomllib
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Generic, TypeVar

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

from susi.core.allometric_road_map import Growth_and_Yield_Table
from susi.io.load_output_data import StandID
from tools.xml_to_allometry.xml_to_allometry import out_of_range_message

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

# xml_to_allometry.py's own enforced-range bounds for altitude/ddy, reused
# here for the same reason (catch a mistyped value, e.g. metres vs feet).
ALTITUDE_MIN = 0.0
ALTITUDE_MAX = 1000.0
DDY_MIN = 500.0
DDY_MAX = 2000.0

# treestand.type: whether the data is measured or projected date.
# Types 2 and 3 are Metsäkeskus's own grown-forward projections
# (2: to a common "current" date, 3: +10y future extrapolation)
# Type 1 is the only measured data.
TREESTAND_MEASURED_TYPE = 1

SOURCE_CRS = "EPSG:3067"  # ETRS-TM35FIN, the CRS Metsäkeskus geometries ship in
YKJ_CRS = "EPSG:2393"  # Finnish YKJ grid, what Growth_and_Yield_Table's x/y expect

# Fields required in the config file
REQUIRED_CONFIG_FIELDS = ("target_year", "altitude", "ddy")


# %% dataclasses

T = TypeVar("T")


@dataclass(frozen=True)
class PerSpecies(Generic[T]):
    """One value per SUSI growth-model species slot. Mirrors PerNutrient in
    susi/core/allometry.py -- same 'three independently-spelled fields ->
    one X[T] field' pattern, one slot per species instead of per macronutrient."""

    pine: T
    spruce: T
    deciduous: T


@dataclass(frozen=True)
class TreeStratum:
    """
    One species' aggregated growth-model inputs for one stand.
    Same shape as xml_to_allometry.py's TreeStratum
    """

    age: int
    basal_area: float
    stem_count: int
    mean_diameter: float
    mean_height: float


# A TreeStratum for "nothing recorded here"
_ZERO_STRATUM = TreeStratum(
    age=0,
    basal_area=0.0,
    stem_count=0,
    mean_diameter=0.0,
    mean_height=0.0,
)


@dataclass(frozen=True)
class StandSiteAttributes:
    """Site-level facts a stand carries into the output. soiltype is
    Optional -- unlike drainagestate/fertilityclass/etc., it never feeds
    Growth_and_Yield_Table (search build_growth_and_yield_table: it isn't
    one of the parameters), so there is nothing to fabricate a value for.
    A missing soiltype stays None end-to-end (JSON dump: null; XML dump:
    the <st:SoilType> tag is simply omitted, and xml_to_allometry.py's
    reader already tolerates that) rather than being reported as a specific,
    invented number indistinguishable from a real measurement."""

    subgroup: int
    fertilityclass: int
    developmentclass: int
    drainagestate: int
    soiltype: int | None


@dataclass(frozen=True)
class StandCandidate:
    """
    Represents one stand after merge + species aggregation, before the viability
    check (partition_viable_candidates) that decides whether it's actually
    processed.
    """

    id: StandID
    site: StandSiteAttributes
    strata: PerSpecies[TreeStratum]
    geometry: BaseGeometry


@dataclass(frozen=True)
class ValidStand:
    """
    One stand ready for Growth_and_Yield_Table.
    """

    id: StandID
    site: StandSiteAttributes
    strata: PerSpecies[TreeStratum]
    stand_meanage: float
    stand_basalarea: float
    stand_meanheight: float
    stand_meandiameter: float
    x_ykj: int
    y_ykj: int
    geometry: BaseGeometry
    dominant_species: int
    subdominant_species: int


@dataclass(frozen=True)
class GpkgLayers:
    """
    The three gpkg layers this tool reads.
    treestandsummary is deliberately absent:
    it only has any values for Metsäkeskus future projections, not for measurements
    """

    stand: gpd.GeoDataFrame
    treestand: pd.DataFrame
    treestratum: pd.DataFrame


@dataclass(frozen=True)
class ExtractionConfig:
    """Defaulted/required parameters, loaded from a TOML file (see
    parse_extraction_config). Hard-coded, non-negotiable parameters live as
    module constants above instead."""

    # Required, no defaults
    target_year: int
    altitude: float
    ddy: float

    # Defaulted

    # 1 = open/seedling, 2 = young growing, 3 = grown-up
    developmentclass_filter: tuple[int, ...] = (1, 2, 3)

    # peatland fertility classes of interest
    # restricting to 2-5 excludes the very richest and very poorest extremes.
    fertilityclass_filter: tuple[int, ...] = (2, 3, 4, 5)
    n_trees: int = 20
    start_year: int = 5
    end_year: int = 80
    step_years: int = 5


@dataclass(frozen=True)
class CLIArguments:
    input_gpkg: Path
    config: ExtractionConfig
    config_path: Path
    project_dir: Path
    allow_out_of_range_values: bool
    emit_xml: bool
    dry_run: bool


@dataclass(frozen=True)
class StandPlanned:
    """The files one stand WOULD produce -- a fact about the stand and the
    output folder, knowable before any growth table is computed (see
    plan_stand_outputs)"""

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
    dominant_csv: Path
    # When the second species carries zero basal area, no subdominant allometry file is written.
    subdominant_csv: Path | None


@dataclass(frozen=True)
class StandSkipped:
    """
    A stand that does not survives the filters and the checks,
    so no allometry file gets written.
    """

    stand_id: StandID
    reason: str


StandOutcome = StandWritten | StandSkipped


# %% Config parsing


def parse_extraction_config(raw: dict) -> ExtractionConfig:
    """Build an ExtractionConfig from a parsed TOML dict, collecting every
    missing-required-field violation before raising -- same one-shot
    reporting style as xml_to_allometry.py's altitude/ddy range checks."""
    missing = [name for name in REQUIRED_CONFIG_FIELDS if name not in raw]
    if missing:
        raise ValueError(
            f"Config file is missing required field(s): {', '.join(missing)}"
        )

    known_fields = {f.name for f in fields(ExtractionConfig)}
    unexpected = sorted(set(raw) - known_fields)
    if unexpected:
        raise ValueError(f"Config file has unknown field(s): {', '.join(unexpected)}")

    return ExtractionConfig(
        target_year=int(raw["target_year"]),
        altitude=float(raw["altitude"]),
        ddy=float(raw["ddy"]),
        developmentclass_filter=tuple(raw.get("developmentclass_filter", (1, 2, 3))),
        fertilityclass_filter=tuple(raw.get("fertilityclass_filter", (2, 3, 4, 5))),
        n_trees=int(raw.get("n_trees", 20)),
        start_year=int(raw.get("start_year", 5)),
        end_year=int(raw.get("end_year", 80)),
        step_years=int(raw.get("step_years", 5)),
    )


def load_extraction_config(config_path: Path) -> ExtractionConfig:
    with open(config_path, "rb") as config_file:
        raw = tomllib.load(config_file)
    return parse_extraction_config(raw)


# %% gpkg loading (I/O)


def load_gpkg_layers(gpkg_path: Path) -> GpkgLayers:
    """Loads the three gpkg layers this tool needs. See the module docstring
    on GpkgLayers for why treestandsummary is not among them."""
    return GpkgLayers(
        stand=gpd.read_file(gpkg_path, layer="stand"),
        treestand=gpd.read_file(gpkg_path, layer="treestand"),
        treestratum=gpd.read_file(gpkg_path, layer="treestratum"),
    )


# %% Filtering


def filter_stands_by_site_attributes(
    stand: gpd.GeoDataFrame,
    fertilityclass_filter: tuple[int, ...],
) -> gpd.GeoDataFrame:
    """Restrict to forest land, on peatland, already drained (hard-coded --
    see MAINGROUP_FOREST_LAND / SUBGROUP_PEATLAND / DRAINAGESTATE_DRAINED
    above), within the configured fertility-class range."""
    numeric = stand.copy()
    for column in ("maingroup", "subgroup", "drainagestate", "fertilityclass"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    mask = (
        (numeric["maingroup"] == MAINGROUP_FOREST_LAND)
        & numeric["subgroup"].isin(SUBGROUP_PEATLAND)
        & numeric["drainagestate"].isin(DRAINAGESTATE_DRAINED)
        & numeric["fertilityclass"].isin(fertilityclass_filter)
    )
    return numeric[mask].copy()


def _with_parsed_measurement_columns(
    treestand: pd.DataFrame, stand_ids: set[int]
) -> pd.DataFrame:
    """Shared prep for select_target_year_snapshot and compute_year_distribution:
    restrict to the given stands' rows and parse date/type/year."""
    candidates = treestand[treestand["standid"].isin(stand_ids)].copy()
    candidates["date_dt"] = pd.to_datetime(candidates["date"], errors="coerce")
    candidates["type_num"] = pd.to_numeric(candidates["type"], errors="coerce")
    candidates["year"] = candidates["date_dt"].dt.year
    return candidates


def compute_year_distribution(
    treestand: pd.DataFrame, stand_ids: set[int]
) -> pd.DataFrame:
    """Per-year count of distinct stands with a measured (type=1) snapshot --
    for the progress report, mirrors metsakeskus.py's own year-availability
    printout."""
    candidates = _with_parsed_measurement_columns(treestand, stand_ids)
    measured = candidates[candidates["type_num"] == TREESTAND_MEASURED_TYPE]
    return (
        measured.groupby("year")["standid"]
        .nunique()
        .reset_index()
        .rename(columns={"standid": "n_stands"})
        .sort_values("year")
    )


def select_target_year_snapshot(
    treestand: pd.DataFrame,
    stand_ids: set[int],
    target_year: int,
) -> pd.DataFrame:
    """One treestand row per stand: the type=1 (measured) record whose date
    falls exactly in target_year. A stand without one is simply absent from
    the result -- no '>= target_year, else next available year' fallback
    (matches the script's actual behavior, not its docstring's stated intent;
    see the grilling-session notes for why that fallback was deliberately not
    revived)."""
    candidates = _with_parsed_measurement_columns(treestand, stand_ids)
    measured = candidates[candidates["type_num"] == TREESTAND_MEASURED_TYPE]
    exact_year = measured[measured["year"] == target_year]

    return (
        exact_year.sort_values(["standid", "date_dt"])
        .drop_duplicates(subset="standid", keep="first")
        .reset_index(drop=True)
    )


def attach_stand_attributes(
    treestand_snapshot: pd.DataFrame,
    filtered_stand: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Re-attaches the stand-layer's site columns onto each selected
    treestand snapshot -- treestand itself carries no site metadata."""
    site_columns = [
        "standid",
        "subgroup",
        "fertilityclass",
        "drainagestate",
        "soiltype",
        "geometry",
        "developmentclass",
    ]
    present_columns = [c for c in site_columns if c in filtered_stand.columns]
    return treestand_snapshot.merge(
        filtered_stand[present_columns], on="standid", how="left"
    )


def filter_by_developmentclass(
    merged: pd.DataFrame,
    developmentclass_filter: tuple[int, ...],
) -> pd.DataFrame:
    result = merged.copy()
    result["developmentclass"] = pd.to_numeric(
        result["developmentclass"], errors="coerce"
    )
    return result[result["developmentclass"].isin(developmentclass_filter)].copy()


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
    positive measured basal area but no usable mean diameter and/or mean
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
    _ZERO_STRATUM and isolate_species_layer, which enforces the same "not
    passed forward" rule one level down, per canopy layer).

    A species entirely absent from this stand (rows empty) gets the flat
    _ZERO_STRATUM -- there is nothing recorded to preserve.

    A species present with rows summing to zero basal area keeps whatever
    was actually recorded: the real summed stem count, and age/diameter/
    height from a plain average of the recorded rows. Only a field with
    truly nothing to average (an all-NaN column) falls back to 0 -- not a
    fabricated placeholder, and deliberately not NaN either, since NaN would
    silently poison build_valid_stand's basal-area-weighted stand-level
    age/height/diameter (NaN * 0 is NaN, not 0).

    A species present with real, POSITIVE basal area but degenerate
    diameter/height data raises DegenerateSpeciesDataError instead (see that
    class's docstring): unlike the zero-basal-area case, this species can
    actually reach Growth_and_Yield_Table, so there is no safe value to
    invent. species_name is only used to name it in that error message."""
    if rows.empty:
        return _ZERO_STRATUM

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

        # This species has real, measured basal area -- it can end up as the
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
        age = int(round(weighted_age)) if pd.notna(weighted_age) else 0
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


def total_basal_area(strata: PerSpecies[TreeStratum]) -> float:
    return (
        strata.pine.basal_area + strata.spruce.basal_area + strata.deciduous.basal_area
    )


# %% Candidate assembly and viability


def build_stand_candidates(
    merged_filtered: pd.DataFrame,
    treestratum: pd.DataFrame,
) -> tuple[list[StandCandidate], list[StandSkipped]]:
    """One StandCandidate per row of merged_filtered. A row with no usable
    geometry is excluded here, as a StandSkipped -- not carried downstream as
    a None geometry. Same treatment for a species with real basal area but
    degenerate diameter/height data (build_species_strata's
    DegenerateSpeciesDataError, see aggregate_species_group) -- caught here
    and turned into a StandSkipped for the whole stand, rather than letting
    artificial data flow forward into a ValidStand."""
    candidates: list[StandCandidate] = []
    skipped: list[StandSkipped] = []
    treestratum_treestandid = pd.to_numeric(treestratum["treestandid"], errors="coerce")

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

        site = StandSiteAttributes(
            subgroup=int(row["subgroup"]),
            fertilityclass=int(row["fertilityclass"]),
            developmentclass=int(row["developmentclass"]),
            drainagestate=(
                int(row["drainagestate"])
                if pd.notna(row.get("drainagestate"))
                else DRAINAGESTATE_DRAINED[0]
            ),
            soiltype=(int(row["soiltype"]) if pd.notna(row.get("soiltype")) else None),
        )

        candidates.append(
            StandCandidate(id=stand_id, site=site, strata=strata, geometry=geometry)
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


# %% Building the final stand record


def determine_dominant_and_subdominant_species(
    strata: PerSpecies[TreeStratum],
) -> tuple[int, int]:
    """Species codes (1=pine, 2=spruce, 3=deciduous) ranked by basal area.
    The subdominant is always the second-ranked species' own stratum, even
    when its basal area is zero -- see docs/adr/0002 for why this
    deliberately does not duplicate the dominant species for a monoculture
    stand."""
    basal_areas = {
        1: strata.pine.basal_area,
        2: strata.spruce.basal_area,
        3: strata.deciduous.basal_area,
    }
    ranked = sorted(
        basal_areas, key=lambda species_code: basal_areas[species_code], reverse=True
    )
    return ranked[0], ranked[1]


@lru_cache(maxsize=1)
def _ykj_transformer() -> Transformer:
    """Built once and reused -- constructing a Transformer is comparatively
    expensive, and centroid_to_ykj runs once per viable stand (thousands per
    real run)."""
    return Transformer.from_crs(SOURCE_CRS, YKJ_CRS, always_xy=True)


def centroid_to_ykj(geometry: BaseGeometry) -> tuple[int, int]:
    """Stand-polygon centroid -> YKJ grid coordinates, scaled to the units
    Growth_and_Yield_Table expects (10 km easting units, 1 km northing units)."""
    transformer = _ykj_transformer()
    easting, northing = transformer.transform(geometry.centroid.x, geometry.centroid.y)
    return round(easting / 10000), round(northing / 1000)


def build_valid_stand(candidate: StandCandidate) -> ValidStand:
    """Builds the final stand record from an already-viable candidate
    (nonzero total basal area, guaranteed by partition_viable_candidates;
    non-empty geometry, guaranteed by build_stand_candidates). Not fully
    total, though: centroid_to_ykj can still raise for a geometry that's
    present and non-empty but otherwise degenerate (e.g. all-coincident
    points) -- callers processing a batch should use build_valid_stands,
    which isolates that per candidate instead of aborting the whole run."""
    strata = candidate.strata
    stand_total_ba = total_basal_area(strata)

    # Stand-level age/height/diameter: basal-area-weighted across the three
    # species slots (mirrors aggregate_species_group's weighting, one level up).
    stand_age = (
        strata.pine.age * strata.pine.basal_area
        + strata.spruce.age * strata.spruce.basal_area
        + strata.deciduous.age * strata.deciduous.basal_area
    ) / stand_total_ba
    stand_height = (
        strata.pine.mean_height * strata.pine.basal_area
        + strata.spruce.mean_height * strata.spruce.basal_area
        + strata.deciduous.mean_height * strata.deciduous.basal_area
    ) / stand_total_ba
    stand_diameter = (
        strata.pine.mean_diameter * strata.pine.basal_area
        + strata.spruce.mean_diameter * strata.spruce.basal_area
        + strata.deciduous.mean_diameter * strata.deciduous.basal_area
    ) / stand_total_ba

    dominant_species, subdominant_species = determine_dominant_and_subdominant_species(
        strata
    )
    x_ykj, y_ykj = centroid_to_ykj(candidate.geometry)

    return ValidStand(
        id=candidate.id,
        site=candidate.site,
        strata=strata,
        stand_meanage=float(stand_age),
        stand_basalarea=float(stand_total_ba),
        stand_meanheight=float(stand_height),
        stand_meandiameter=float(stand_diameter),
        x_ykj=x_ykj,
        y_ykj=y_ykj,
        geometry=candidate.geometry,
        dominant_species=dominant_species,
        subdominant_species=subdominant_species,
    )


def build_valid_stands(
    candidates: list[StandCandidate],
) -> tuple[list[ValidStand], list[StandSkipped]]:
    """Batch build_valid_stand, isolating one candidate's failure (see
    build_valid_stand's docstring) as a StandSkipped instead of letting it
    abort every other stand in the run -- the same per-stand isolation
    principle process_stand already applies to the growth-table stage."""
    built: list[ValidStand] = []
    skipped: list[StandSkipped] = []
    for candidate in candidates:
        try:
            built.append(build_valid_stand(candidate))
        except Exception as error:  # noqa: BLE001 -- one bad stand's geometry math must not abort the batch
            skipped.append(StandSkipped(stand_id=candidate.id, reason=str(error)))
    return built, skipped


# %% Growth-and-yield table construction

# _ZERO_STRATUM (used below by isolate_species_layer) is defined once, next
# to TreeStratum, and shared with aggregate_species_group -- see its
# docstring there.


def species_stratum(strata: PerSpecies[TreeStratum], species_code: int) -> TreeStratum:
    """The one TreeStratum a species code (1=pine, 2=spruce, 3=deciduous) refers to."""
    return {1: strata.pine, 2: strata.spruce, 3: strata.deciduous}[species_code]


def isolate_species_layer(
    strata: PerSpecies[TreeStratum], active_species: int
) -> PerSpecies[TreeStratum]:
    """Zeroes every species slot except active_species -- this is how a
    single canopy layer (dominant or subdominant) is modeled as that one
    species growing alone (see docs/adr/0002)."""
    if active_species == 1:
        return PerSpecies(
            pine=strata.pine, spruce=_ZERO_STRATUM, deciduous=_ZERO_STRATUM
        )
    if active_species == 2:
        return PerSpecies(
            pine=_ZERO_STRATUM, spruce=strata.spruce, deciduous=_ZERO_STRATUM
        )
    return PerSpecies(
        pine=_ZERO_STRATUM, spruce=_ZERO_STRATUM, deciduous=strata.deciduous
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
    rows from start_year to end_year), modeled as active_species growing alone."""
    layer = isolate_species_layer(strata, active_species)

    growth_and_yield_table = Growth_and_Yield_Table(
        age_1=layer.pine.age,
        G_1=layer.pine.basal_area,
        N_1=layer.pine.stem_count,
        Dg_1=layer.pine.mean_diameter,
        Hg_1=layer.pine.mean_height,
        age_2=layer.spruce.age,
        G_2=layer.spruce.basal_area,
        N_2=layer.spruce.stem_count,
        Dg_2=layer.spruce.mean_diameter,
        Hg_2=layer.spruce.mean_height,
        age_3=layer.deciduous.age,
        G_3=layer.deciduous.basal_area,
        N_3=layer.deciduous.stem_count,
        Dg_3=layer.deciduous.mean_diameter,
        Hg_3=layer.deciduous.mean_height,
        DDY=ddy,
        fertility_class=fertility_class,
        peat=PEAT,
        y=y_ykj,
        x=x_ykj,
        altitude=altitude,
        n_trees=n_trees,
    )
    return growth_and_yield_table.get_table(
        start_year=start_year, end_year=end_year, step_years=step_years
    )


# %% Writing output (I/O)


def write_allometry_csv(
    table: pd.DataFrame, species_id: int, output_path: Path
) -> None:
    """Writes one CanopyLayerAllometry-contract CSV -- readable directly by
    susi.io.susi_parameter_model.read_allometry_info_from_csv."""
    table_with_species = table.copy()
    table_with_species.insert(0, "Species_ID", species_id)
    table_with_species.to_csv(output_path, index=False)


def valid_stand_to_json_dict(stand: ValidStand) -> dict:
    data = dataclasses.asdict(stand)
    data["id"] = str(stand.id)
    data["geometry"] = stand.geometry.wkt
    return data


def dump_valid_stands_json(stands: list[ValidStand], output_path: Path) -> None:
    """Informational dump of every processed stand's data, including the
    filter/stratum columns -- mirrors xml_to_allometry.py's
    extra_XML_info.json for anyone who wants to regroup or audit later."""
    payload = {"stands": [valid_stand_to_json_dict(stand) for stand in stands]}
    output_path.write_text(json.dumps(payload, indent=2))


def _stratum_xml_block(stratum: TreeStratum, tree_species_code: int) -> str:
    return (
        "        <tst:TreeStratum>\n"
        f"          <tst:TreeSpecies>{tree_species_code}</tst:TreeSpecies>\n"
        f"          <tst:Age>{stratum.age}</tst:Age>\n"
        f"          <tst:BasalArea>{stratum.basal_area:.6f}</tst:BasalArea>\n"
        f"          <tst:StemCount>{stratum.stem_count}</tst:StemCount>\n"
        f"          <tst:MeanDiameter>{stratum.mean_diameter:.4f}</tst:MeanDiameter>\n"
        f"          <tst:MeanHeight>{stratum.mean_height:.4f}</tst:MeanHeight>\n"
        "        </tst:TreeStratum>\n"
    )


def stand_to_xml_block(stand: ValidStand) -> str:
    """Renders one stand as a <st:Stand> block in the Finnish
    ForestPropertyData schema -- the format xml_to_allometry.py reads.

    soiltype is Optional (see StandSiteAttributes): when it's None, the
    <st:SoilType> tag is omitted entirely rather than writing a fabricated
    number. xml_to_allometry.py's reader already tolerates a missing tag
    (reads it back as None, matching its own soil_type: Optional[int])."""
    coords_str = " ".join(f"{x},{y}" for x, y in stand.geometry.exterior.coords)
    strata_block = (
        _stratum_xml_block(stand.strata.pine, 1)
        + _stratum_xml_block(stand.strata.spruce, 2)
        + _stratum_xml_block(stand.strata.deciduous, 3)
    )
    soiltype_line = (
        f"        <st:SoilType>{stand.site.soiltype}</st:SoilType>\n"
        if stand.site.soiltype is not None
        else ""
    )
    return (
        f'    <st:Stand id="{stand.id}">\n'
        "      <st:StandBasicData>\n"
        f"        <st:FertilityClass>{stand.site.fertilityclass}</st:FertilityClass>\n"
        f"        <st:MainGroup>{MAINGROUP_FOREST_LAND}</st:MainGroup>\n"
        f"        <st:SubGroup>{stand.site.subgroup}</st:SubGroup>\n"
        f"{soiltype_line}"
        f"        <st:DrainageState>{stand.site.drainagestate}</st:DrainageState>\n"
        f"        <st:Area>0</st:Area>\n"
        "        <gdt:PolygonGeometry>\n"
        "          <gml:polygonProperty>\n"
        "            <gml:Polygon>\n"
        "              <gml:exterior>\n"
        "                <gml:LinearRing>\n"
        f"                  <gml:coordinates>{coords_str}</gml:coordinates>\n"
        "                </gml:LinearRing>\n"
        "              </gml:exterior>\n"
        "            </gml:Polygon>\n"
        "          </gml:polygonProperty>\n"
        "        </gdt:PolygonGeometry>\n"
        "      </st:StandBasicData>\n"
        "      <ts:TreeStandData>\n"
        "        <ts:TreeStandDataDate>\n"
        "          <tss:TreeStandSummary>\n"
        f"            <tss:MeanAge>{round(stand.stand_meanage)}</tss:MeanAge>\n"
        f"            <tss:BasalArea>{stand.stand_basalarea:.6f}</tss:BasalArea>\n"
        f"            <tss:MeanHeight>{stand.stand_meanheight:.4f}</tss:MeanHeight>\n"
        f"            <tss:MeanDiameter>{stand.stand_meandiameter:.4f}</tss:MeanDiameter>\n"
        "            <tss:Volume>0.0</tss:Volume>\n"
        "          </tss:TreeStandSummary>\n"
        "          <tst:TreeStrata>\n"
        f"{strata_block}"
        "          </tst:TreeStrata>\n"
        "        </ts:TreeStandDataDate>\n"
        "      </ts:TreeStandData>\n"
        "    </st:Stand>\n"
    )


def write_stands_xml(stands: list[ValidStand], output_path: Path) -> None:
    """Writes every processed stand into ONE ForestPropertyData XML file, not
    one file per stand: xml_to_allometry.py's own reader (xmltodict) returns
    a dict instead of a list when a document contains exactly one <st:Stand>,
    which would break the very replay this file exists for. A per-run batch
    file sidesteps that (a run with a single stand is the one remaining edge
    case, same as documented in xml_to_allometry.py)."""
    stands_xml = "".join(stand_to_xml_block(stand) for stand in stands)
    output_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<ForestPropertyData>\n"
        "  <st:Stands>\n"
        f"{stands_xml}"
        "  </st:Stands>\n"
        "</ForestPropertyData>\n",
        encoding="utf-8",
    )


# %% Per-stand orchestration


def plan_stand_outputs(stand: ValidStand, output_dir: Path) -> StandPlanned:
    """Which files this stand produces, and where. Pure, total, and cheap: it
    reads only what partition_viable_candidates and build_valid_stand have
    already settled, so it can run long before the growth model does.

    Both paths go through here -- process_stand for a real run, main() for a
    dry run -- so the dry run reports the real run's own decision rather than
    a parallel reimplementation of the same naming rules that could drift
    from it.

    The subdominant file exists exactly when the second-ranked species carries
    basal area of its own; see docs/adr/0002 for why a monoculture's
    subdominant layer is not written at all rather than duplicating the
    dominant one."""
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


def process_stand(
    stand: ValidStand, config: ExtractionConfig, output_dir: Path
) -> StandOutcome:
    """Builds the dominant canopy layer's allometry CSV (always) and the
    subdominant's (only when a genuine second species is present -- see
    StandWritten.subdominant_csv). Any failure here is this one stand's
    problem, not the run's: it's caught and turned into a StandSkipped
    rather than aborting the batch.

    Both growth tables are computed in full BEFORE either is written to
    disk: if the subdominant table's computation fails, we must not have
    already written the dominant CSV -- otherwise a StandSkipped outcome
    would leave a stray, unreferenced CSV behind, contradicting the
    reported result."""
    plan = plan_stand_outputs(stand, output_dir)
    try:
        dominant_table = build_growth_and_yield_table(
            stand.strata,
            stand.dominant_species,
            stand.site.fertilityclass,
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
                stand.strata,
                stand.subdominant_species,
                stand.site.fertilityclass,
                stand.x_ykj,
                stand.y_ykj,
                config.altitude,
                config.ddy,
                config.n_trees,
                config.start_year,
                config.end_year,
                config.step_years,
            )

        # Both tables computed successfully (or there is no subdominant
        # layer to compute) -- only now do we write anything to disk.
        write_allometry_csv(dominant_table, stand.dominant_species, plan.dominant_csv)

        if subdominant_table is not None:
            # plan.subdominant_csv is not None here: it is the very condition
            # that produced subdominant_table above.
            write_allometry_csv(
                subdominant_table, stand.subdominant_species, plan.subdominant_csv
            )

        return StandWritten(
            stand_id=plan.stand_id,
            dominant_csv=plan.dominant_csv,
            subdominant_csv=plan.subdominant_csv,
        )
    except Exception as error:  # noqa: BLE001 -- deliberately broad: any per-stand failure becomes a skip, not a run-aborting exception
        return StandSkipped(stand_id=stand.id, reason=str(error))


# %% CLI


def output_dir_for_project(project_dir: Path) -> Path:
    """Where a project's allometry files go. There is no flag to point the
    output somewhere else: appending allometry/ onto --project-dir is an
    invariant of this tool rather than a default."""
    return project_dir / "allometry"


def valid_project_dir_path(value: str) -> Path:
    """--project-dir is the ONLY thing deciding where files are written
    (output_dir_for_project) and -- unless --config overrides it -- where the
    config file is looked up (see parse_CLI_arguments's config_path.toml
    default). It must already exist as a real directory: the one the docs
    have the user set up beforehand, with the .gpkg and config.toml colocated
    inside it, not a bare name fed into a hard-coded inputs/ prefix."""
    if not value.strip():
        raise argparse.ArgumentTypeError("Project directory must not be empty")
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Project directory does not exist: {value}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Not a directory: {value}")
    return path


def valid_gpkg_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Not a file: {value}")
    if path.suffix.lower() != ".gpkg":
        raise argparse.ArgumentTypeError("File must have .gpkg extension")
    return path


def valid_config_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Config file does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Not a file: {value}")
    if path.suffix.lower() != ".toml":
        raise argparse.ArgumentTypeError("Config file must have .toml extension")
    return path


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description="Convert Metsäkeskus .gpkg stand data to SUSI allometry CSV inputs"
    )
    parser.add_argument(
        "input_gpkg", type=valid_gpkg_path, help="Path to the Metsäkeskus .gpkg file"
    )
    parser.add_argument(
        "--config",
        type=valid_config_path,
        default=None,
        help=(
            "Path to the TOML config file. Defaults to config.toml directly "
            "inside --project-dir."
        ),
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        type=valid_project_dir_path,
        help=(
            "Path to the project's folder. Decides the output directory, "
            "<project-dir>/allometry/, and -- unless --config is given -- "
            "where the config file is looked up: <project-dir>/config.toml."
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
        "--emit-xml",
        action="store_true",
        help=(
            "Also write a combined ForestPropertyData XML alongside the CSVs, "
            "replayable through xml_to_allometry.py independently of the .gpkg."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report what the run would produce, then exit having written "
            "nothing at all: no CSVs, no JSON, no XML, not even the output "
            "folder. Stops before the growth model, which is what makes a "
            "real run slow, so the filter report arrives in a fraction of "
            "the time."
        ),
    )

    args = parser.parse_args()

    # --config defaults to config.toml directly inside --project-dir -- the
    # layout the docs have the user set up beforehand. Applied here, after
    # parsing, rather than as an argparse default: the default path depends
    # on another argument's value, which add_argument can't express.
    if args.config is not None:
        config_path = args.config
    else:
        config_path = args.project_dir / "config.toml"
        if not config_path.exists() or not config_path.is_file():
            parser.error(
                f"No config file found at the default location: {config_path}. "
                "Pass --config to use a different name or location."
            )
        if config_path.suffix.lower() != ".toml":
            parser.error(f"Default config path is not a .toml file: {config_path}")

    config = load_extraction_config(config_path)

    # NaN is not a physically meaningful altitude/DDY value under any
    # circumstances (unlike an out-of-range-but-real number), so it is
    # rejected outright -- --allow-out-of-range-values does not apply.
    nan_names = [
        name
        for name, value in (("altitude", config.altitude), ("ddy", config.ddy))
        if math.isnan(value)
    ]
    if nan_names:
        parser.error(f"{', '.join(nan_names)} must be a real number, not NaN.")

    out_of_range_messages = [
        message
        for message in (
            out_of_range_message(
                "altitude", config.altitude, ALTITUDE_MIN, ALTITUDE_MAX
            ),
            out_of_range_message("ddy", config.ddy, DDY_MIN, DDY_MAX),
        )
        if message is not None
    ]
    if out_of_range_messages:
        if args.allow_out_of_range_values:
            for message in out_of_range_messages:
                print(
                    f"Warning: {message}; proceeding due to --allow-out-of-range-values"
                )
        else:
            parser.error(
                "; ".join(out_of_range_messages)
                + ". Pass --allow-out-of-range-values to override."
            )

    # Refuse to reuse an existing folder rather than silently overwriting
    # (or, previously, deleting) whatever a prior run left there -- the user
    # must pick a different --project-dir instead. This check runs in dry-run
    # mode too: "would this run even start?" is exactly what a dry run is for.
    # Creating the folder is main()'s job, and only on a real run.
    output_dir = output_dir_for_project(args.project_dir)
    if output_dir.exists():
        parser.error(
            f"Output folder already exists: {output_dir}. Refusing to run into "
            "an existing folder. Pass a different --project-dir instead."
        )

    return CLIArguments(
        input_gpkg=args.input_gpkg,
        config=config,
        config_path=config_path,
        project_dir=args.project_dir,
        allow_out_of_range_values=args.allow_out_of_range_values,
        emit_xml=args.emit_xml,
        dry_run=args.dry_run,
    )


# %% Progress-report printing (side-effecting; kept out of the pure layer above)


def print_section(title: str) -> None:
    """Marks one phase of main()'s reading -> filtering -> writing pipeline
    in the console output, so the three phases are visually separated."""
    print()
    print(title)
    print("-" * len(title))


def print_year_distribution(year_distribution: pd.DataFrame) -> None:
    print("Available measured inventory years (type=1):")
    print(year_distribution.to_string(index=False))
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
    emit_xml: bool,
) -> None:
    """The dry run's stand-in for the real run's writing report: the same
    counts and the same file names, with nothing on disk. The folder is named
    as the one that WOULD be created -- parse_CLI_arguments has already
    refused the run if it exists, so this path is known to be free."""
    dominant, subdominant = csv_counts(plans)
    json_path = output_dir / "extra_gpkg_info.json"

    print(f"Destination folder: {output_dir.resolve()} (not created)")
    print()
    print(
        f"Would write: {len(plans):,} stand(s) -- {dominant:,} dominant + "
        f"{subdominant:,} subdominant = {dominant + subdominant:,} CSV(s)"
    )
    print(f"Would write informational JSON: {json_path}")
    if emit_xml:
        print(f"Would write XML: {output_dir / f'{project_dir.name}.xml'}")
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


def print_skips(skips: list[StandSkipped], label: str) -> None:
    if not skips:
        return
    print(f"{label}: {len(skips)}")
    for skip in skips:
        print(f"  {skip.stand_id}: {skip.reason}")
    print()


# %% main


def main() -> None:
    cli_args = parse_CLI_arguments()
    output_dir = output_dir_for_project(cli_args.project_dir)

    print_section("Reading")
    print("Tool initialized with:")
    print(f"    - input_gpkg  = {cli_args.input_gpkg.resolve()}")
    print(f"    - project_dir = {cli_args.project_dir.resolve()}")
    print(f"    - config      = {cli_args.config_path.resolve()}")
    print(f"    - output_dir  = {output_dir.resolve()}")
    print(f"    - target_year = {cli_args.config.target_year}")
    print(f"    - altitude    = {cli_args.config.altitude}")
    print(f"    - ddy         = {cli_args.config.ddy}")
    if cli_args.dry_run:
        print("    - DRY RUN -- nothing will be written")
    print()

    layers = load_gpkg_layers(cli_args.input_gpkg)
    print(f"stand      : {len(layers.stand):>7,} rows")
    print(f"treestand  : {len(layers.treestand):>7,} rows")
    print(f"treestratum: {len(layers.treestratum):>7,} rows")

    print_section("Filtering")

    print(
        "1. Site filter -- forest land, peatland (Korpi/Räme), already drained, "
        f"fertility class in {cli_args.config.fertilityclass_filter}:"
    )
    filtered_stand = filter_stands_by_site_attributes(
        layers.stand, cli_args.config.fertilityclass_filter
    )
    print(f"   -> {len(filtered_stand):,} / {len(layers.stand):,} stands kept")
    print()

    stand_ids = set(
        pd.to_numeric(filtered_stand["standid"], errors="coerce").dropna().astype(int)
    )
    print_year_distribution(compute_year_distribution(layers.treestand, stand_ids))

    print(f"2. Measured-snapshot filter -- exact {cli_args.config.target_year} match:")
    snapshot = select_target_year_snapshot(
        layers.treestand, stand_ids, cli_args.config.target_year
    )
    n_excluded = len(stand_ids) - len(snapshot)
    print(f"   -> {len(snapshot):,} / {len(stand_ids):,} stands kept")
    if n_excluded > 0:
        print(f"      ({n_excluded} stand(s) had no exact-year match and are excluded)")
    print()

    merged = attach_stand_attributes(snapshot, filtered_stand)
    print(
        "3. Development-class filter -- classes "
        f"{cli_args.config.developmentclass_filter}:"
    )
    merged_filtered = filter_by_developmentclass(
        merged, cli_args.config.developmentclass_filter
    )
    print(f"   -> {len(merged_filtered):,} / {len(merged):,} stands kept")
    print()

    print(
        "4. Structural + species-data checks -- geometry, treestandid, diameter/height:"
    )
    candidates, structural_skips = build_stand_candidates(
        merged_filtered, layers.treestratum
    )
    print(f"   -> {len(candidates):,} / {len(merged_filtered):,} stands kept")
    # structural_skips covers both missing structural fields (no usable
    # geometry, no treestandid) and a species with real basal area but no
    # usable diameter/height (DegenerateSpeciesDataError) -- build_stand_candidates.
    print_skips(structural_skips, "   Skipped (invalid or missing data)")
    print()

    print("5. Viability check -- nonzero total basal area:")
    viable_candidates, ba_skips = partition_viable_candidates(candidates)
    print(f"   -> {len(viable_candidates):,} / {len(candidates):,} stands kept")
    print_skips(ba_skips, "   Skipped (zero basal area)")

    valid_stands, build_skips = build_valid_stands(viable_candidates)
    print_skips(build_skips, "   Skipped (could not build stand record)")
    print()

    print(f"Stands ready for allometry: {len(valid_stands):,}")

    # Everything above this point is identical in a dry run: the filters and
    # the skip reports are precisely what a dry run exists to show. What it
    # skips is everything below -- the growth model (~all of a real run's
    # time) and every write, the output folder included.
    if cli_args.dry_run:
        print_section("Writing (dry run -- nothing is written)")
        plans = [plan_stand_outputs(stand, output_dir) for stand in valid_stands]
        print_dry_run_plan(plans, output_dir, cli_args.project_dir, cli_args.emit_xml)
        return

    print_section("Writing")
    print(f"Destination folder: {output_dir.resolve()}")
    # Created here rather than at argument-parsing time, so a run that fails
    # while reading or filtering leaves no empty folder behind to block the
    # next attempt.
    output_dir.mkdir(parents=True)
    print()

    outcomes = [
        process_stand(stand, cli_args.config, output_dir) for stand in valid_stands
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

    json_path = output_dir / "extra_gpkg_info.json"
    dump_valid_stands_json(valid_stands, json_path)
    print(f"Informational JSON written: {json_path}")

    if cli_args.emit_xml:
        xml_path = output_dir / f"{cli_args.project_dir.name}.xml"
        write_stands_xml(valid_stands, xml_path)
        print(f"XML written: {xml_path}")


if __name__ == "__main__":
    main()
