"""
Reads Metsäkeskus forest inventory data (.gpkg) --> generates SUSI allometry
CSV inputs (the CanopyLayerAllometry contract, see susi.io.susi_parameter_model)
for every drained-peatland forest stand that survives filtering.

Generalizes src/scripts/metsakeskus.py into a tool, following the conventions
of src/tools/xml_to_allometry.py. Two decisions here deliberately diverge from
both source scripts -- see docs/adr/0001-*.md and docs/adr/0002-*.md.

Every stand that survives filtering is processed (no representative-sampling
subset, unlike metsakeskus.py).
"""

# %% Imports
from __future__ import annotations

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
from tools.xml_to_allometry import out_of_range_message

# %% Constants -- hard-coded, non-negotiable (see module docstring / ADRs).
#
# These are not analyst choices: they define what "a stand SUSI can simulate"
# means at all. Compare with ExtractionConfig below, whose fields ARE analyst
# choices (which fertility/development classes to include this run, how many
# reference trees, ...).

# SUSI only ever simulates forest land on drained peatland.
MAINGROUP_FOREST_LAND = 1  # Metsäkeskus maingroup code: forest land (excludes agricultural/other land)
SUBGROUP_PEATLAND = (2, 3)  # Korpi (spruce mire), Räme (pine mire)
# ojikko / muuttuma / turvekangas -- already-ditched drainage-succession
# stages. Excludes code 6 (ojittamaton suo, undrained/pristine mire): SUSI
# simulates drained-peatland hydrology, which doesn't apply to a mire that's
# never been ditched. See src/scripts/metsakeskus.py's DRAINAGESTATE_FILTER
# comment for the counts this was verified against.
DRAINAGESTATE_DRAINED = (7, 8, 9)

# Metsäkeskus tree-species codes -> SUSI's fixed three growth-model slots.
PINE_SPECIES_CODES = frozenset({1})
SPRUCE_SPECIES_CODES = frozenset({2})
DECIDUOUS_SPECIES_CODES = frozenset({3, 4, 5, 6, 7, 8, 9, 15, 20, 29})

# Growth_and_Yield_Table.peat: every stand this tool processes is peatland by
# construction (SUBGROUP_PEATLAND above already restricts to Korpi/Räme).
PEAT = 1

# Nominal sapling-scale values substituted when a species group has zero
# measured basal area (genuinely absent from the stand) or degenerate
# diameter/height data. Growth_and_Yield_Table divides by diameter in several
# places; these prevent a zero/NaN input from breaking that math. Matches
# metsakeskus.py's own dummy-stratum values.
NOMINAL_DIAMETER_CM = 5.0
NOMINAL_HEIGHT_M = 3.0
# Age shown for a species with zero basal area. Cosmetic only: G=0 means this
# species slot never enters Growth_and_Yield_Table's actual growth math.
NOMINAL_AGE_YEARS = 30

# xml_to_allometry.py's own enforced-range bounds for altitude/ddy, reused
# here for the same reason (catch a mistyped value, e.g. metres vs feet).
ALTITUDE_MIN = 0.0
ALTITUDE_MAX = 1000.0
DDY_MIN = 500.0
DDY_MAX = 2000.0

# treestand.type: the REAL measured/interpreted inventory date. Types 2 and 3
# are Metsäkeskus's own grown-forward projections (to a common "current" date,
# then +10y) -- SUSI does its own growth simulation from the measured
# starting point, so using Metsäkeskus's projections as the seed would double
# up on growth modeling. See src/scripts/metsakeskus.py for the full
# empirical verification of this on MV_Uusimaa.gpkg.
TREESTAND_MEASURED_TYPE = 1

SOURCE_CRS = "EPSG:3067"  # ETRS-TM35FIN, the CRS Metsäkeskus geometries ship in
YKJ_CRS = "EPSG:2393"  # Finnish YKJ grid, what Growth_and_Yield_Table's x/y expect

MISSING_SOILTYPE_FALLBACK = 62  # metsakeskus.py's own fallback for an unrecorded soiltype

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
    """One species' aggregated growth-model inputs for one stand. Same shape
    as xml_to_allometry.py's TreeStratum (a frozen dataclass here, not
    pydantic -- this module's domain structs are plain data, not validated
    input)."""

    age: int
    basal_area: float
    stem_count: int
    mean_diameter: float
    mean_height: float


@dataclass(frozen=True)
class StandSiteAttributes:
    """Site-level facts a stand carries into the output (informational JSON
    dump, optional XML). soiltype is always concrete: MISSING_SOILTYPE_FALLBACK
    is applied once, at parse time (build_stand_candidates), not carried as
    an Optional for every downstream reader to re-handle."""

    subgroup: int
    fertilityclass: int
    developmentclass: int
    drainagestate: int
    soiltype: int


@dataclass(frozen=True)
class StandCandidate:
    """One stand after merge + species aggregation, before the viability
    check (partition_viable_candidates) that decides whether it's actually
    processed."""

    id: StandID
    site: StandSiteAttributes
    strata: PerSpecies[TreeStratum]
    geometry: BaseGeometry


@dataclass(frozen=True)
class FilteredStand:
    """One stand ready for Growth_and_Yield_Table. Always fully valid --
    build_filtered_stand only ever runs on the viable half of
    partition_viable_candidates' output, so there is no FilteredStand-or-None
    branch anywhere downstream."""

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
    """The three gpkg layers this tool reads. treestandsummary is
    deliberately absent: verified to have zero coverage of type=1 (measured)
    treestand rows, which is all this tool ever keeps (see
    select_target_year_snapshot) -- see src/scripts/metsakeskus.py for the
    full verification."""

    stand: gpd.GeoDataFrame
    treestand: pd.DataFrame
    treestratum: pd.DataFrame


@dataclass(frozen=True)
class ExtractionConfig:
    """Defaulted/required parameters, loaded from a TOML file (see
    parse_extraction_config). Hard-coded, non-negotiable parameters live as
    module constants above instead."""

    # Required -- no default. Python's dataclass field ordering (required
    # fields before defaulted ones) makes these mandatory at construction;
    # parse_extraction_config additionally checks for them explicitly so a
    # missing one is reported by name instead of a generic TypeError.
    target_year: int
    altitude: float
    ddy: float
    # Defaulted -- tuples, not lists: a frozen dataclass with a mutable list
    # default is a footgun (and these should never be mutated in place).
    developmentclass_filter: tuple[int, ...] = (1, 2, 3)
    fertilityclass_filter: tuple[int, ...] = (2, 3, 4, 5)
    n_trees: int = 20
    start_year: int = 5
    end_year: int = 80
    step_years: int = 5


@dataclass(frozen=True)
class CLIArguments:
    input_gpkg: Path
    output_dir: Path  # always concrete -- inputs/<project_name>/allometry/ default resolved before construction
    config: ExtractionConfig
    project_name: str
    allow_out_of_range_values: bool
    emit_xml: bool


@dataclass(frozen=True)
class StandWritten:
    stand_id: StandID
    dominant_csv: Path
    # None for a monoculture stand: when the runner-up species carries zero
    # basal area, Growth_and_Yield_Table has no live trees left to build a
    # table from (every species slot would have N=0) -- there is no second
    # canopy layer to write, not a failure. This is a deliberate exception to
    # this module's usual no-Optional convention: it mirrors
    # CanopyLayerAllometry.pointers' own `| None` for "this layer is unused",
    # not an incidental gap pushed downstream. See docs/adr/0002.
    subdominant_csv: Path | None


@dataclass(frozen=True)
class StandSkipped:
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
        raise ValueError(f"Config file is missing required field(s): {', '.join(missing)}")

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


def _with_parsed_measurement_columns(treestand: pd.DataFrame, stand_ids: set[int]) -> pd.DataFrame:
    """Shared prep for select_target_year_snapshot and compute_year_distribution:
    restrict to the given stands' rows and parse date/type/year."""
    candidates = treestand[treestand["standid"].isin(stand_ids)].copy()
    candidates["date_dt"] = pd.to_datetime(candidates["date"], errors="coerce")
    candidates["type_num"] = pd.to_numeric(candidates["type"], errors="coerce")
    candidates["year"] = candidates["date_dt"].dt.year
    return candidates


def compute_year_distribution(treestand: pd.DataFrame, stand_ids: set[int]) -> pd.DataFrame:
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
    return treestand_snapshot.merge(filtered_stand[present_columns], on="standid", how="left")


def filter_by_developmentclass(
    merged: pd.DataFrame,
    developmentclass_filter: tuple[int, ...],
) -> pd.DataFrame:
    result = merged.copy()
    result["developmentclass"] = pd.to_numeric(result["developmentclass"], errors="coerce")
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


def aggregate_species_group(rows: pd.DataFrame, dominant_age: int) -> TreeStratum:
    """Collapses every treestratum row belonging to one species group (there
    can be several, e.g. different diameter cohorts) into one basal-area-
    weighted TreeStratum. A species entirely absent from this stand (rows
    empty) gets the nominal zero-basal-area stratum -- a real domain value,
    not a sentinel, so callers never handle absence separately from presence."""
    if rows.empty:
        return TreeStratum(
            age=dominant_age,
            basal_area=0.0,
            stem_count=0,
            mean_diameter=NOMINAL_DIAMETER_CM,
            mean_height=NOMINAL_HEIGHT_M,
        )

    total_basal_area = float(rows["basalarea"].sum(skipna=True))
    total_stem_count = float(rows["stemcount"].sum(skipna=True))

    if total_basal_area > 0:
        # Basal-area-weighted means: strata with more BA (bigger/more trees)
        # dominate the aggregate age/diameter/height for the species group.
        weights = rows["basalarea"].fillna(0.0)
        weighted_age = (rows["age"].fillna(0.0) * weights).sum() / total_basal_area
        weighted_diameter = (rows["meandiameter"].fillna(0.0) * weights).sum() / total_basal_area
        weighted_height = (rows["meanheight"].fillna(0.0) * weights).sum() / total_basal_area
    else:
        # No basal area to weight by -- fall back to plain averages.
        weighted_age = rows["age"].mean()
        weighted_diameter = rows["meandiameter"].mean()
        weighted_height = rows["meanheight"].mean()

    # Guard against zero/NaN diameter or height, which would break
    # downstream allometry math.
    diameter = (
        float(weighted_diameter)
        if pd.notna(weighted_diameter) and weighted_diameter > 0
        else NOMINAL_DIAMETER_CM
    )
    height = (
        float(weighted_height)
        if pd.notna(weighted_height) and weighted_height > 0
        else NOMINAL_HEIGHT_M
    )

    if total_stem_count <= 0 and total_basal_area > 0:
        total_stem_count = estimate_stemcount(total_basal_area, diameter)

    return TreeStratum(
        age=max(1, round(weighted_age)) if pd.notna(weighted_age) else NOMINAL_AGE_YEARS,
        basal_area=total_basal_area,
        stem_count=round(total_stem_count) if total_stem_count > 0 else 0,
        mean_diameter=diameter,
        mean_height=height,
    )


def build_species_strata(stand_strata: pd.DataFrame, dominant_age: int) -> PerSpecies[TreeStratum]:
    """Splits one stand's treestratum rows into the three SUSI species slots."""
    return PerSpecies(
        pine=aggregate_species_group(
            stand_strata[stand_strata["treespecies"].isin(PINE_SPECIES_CODES)], dominant_age
        ),
        spruce=aggregate_species_group(
            stand_strata[stand_strata["treespecies"].isin(SPRUCE_SPECIES_CODES)], dominant_age
        ),
        deciduous=aggregate_species_group(
            stand_strata[stand_strata["treespecies"].isin(DECIDUOUS_SPECIES_CODES)], dominant_age
        ),
    )


def total_basal_area(strata: PerSpecies[TreeStratum]) -> float:
    return strata.pine.basal_area + strata.spruce.basal_area + strata.deciduous.basal_area


# %% Candidate assembly and viability


def build_stand_candidates(
    merged_filtered: pd.DataFrame,
    treestratum: pd.DataFrame,
) -> tuple[list[StandCandidate], list[StandSkipped]]:
    """One StandCandidate per row of merged_filtered. A row with no usable
    geometry is excluded here, as a StandSkipped -- not carried downstream as
    a None geometry."""
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

        treestandid_value = pd.to_numeric(pd.Series([row.get("treestandid")]), errors="coerce").iloc[0]
        if pd.isna(treestandid_value):
            skipped.append(StandSkipped(stand_id=stand_id, reason="no treestandid"))
            continue

        stand_strata = treestratum[treestratum_treestandid == int(treestandid_value)]
        strata = build_species_strata(stand_strata, dominant_age=NOMINAL_AGE_YEARS)

        site = StandSiteAttributes(
            subgroup=int(row["subgroup"]),
            fertilityclass=int(row["fertilityclass"]),
            developmentclass=int(row["developmentclass"]),
            drainagestate=(
                int(row["drainagestate"]) if pd.notna(row.get("drainagestate")) else DRAINAGESTATE_DRAINED[0]
            ),
            soiltype=(
                int(row["soiltype"]) if pd.notna(row.get("soiltype")) else MISSING_SOILTYPE_FALLBACK
            ),
        )

        candidates.append(StandCandidate(id=stand_id, site=site, strata=strata, geometry=geometry))

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
                StandSkipped(stand_id=candidate.id, reason="zero basal area across all species")
            )
        else:
            viable.append(candidate)
    return viable, skipped


# %% Building the final stand record


def determine_dominant_and_subdominant_species(strata: PerSpecies[TreeStratum]) -> tuple[int, int]:
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
    ranked = sorted(basal_areas, key=lambda species_code: basal_areas[species_code], reverse=True)
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


def build_filtered_stand(candidate: StandCandidate) -> FilteredStand:
    """Builds the final stand record from an already-viable candidate
    (nonzero total basal area, guaranteed by partition_viable_candidates;
    non-empty geometry, guaranteed by build_stand_candidates). Not fully
    total, though: centroid_to_ykj can still raise for a geometry that's
    present and non-empty but otherwise degenerate (e.g. all-coincident
    points) -- callers processing a batch should use build_filtered_stands,
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

    dominant_species, subdominant_species = determine_dominant_and_subdominant_species(strata)
    x_ykj, y_ykj = centroid_to_ykj(candidate.geometry)

    return FilteredStand(
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


def build_filtered_stands(
    candidates: list[StandCandidate],
) -> tuple[list[FilteredStand], list[StandSkipped]]:
    """Batch build_filtered_stand, isolating one candidate's failure (see
    build_filtered_stand's docstring) as a StandSkipped instead of letting it
    abort every other stand in the run -- the same per-stand isolation
    principle process_stand already applies to the growth-table stage."""
    built: list[FilteredStand] = []
    skipped: list[StandSkipped] = []
    for candidate in candidates:
        try:
            built.append(build_filtered_stand(candidate))
        except Exception as error:  # noqa: BLE001 -- one bad stand's geometry math must not abort the batch
            skipped.append(StandSkipped(stand_id=candidate.id, reason=str(error)))
    return built, skipped


# %% Growth-and-yield table construction


_ZERO_STRATUM = TreeStratum(
    age=NOMINAL_AGE_YEARS,
    basal_area=0.0,
    stem_count=0,
    mean_diameter=NOMINAL_DIAMETER_CM,
    mean_height=NOMINAL_HEIGHT_M,
)


def species_stratum(strata: PerSpecies[TreeStratum], species_code: int) -> TreeStratum:
    """The one TreeStratum a species code (1=pine, 2=spruce, 3=deciduous) refers to."""
    return {1: strata.pine, 2: strata.spruce, 3: strata.deciduous}[species_code]


def isolate_species_layer(strata: PerSpecies[TreeStratum], active_species: int) -> PerSpecies[TreeStratum]:
    """Zeroes every species slot except active_species -- this is how a
    single canopy layer (dominant or subdominant) is modeled as that one
    species growing alone (see docs/adr/0002)."""
    if active_species == 1:
        return PerSpecies(pine=strata.pine, spruce=_ZERO_STRATUM, deciduous=_ZERO_STRATUM)
    if active_species == 2:
        return PerSpecies(pine=_ZERO_STRATUM, spruce=strata.spruce, deciduous=_ZERO_STRATUM)
    return PerSpecies(pine=_ZERO_STRATUM, spruce=_ZERO_STRATUM, deciduous=strata.deciduous)


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


def clear_previous_outputs(output_dir: Path) -> None:
    """Removes this tool's own per-stand CSVs from a previous run in
    output_dir, before writing new ones. Without this, a stand that used to
    have a subdominant layer but is now a monoculture under the current
    config (StandWritten.subdominant_csv=None) would leave last run's stale
    susi_input_<id>_subdominant.csv on disk, contradicting the current run's
    result -- mirrors src/scripts/metsakeskus.py's own pre-write cleanup."""
    for pattern in ("susi_input_*_dominant.csv", "susi_input_*_subdominant.csv"):
        for stale_file in output_dir.glob(pattern):
            stale_file.unlink()


def write_allometry_csv(table: pd.DataFrame, species_id: int, output_path: Path) -> None:
    """Writes one CanopyLayerAllometry-contract CSV -- readable directly by
    susi.io.susi_parameter_model.read_allometry_info_from_csv."""
    table_with_species = table.copy()
    table_with_species.insert(0, "Species_ID", species_id)
    table_with_species.to_csv(output_path, index=False)


def filtered_stand_to_json_dict(stand: FilteredStand) -> dict:
    data = dataclasses.asdict(stand)
    data["id"] = str(stand.id)
    data["geometry"] = stand.geometry.wkt
    return data


def dump_filtered_stands_json(stands: list[FilteredStand], output_path: Path) -> None:
    """Informational dump of every processed stand's data, including the
    filter/stratum columns -- mirrors xml_to_allometry.py's
    extra_XML_info.json for anyone who wants to regroup or audit later."""
    payload = {"stands": [filtered_stand_to_json_dict(stand) for stand in stands]}
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


def stand_to_xml_block(stand: FilteredStand) -> str:
    """Renders one stand as a <st:Stand> block in the Finnish
    ForestPropertyData schema -- the format xml_to_allometry.py reads."""
    coords_str = " ".join(f"{x},{y}" for x, y in stand.geometry.exterior.coords)
    strata_block = (
        _stratum_xml_block(stand.strata.pine, 1)
        + _stratum_xml_block(stand.strata.spruce, 2)
        + _stratum_xml_block(stand.strata.deciduous, 3)
    )
    return (
        f'    <st:Stand id="{stand.id}">\n'
        "      <st:StandBasicData>\n"
        f"        <st:FertilityClass>{stand.site.fertilityclass}</st:FertilityClass>\n"
        f"        <st:MainGroup>{MAINGROUP_FOREST_LAND}</st:MainGroup>\n"
        f"        <st:SubGroup>{stand.site.subgroup}</st:SubGroup>\n"
        f"        <st:SoilType>{stand.site.soiltype}</st:SoilType>\n"
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


def write_stands_xml(stands: list[FilteredStand], output_path: Path) -> None:
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


def process_stand(stand: FilteredStand, config: ExtractionConfig, output_dir: Path) -> StandOutcome:
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

        subdominant_stratum = species_stratum(stand.strata, stand.subdominant_species)
        subdominant_table: pd.DataFrame | None = None
        if subdominant_stratum.basal_area > 0:
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
        dominant_path = output_dir / f"susi_input_{stand.id}_dominant.csv"
        write_allometry_csv(dominant_table, stand.dominant_species, dominant_path)

        subdominant_path: Path | None = None
        if subdominant_table is not None:
            subdominant_path = output_dir / f"susi_input_{stand.id}_subdominant.csv"
            write_allometry_csv(subdominant_table, stand.subdominant_species, subdominant_path)

        return StandWritten(
            stand_id=stand.id, dominant_csv=dominant_path, subdominant_csv=subdominant_path
        )
    except Exception as error:  # noqa: BLE001 -- deliberately broad: any per-stand failure becomes a skip, not a run-aborting exception
        return StandSkipped(stand_id=stand.id, reason=str(error))


# %% CLI


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
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Output folder. Defaults to inputs/<project-name>/allometry/",
    )
    parser.add_argument(
        "--config", type=valid_config_path, required=True, help="Path to the TOML config file"
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="Names this run. Drives the default output directory, inputs/<project-name>/allometry/",
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

    args = parser.parse_args()
    config = load_extraction_config(args.config)

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
            out_of_range_message("altitude", config.altitude, ALTITUDE_MIN, ALTITUDE_MAX),
            out_of_range_message("ddy", config.ddy, DDY_MIN, DDY_MAX),
        )
        if message is not None
    ]
    if out_of_range_messages:
        if args.allow_out_of_range_values:
            for message in out_of_range_messages:
                print(f"Warning: {message}; proceeding due to --allow-out-of-range-values")
        else:
            parser.error(
                "; ".join(out_of_range_messages)
                + ". Pass --allow-out-of-range-values to override."
            )

    output_dir = args.output_dir or (Path("inputs") / args.project_name / "allometry")
    output_dir.mkdir(parents=True, exist_ok=True)

    return CLIArguments(
        input_gpkg=args.input_gpkg,
        output_dir=output_dir,
        config=config,
        project_name=args.project_name,
        allow_out_of_range_values=args.allow_out_of_range_values,
        emit_xml=args.emit_xml,
    )


# %% Progress-report printing (side-effecting; kept out of the pure layer above)


def print_year_distribution(year_distribution: pd.DataFrame) -> None:
    print("Available measured inventory years (type=1):")
    print(year_distribution.to_string(index=False))
    print()


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

    print("Tool initialized with:")
    print(f"    - input_gpkg  = {cli_args.input_gpkg}")
    print(f"    - output_dir  = {cli_args.output_dir}")
    print(f"    - project     = {cli_args.project_name}")
    print(f"    - target_year = {cli_args.config.target_year}")
    print(f"    - altitude    = {cli_args.config.altitude}")
    print(f"    - ddy         = {cli_args.config.ddy}")
    print()

    layers = load_gpkg_layers(cli_args.input_gpkg)
    print(f"stand      : {len(layers.stand):>7,} rows")
    print(f"treestand  : {len(layers.treestand):>7,} rows")
    print(f"treestratum: {len(layers.treestratum):>7,} rows")
    print()

    filtered_stand = filter_stands_by_site_attributes(
        layers.stand, cli_args.config.fertilityclass_filter
    )
    print(f"Filtered peatland forest stands: {len(filtered_stand):,}")
    print()

    stand_ids = set(
        pd.to_numeric(filtered_stand["standid"], errors="coerce").dropna().astype(int)
    )
    print_year_distribution(compute_year_distribution(layers.treestand, stand_ids))

    snapshot = select_target_year_snapshot(layers.treestand, stand_ids, cli_args.config.target_year)
    n_excluded = len(stand_ids) - len(snapshot)
    print(f"Stands with a {cli_args.config.target_year} measured snapshot: {len(snapshot):,}")
    if n_excluded > 0:
        print(f"  ({n_excluded} stand(s) had no exact-year match and are excluded)")
    print()

    merged = attach_stand_attributes(snapshot, filtered_stand)
    merged_filtered = filter_by_developmentclass(merged, cli_args.config.developmentclass_filter)
    print(
        f"After developmentclass filter {cli_args.config.developmentclass_filter}: "
        f"{len(merged_filtered):,}"
    )
    print()

    candidates, structural_skips = build_stand_candidates(merged_filtered, layers.treestratum)
    viable_candidates, ba_skips = partition_viable_candidates(candidates)
    print(f"Viable stands: {len(viable_candidates):,}")
    print_skips(structural_skips, "Skipped (structural)")
    print_skips(ba_skips, "Skipped (zero basal area)")

    filtered_stands, build_skips = build_filtered_stands(viable_candidates)
    print_skips(build_skips, "Skipped (could not build stand record)")

    clear_previous_outputs(cli_args.output_dir)
    outcomes = [process_stand(stand, cli_args.config, cli_args.output_dir) for stand in filtered_stands]
    written: list[StandWritten] = [o for o in outcomes if isinstance(o, StandWritten)]
    processing_skips: list[StandSkipped] = [o for o in outcomes if isinstance(o, StandSkipped)]

    n_csvs = len(written) + sum(1 for w in written if w.subdominant_csv is not None)
    print(f"Allometry files written: {len(written)} stand(s), {n_csvs} CSV(s)")
    print_skips(processing_skips, "Skipped (processing failure)")

    json_path = cli_args.output_dir / "extra_gpkg_info.json"
    dump_filtered_stands_json(filtered_stands, json_path)
    print(f"Informational JSON written: {json_path}")

    if cli_args.emit_xml:
        xml_path = cli_args.output_dir / f"{cli_args.project_name}.xml"
        write_stands_xml(filtered_stands, xml_path)
        print(f"XML written: {xml_path}")


if __name__ == "__main__":
    main()
