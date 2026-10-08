"""
Reads Metsakeskus forest inventory data (.gpkg) --> filters --> draws a
PROBABILITY SAMPLE of stands stratified by peat type, fertility class and
development class --> generates SUSI allometry CSVs + stand_data.json for the
sampled stands.

Differences from metsakeskus_to_allometry.py
--------------------------------------------
1.  No single shared `target_year`. Each stand uses its OWN latest type=1
    (measured) treestand record, provided that record is from
    `min_measurement_year` (default 1995, where the SUSI simulation window
    starts) or later. A stand whose newest inventory predates that is dropped
    rather than being treated as current.

    Note that taking the latest record is NOT what prevents a stand being
    counted twice -- `drop_duplicates(subset="standid")` does that, in this
    version and in the single-year one alike. The real double-counting risks
    are a duplicated `standid` in the stand layer and re-delineated
    overlapping polygons; both are checked and reported by
    report_frame_integrity() below.

2.  A stratified probability sample instead of "keep every viable stand".
    See sampling_design.py for the design itself. In short: ONE budget
    allocated across all strata in proportion to stand area, strata too thin
    to represent excluded rather than floored up, and Madow PPS-systematic
    selection within each stratum from a frame sorted by basal area, with a
    random start.

    Area-proportional allocation is what makes the sample a picture of the
    region: each stratum's share of the sample equals its share of the area,
    so peat type, fertility class and development class are each correctly
    composed and can be compared directly, and pooled statistics need no
    correcting. A level that comes out too small to compare -- fertility
    class 2 in Pohjois-Pohjanmaa, at 0.7% of the area -- is reported as not
    estimable rather than inflated by reallocation.

3.  The sampling frame is aligned with what the simulation will actually run.
    run_factorial_region_wide.py discards deciduous-dominant stands and
    stands at or past MAX_INITIAL_AGE; applying those here, BEFORE allocation,
    is what stops a 200-stand sample collapsing to a handful of simulated
    stands and losing every representativeness property on the way.

4.  A stand whose soiltype is missing or unrecognized gets the explicit
    "unknown" peat-type stratum rather than being dropped: a gap in the
    sampling frame's bookkeeping should not silently decide which stands are
    simulated.

Everything else -- site/developmentclass filters, candidate building, species
aggregation, growth-and-yield construction, CSV/JSON writing, peat_type
derivation -- is reused unchanged from metsakeskus_to_allometry.py, so a
sampled stand's output is byte-for-byte what the single-year tool would have
produced for that same stand.
"""

# %% Imports
import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from susi.io.project_layout import (
    CONFIG_FILENAME,
    allometry_dir_for_project,
    stand_data_path_for_project,
)
from susi.io.stand_data import (
    SOURCE_CRS,
    StandDataDocument,
    dump_stand_data_document,
)

# Reused unchanged from the single-year tool.
from tools.metsakeskus_to_allometry.metsakeskus_to_allometry import (
    TREESTAND_MEASURED_TYPE,
    ParsedStand,
    StandOutcome,
    StandProcessingConfig,
    StandWritten,
    attach_stand_attributes,
    build_stand_candidates,
    build_stands,
    csv_counts,
    filter_by_developmentclass,
    filter_stands_by_site_attributes,
    load_gpkg_layers,
    partition_viable_candidates,
    plan_stand_outputs,
    print_dry_run_plan,
    process_stand,
    species_stratum,
    stand_layer_in_source_crs,
    with_parsed_measurement_columns,
)
from tools.shared_allometry_tool_utils.cli_paths import (
    finalize_cli_config,
    resolve_config_path,
)
from tools.shared_allometry_tool_utils.dense_young_stand_scaling import (
    DenseYoungStandScalingConfig,
    decide_scaling_factors,
)
from tools.shared_allometry_tool_utils.input_validation import (
    load_toml_config,
    make_existing_file_validator,
    valid_existing_directory,
)
from tools.shared_allometry_tool_utils.print_formatting import (
    StandSkipped,
    print_section,
    print_skips,
)
from tools.shared_allometry_tool_utils.sampling_design import (
    UNKNOWN_PEAT_TYPE,
    SamplingConfigurationError,
    SamplingUnit,
    Selection,
    StratumKey,
    StratumReport,
    format_allocation_table,
    format_marginal_table,
    select_stratified_sample,
)

# %% Config


class StratifiedExtractionConfig(StandProcessingConfig):
    """Like ExtractionConfig, but with a measurement-year FLOOR instead of an
    exact target year, plus the sampling design's own parameters."""

    # Required, no defaults
    altitude: float
    ddy: float

    # Same defaults as ExtractionConfig
    developmentclass_filter: tuple[int, ...] = (1, 2, 3)
    fertilityclass_filter: tuple[int, ...] = (2, 3, 4, 5)

    # Each stand contributes its own latest measured snapshot, but only if
    # that snapshot is from this year or later. 1995 is where the SUSI
    # simulation window starts, so an older inventory cannot describe the
    # stand's state at the start of the run.
    min_measurement_year: int = 1995

    # --- Sampling frame -----------------------------------------------------
    # These mirror run_factorial_region_wide.py's own discovery filters.
    # Applying them here means the budget is spent on stands that will
    # actually be simulated. Keep them in step with the runner.
    dominant_species_filter: tuple[int, ...] = (1, 2)  # pine, spruce
    max_initial_age: int = 50  # runner's MAX_INITIAL_AGE
    min_table_headroom_years: int = 30  # runner's SIM_LENGTH_YEARS

    # --- Sampling design ----------------------------------------------------
    # ONE budget across all strata, allocated in proportion to stand area --
    # not a budget per development class. Proportional allocation is what
    # makes the sample a picture of the region: each stratum's share of the
    # sample equals its share of the area, so peat type, fertility class and
    # development class are each correctly composed and can be compared
    # directly, while pooled statistics need no rescuing.
    sample_budget_total: int = 150

    # A stratum thinner than either threshold is EXCLUDED and reported, not
    # floored up to a minimum count. Flooring a stratum holding 0.005% of the
    # area over-represents it several-hundred-fold and invents a category the
    # region does not have; a stratum too thin to sample proportionally is too
    # thin to report.
    min_frame_stands_per_stratum: int = 200
    min_stratum_area_share: float = 0.005

    # Fixed so a run is reproducible; change it to draw an independent sample.
    random_seed: int = 19950101

    # --- Growth-table projection --------------------------------------------
    # Overrides AllometryGenerationDefaults' end_year = 80 for THIS tool only,
    # leaving xml_to_allometry.py and new_growth_allometry.py on the shared
    # default. n_trees, start_year and step_years are inherited unchanged.
    #
    # end_year sets how far past the stand's measured age the trajectory runs:
    # the Age column reaches initial_age + end_year, and the table's maximum
    # biomass is the ceiling SUSI's biomass-indexed lookups run into. 120
    # raises that ceiling -- the reason SUSI_CLAMP_ALLOMETRY exists -- at the
    # cost of projecting the growth model past the ages it was built for (a
    # stand measured at 45 reaches Age 165). Worth confirming against an
    # 80-year run with the clamp off before the result is relied on.
    end_year: int = 120

    dense_young_stand_scaling: DenseYoungStandScalingConfig = (
        DenseYoungStandScalingConfig()
    )


@dataclass(frozen=True)
class CLIArguments:
    input_gpkg: Path
    config: StratifiedExtractionConfig
    config_path: Path
    project_dir: Path
    allow_out_of_range_values: bool
    dry_run: bool


def load_extraction_config(config_path: Path) -> StratifiedExtractionConfig:
    return load_toml_config(config_path, StratifiedExtractionConfig.model_validate)


# %% Frame integrity (double-counting checks)


def report_frame_integrity(filtered_stand, max_overlap_pairs_examined: int = 5000) -> None:
    """Two ways the same forest can be counted twice, neither of which
    drop_duplicates(subset="standid") can catch.

    1. A duplicated `standid` in the stand layer. attach_stand_attributes
       merges the stand layer onto the snapshot on standid, so a duplicate
       multiplies rows: the stand enters the sampling frame twice, two
       identical CSVs are written, and the stand_data dict silently collapses
       them again at write time -- so the printed count and the JSON disagree.

    2. Overlapping polygons. Metsakeskus re-delineates stands between
       inventory rounds, so one physical forest can appear under two different
       standids covering the same ground. The ids differ, so nothing upstream
       notices.

    Reported, not enforced: whether an overlap is a genuine duplicate or two
    legitimately adjacent stands sharing an edge is a judgement about the
    data, not something this tool should decide.
    """
    duplicate_ids = int(filtered_stand["standid"].duplicated().sum())
    print(f"   duplicate standid rows in stand layer : {duplicate_ids}")
    if duplicate_ids:
        print(
            "      WARNING: duplicates inflate the sampling frame and are written twice. "
            "De-duplicate the stand layer before trusting the counts below."
        )

    try:
        geometries = filtered_stand.set_geometry("geometry")
        pairs = geometries.sjoin(geometries, predicate="overlaps", how="inner")
        pairs = pairs[pairs.index != pairs["index_right"]]
        n_overlaps = len(pairs) // 2
        print(f"   overlapping stand polygon pairs       : {n_overlaps:,}")
        if n_overlaps:
            # A raw pair count cannot distinguish a digitising sliver along a
            # shared edge -- harmless -- from one forest appearing twice under
            # two ids, which is real double counting. The overlap AREA as a
            # fraction of the smaller polygon separates them: slivers are a
            # fraction of a percent, a re-delineated duplicate approaches 1.
            sample = pairs.head(max_overlap_pairs_examined)
            left = geometries.geometry.loc[sample.index].reset_index(drop=True)
            right = geometries.geometry.loc[sample["index_right"].to_numpy()].reset_index(
                drop=True
            )
            smaller = pd.concat([left.area, right.area], axis=1).min(axis=1)
            fraction = (left.intersection(right).area / smaller).dropna()
            substantial = float((fraction > 0.5).mean())
            print(
                f"      overlap as a fraction of the smaller polygon "
                f"(n={len(fraction):,} pairs examined): "
                f"median {fraction.median():.4f}, p90 {fraction.quantile(0.9):.4f}, "
                f"max {fraction.max():.4f}"
            )
            if substantial > 0.01:
                print(
                    f"      WARNING: {substantial:.1%} of examined pairs overlap by more "
                    "than half the smaller polygon -- that is the same ground entering "
                    "the sampling frame twice, which drop_duplicates(standid) cannot see."
                )
            else:
                print(
                    "      These are slivers along shared edges, not duplicated stands: "
                    "no action needed."
                )
    except Exception as error:  # noqa: BLE001 -- a diagnostic must never abort the run
        print(f"   overlap check skipped ({error})")


# %% Latest-measurement selection


def select_latest_measured_snapshot(
    treestand: pd.DataFrame, stand_ids: set[int], min_year: int
) -> pd.DataFrame:
    """One treestand row per stand: its latest measured (type=1) record whose
    year is >= min_year. Stands whose newest measurement predates min_year are
    dropped -- they cannot describe the stand at the simulation's start."""
    candidates = with_parsed_measurement_columns(treestand, stand_ids)
    measured = candidates[candidates["type_num"] == TREESTAND_MEASURED_TYPE]
    recent = measured[measured["year"] >= min_year]
    return (
        recent.sort_values(["standid", "date_dt"], ascending=[True, False])
        .drop_duplicates(subset="standid", keep="first")
        .reset_index(drop=True)
    )


def print_year_distribution(snapshot: pd.DataFrame) -> None:
    """The realized spread of inventory dates. Every stand in the run is
    simulated from its measured state over one common 1995-2025 weather
    series, so this table IS the heterogeneity that assumption carries, and
    it belongs in the methods section."""
    counts = (
        snapshot.groupby("year")["standid"]
        .count()
        .reset_index()
        .rename(columns={"standid": "n_stands"})
        .sort_values("year")
    )
    print("Selected snapshot years (each stand's own latest measurement):")
    print(counts.to_string(index=False))
    print()


# %% Sampling frame


def dominant_layer_initial_age(stand: ParsedStand) -> float:
    """The age the dominant layer's allometry table will start at.

    Each layer is grown with the other species zeroed, and
    Growth_and_Yield_Table sets its starting age to the basal-area-weighted
    mean over the LIVE species slots -- with one slot alive, that is just that
    slot's own age. So this predicts the table's first Age row without paying
    for the growth model, which is what makes an age-filtered sampling frame
    affordable.

    Deliberately NOT ParsedStand.stand_meanage: that is weighted across all
    three species and is informational metadata only.
    """
    return species_stratum(stand.strata, stand.dominant_species).age


def apply_simulation_frame_filters(
    stands: list[ParsedStand], config: StratifiedExtractionConfig
) -> tuple[list[ParsedStand], list[StandSkipped], dict[str, int]]:
    """Removes stands the simulation runner would discard anyway.

    Without this the budget is allocated over stands that are then deleted by
    run_factorial_region_wide.py, and the realized simulated set has no stated
    relationship to the design -- which is how a 200-stand sample became 7
    simulated stands.
    """
    kept: list[ParsedStand] = []
    skipped: list[StandSkipped] = []
    counts = {"species": 0, "age": 0}

    for stand in stands:
        if stand.dominant_species not in config.dominant_species_filter:
            counts["species"] += 1
            skipped.append(
                StandSkipped(
                    stand_id=stand.id,
                    reason=(
                        f"dominant species {stand.dominant_species} not in "
                        f"{config.dominant_species_filter} (simulation frame)"
                    ),
                )
            )
            continue
        age = dominant_layer_initial_age(stand)
        if age >= config.max_initial_age:
            counts["age"] += 1
            skipped.append(
                StandSkipped(
                    stand_id=stand.id,
                    reason=f"initial age {age} >= {config.max_initial_age} (simulation frame)",
                )
            )
            continue
        kept.append(stand)

    return kept, skipped, counts


def check_table_headroom(config: StratifiedExtractionConfig) -> None:
    """The runner also demands that a stand's allometry table extend at least
    SIM_LENGTH_YEARS past its initial age. Every table this tool writes spans
    exactly `end_year` years beyond the stand's measured age, so the condition
    is a property of the config, identical for every stand -- checked once
    here rather than per stand."""
    if config.end_year < config.min_table_headroom_years:
        raise SamplingConfigurationError(
            f"end_year={config.end_year} gives every allometry table only "
            f"{config.end_year} years of headroom, but the simulation needs "
            f"{config.min_table_headroom_years}. Every stand would be discarded "
            "by the runner. Raise end_year."
        )


def stratum_of(stand: ParsedStand) -> StratumKey:
    """(peat_type, fertilityclass, developmentclass).

    peat_type_from_soiltype consults fertility class only for soiltype 60
    (generic peat with no sub-type recorded); codes 61-67 decide peat type on
    their own. So the two axes are very nearly independent and the cross has
    no structurally empty cells -- stratifying on both is sound.
    """
    peat = stand.peat_type.value if stand.peat_type is not None else UNKNOWN_PEAT_TYPE
    return (peat, stand.fertilityclass, stand.developmentclass)


def to_sampling_units(stands: list[ParsedStand]) -> list[SamplingUnit]:
    return [
        SamplingUnit(
            unit_id=str(stand.id),
            area=float(stand.area) if stand.area is not None else 0.0,
            basal_area=stand.stand_basalarea,
            stratum=stratum_of(stand),
        )
        for stand in stands
    ]


def print_stage_counts(stands: list[ParsedStand], label: str) -> None:
    """Per-development-class counts at one pipeline stage. Printed at every
    stage so that a development class which ends up empty can be traced to the
    filter that emptied it -- rather than being discovered later as a silent
    absence. DC1 (open/seedling) commonly disappears at the viability check,
    because seedling stands often carry zero basal area across all species."""
    counts: dict[int, int] = {}
    for stand in stands:
        counts[stand.developmentclass] = counts.get(stand.developmentclass, 0) + 1
    rendered = "  ".join(f"DC{dc}={counts.get(dc, 0)}" for dc in sorted(counts)) or "none"
    print(f"   {label}: {len(stands):,} stands  ({rendered})")


# %% Sidecar: the design, written next to the data


def write_sampling_design(
    path: Path,
    config: StratifiedExtractionConfig,
    reports: dict[StratumKey, StratumReport],
    selections: list[Selection],
    measurement_year_per_stand: dict[str, int],
) -> None:
    """The design quantities, in their own file.

    StandData has a fixed schema, so these go beside it rather than into it.

    `analysis_weight` (hectares represented = area / pi) is the field the
    analysis uses. Under area-proportional allocation the design is
    self-weighting, so weighted and unweighted statistics agree and the
    weights are a check rather than a correction -- but they are still
    recorded, because proportionality breaks wherever a stratum's allocation
    was clamped by how many stands exist, and for certainty stands. Carrying
    them costs nothing and makes any such departure measurable.

    `excluded` strata are listed too. They are outside the population this
    sample describes, by deliberate choice, and a reader needs to see which
    and how much.

    measurement_year is here as the covariate for the one sensitivity check
    this design needs: stands carry different inventory dates but share one
    1995-2025 weather series, so any pooled ABSOLUTE quantity should be
    checked against it. Within-stand scenario contrasts are unaffected --
    inventory year is constant across a stand's 192 scenarios.
    """
    document = {
        "design": "stratified (peat_type x fertilityclass x developmentclass); "
        "one budget allocated across all strata in proportion to stand area; "
        "strata below the frame-count or area-share threshold excluded, not floored; "
        "Madow PPS-systematic selection within stratum, frame sorted by basal area",
        "random_seed": config.random_seed,
        "min_measurement_year": config.min_measurement_year,
        "sample_budget_total": config.sample_budget_total,
        "min_frame_stands_per_stratum": config.min_frame_stands_per_stratum,
        "min_stratum_area_share": config.min_stratum_area_share,
        "simulation_frame": {
            "dominant_species_filter": list(config.dominant_species_filter),
            "max_initial_age": config.max_initial_age,
            "min_table_headroom_years": config.min_table_headroom_years,
        },
        "strata": [
            {
                "peat_type": key[0],
                "fertilityclass": key[1],
                "developmentclass": key[2],
                "n_available": report.n_available,
                "area_total_ha": report.area_total,
                "area_share_of_retained": report.area_share,
                "n_allocated": report.n_allocated,
                "n_selected": report.n_selected,
                "excluded": report.excluded,
                "exclusion_reason": report.exclusion_reason,
            }
            for key, report in sorted(reports.items())
        ],
        "stands": [
            {
                "stand_id": selection.unit_id,
                "peat_type": selection.stratum[0],
                "fertilityclass": selection.stratum[1],
                "developmentclass": selection.stratum[2],
                "stand_area_ha": selection.area,
                "inclusion_probability": selection.inclusion_probability,
                "design_weight": selection.design_weight,
                # The one field the analysis multiplies by.
                "analysis_weight": selection.analysis_weight,
                "certainty_unit": selection.certainty,
                "measurement_year": measurement_year_per_stand.get(selection.unit_id),
            }
            for selection in sorted(selections, key=lambda s: s.unit_id)
        ],
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


# %% CLI


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Metsakeskus .gpkg stand data to SUSI allometry CSV inputs, "
            "using each stand's own latest measurement (from min_measurement_year "
            "onwards) and a stratified area-proportional probability sample."
        )
    )
    parser.add_argument(
        "input_gpkg",
        type=make_existing_file_validator(".gpkg"),
        help="Path to the Metsakeskus .gpkg file",
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
        help="Path to the project's folder (its inputs/ and outputs/ parent).",
    )
    parser.add_argument(
        "--allow-out-of-range-values",
        action="store_true",
        help="Allow altitude/ddy outside their enforced range instead of blocking.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report the filters, the allocation table and the sample, then exit "
            "having written nothing. Stops before the growth model, so it returns "
            "in a fraction of a real run's time."
        ),
    )

    args = parser.parse_args()
    config_path = resolve_config_path(
        args.config, args.project_dir, parser, CONFIG_FILENAME
    )
    config = load_extraction_config(config_path)
    check_table_headroom(config)
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


# %% main


def main() -> None:
    cli_args = parse_CLI_arguments()
    config = cli_args.config
    output_dir = allometry_dir_for_project(cli_args.project_dir)

    print_section("Reading")
    print("Tool initialized with:")
    print(f"    - input_gpkg           = {cli_args.input_gpkg.resolve()}")
    print(f"    - project_dir          = {cli_args.project_dir.resolve()}")
    print(f"    - config               = {cli_args.config_path.resolve()}")
    print(f"    - output_dir           = {output_dir.resolve()}")
    print(f"    - altitude             = {config.altitude}")
    print(f"    - ddy                  = {config.ddy}")
    print(f"    - min_measurement_year = {config.min_measurement_year}")
    print(f"    - sample_budget_total  = {config.sample_budget_total} (area-proportional across all strata)")
    print(f"    - min frame per stratum= {config.min_frame_stands_per_stratum}")
    print(f"    - min area share       = {config.min_stratum_area_share:.2%}")
    print(f"    - random_seed          = {config.random_seed}")
    print(
        f"    - simulation frame     = species {config.dominant_species_filter}, "
        f"initial age < {config.max_initial_age}, headroom >= "
        f"{config.min_table_headroom_years} yr"
    )
    if cli_args.dry_run:
        print("    - DRY RUN -- nothing will be written")
    print()

    layers = load_gpkg_layers(cli_args.input_gpkg)
    stand_in_source_crs = stand_layer_in_source_crs(layers.stand)
    if stand_in_source_crs is not layers.stand:
        print(f"stand layer reprojected: {layers.stand.crs} -> {SOURCE_CRS}")
    layers = dataclasses.replace(layers, stand=stand_in_source_crs)
    print(f"stand      : {len(layers.stand):>7,} rows")
    print(f"treestand  : {len(layers.treestand):>7,} rows")
    print(f"treestratum: {len(layers.treestratum):>7,} rows")

    print_section("Filtering")

    print(
        "1. Site filter -- forest land, peatland (Korpi/Rame), already drained, "
        f"fertility class in {config.fertilityclass_filter}:"
    )
    filtered_stand = filter_stands_by_site_attributes(
        layers.stand, config.fertilityclass_filter
    )
    print(f"   -> {len(filtered_stand):,} / {len(layers.stand):,} stands kept")
    report_frame_integrity(filtered_stand)
    print()

    stand_ids = set(
        pd.to_numeric(filtered_stand["standid"], errors="coerce").dropna().astype(int)
    )

    print(
        "2. Latest-measurement selection -- each stand's own latest type=1 record, "
        f"from {config.min_measurement_year} onwards:"
    )
    snapshot = select_latest_measured_snapshot(
        layers.treestand, stand_ids, config.min_measurement_year
    )
    n_excluded = len(stand_ids) - len(snapshot)
    print(f"   -> {len(snapshot):,} / {len(stand_ids):,} stands kept")
    if n_excluded > 0:
        print(
            f"      ({n_excluded} stand(s) had no type=1 record from "
            f"{config.min_measurement_year} onwards and are excluded)"
        )
    print()
    print_year_distribution(snapshot)
    measurement_year_per_stand = {
        str(int(stand_id)): int(year)
        for stand_id, year in zip(snapshot["standid"], snapshot["year"], strict=True)
        if pd.notna(year)
    }

    merged = attach_stand_attributes(snapshot, filtered_stand)
    print(f"3. Development-class filter -- classes {config.developmentclass_filter}:")
    merged_filtered = filter_by_developmentclass(merged, config.developmentclass_filter)
    print(f"   -> {len(merged_filtered):,} / {len(merged):,} stands kept")
    print()

    print("4. Structural + species-data checks -- geometry, treestandid, diameter/height:")
    candidates, structural_skips = build_stand_candidates(
        merged_filtered, layers.treestratum
    )
    print(f"   -> {len(candidates):,} / {len(merged_filtered):,} stands kept")
    print_skips(structural_skips, "   Skipped (invalid or missing data)")
    print()

    print("5. Viability check -- nonzero total basal area:")
    viable_candidates, ba_skips = partition_viable_candidates(candidates)
    print(f"   -> {len(viable_candidates):,} / {len(candidates):,} stands kept")
    print_skips(ba_skips, "   Skipped (zero basal area)")
    parsed_stands, build_skips = build_stands(viable_candidates)
    print_skips(build_skips, "   Skipped (could not build stand record)")
    print_stage_counts(parsed_stands, "after viability")
    print()

    print(
        "6. Simulation frame -- the filters run_factorial_region_wide.py applies, "
        "moved upstream so the budget is spent on stands that will actually run:"
    )
    frame_stands, _frame_skips, frame_counts = apply_simulation_frame_filters(
        parsed_stands, config
    )
    print(
        f"   -> {len(frame_stands):,} / {len(parsed_stands):,} stands kept "
        f"({frame_counts['species']} deciduous-dominant, "
        f"{frame_counts['age']} age >= {config.max_initial_age} removed)"
    )
    print_stage_counts(frame_stands, "sampling frame")
    print()

    print_section("Stratified sampling")
    units = to_sampling_units(frame_stands)
    selections, reports = select_stratified_sample(
        units,
        config.sample_budget_total,
        config.min_frame_stands_per_stratum,
        config.min_stratum_area_share,
        config.random_seed,
    )
    print(format_allocation_table(reports))
    print()
    # The two columns above should agree stratum by stratum; this table is the
    # same agreement for each factor on its own, which is what the comparisons
    # between peat types, fertility classes and development classes rest on.
    print(format_marginal_table(reports))
    print()

    selected_ids = {selection.unit_id for selection in selections}
    selected_stands = [s for s in frame_stands if str(s.id) in selected_ids]
    print(
        f"Sampled for allometry generation: {len(selected_stands):,} / "
        f"{len(frame_stands):,} stands in frame "
        f"({sum(s.certainty for s in selections)} certainty units)"
    )
    print()

    if cli_args.dry_run:
        print_section("Writing (dry run -- nothing is written)")
        plans = [plan_stand_outputs(stand, output_dir) for stand in selected_stands]
        print_dry_run_plan(plans, output_dir, cli_args.project_dir)
        return

    print_section("Writing")
    print(f"Destination folder: {output_dir.resolve()}")
    output_dir.mkdir(parents=True)
    print()

    strata_per_stand = {stand.id: stand.strata for stand in selected_stands}
    scaling_factors = decide_scaling_factors(
        strata_per_stand, config.dense_young_stand_scaling
    )

    outcomes: list[StandOutcome] = [
        process_stand(
            stand, config, output_dir, scaling_factor=scaling_factors[stand.id]
        )
        for stand in selected_stands
    ]
    written: list[StandWritten] = [o for o in outcomes if isinstance(o, StandWritten)]
    processing_skips: list[StandSkipped] = [
        o for o in outcomes if isinstance(o, StandSkipped)
    ]

    dominant, subdominant = csv_counts(written)
    print(
        f"Allometry files written: {len(written):,} stand(s) -- {dominant:,} "
        f"dominant + {subdominant:,} subdominant = {dominant + subdominant:,} CSV(s)"
    )
    print_skips(processing_skips, "Skipped (processing failure)")

    final_stands = {outcome.stand_id: outcome.stand_data for outcome in written}
    json_path = stand_data_path_for_project(cli_args.project_dir)
    dump_stand_data_document(
        output_path=json_path,
        document=StandDataDocument(
            crs=SOURCE_CRS,
            altitude=config.altitude,
            ddy=config.ddy,
            stands=final_stands,
        ),
    )
    print(f"Informational JSON written: {json_path}")

    # The design travels with the data: a stand written without its inclusion
    # probability cannot be re-weighted afterwards, and the run cannot be
    # reproduced without the seed.
    written_ids = {str(stand_id) for stand_id in final_stands}
    design_path = json_path.parent / "sampling_design.json"
    write_sampling_design(
        design_path,
        config,
        reports,
        [s for s in selections if s.unit_id in written_ids],
        measurement_year_per_stand,
    )
    print(f"Sampling design written  : {design_path}")


if __name__ == "__main__":
    main()
