"""
Reads Metsakeskus forest inventory data (.gpkg) --> filters --> picks a
representative, area-weighted stratified SAMPLE of stands (not every stand
that survives filtering, unlike metsakeskus_to_allometry.py) --> generates
SUSI allometry CSVs + stand_data.json for the sampled stands.

Two differences from metsakeskus_to_allometry.py, both requested to bring
this pipeline in line with MK_to_susi_allometry_v2.ipynb:

  1. No single shared `target_year`. Each stand uses its OWN latest
     type=1 (measured) treestand record, whatever year that happens to be
     -- see select_latest_measured_snapshot() below. This mirrors the
     notebook's cell 3 (select_last_treestand_and_first_n_treestratums),
     and avoids silently excluding a stand that simply has no record in one
     shared snapshot year.

  2. Non-uniform, area-weighted stratified sampling instead of "keep every
     viable stand": within each (peat_type x fertilityclass x
     developmentclass) stratum, a target sample count is allocated
     proportional to that stratum's SHARE of total stand area within its
     developmentclass (floored at min_stands_per_stratum), then that many
     stands are picked at evenly-spaced quantiles of the AREA-WEIGHTED
     basal-area distribution -- see allocate_sample_sizes() and
     select_weighted_quantile_representatives() below, ported from the
     notebook's cell 7.

Everything else -- the site/developmentclass filters, per-stand candidate
building, species aggregation, growth-and-yield table construction, CSV/JSON
writing, peat_type derivation -- is reused unchanged from
metsakeskus_to_allometry.py, so a sampled stand's output is byte-for-byte
what the single-year tool would have produced for that same stand.
"""

# %% Imports
import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from susi.io.load_output_data import StandID
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
from tools.shared_allometry_tool_utils.allometry_generation_defaults import (
    AllometryGenerationDefaults,
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
from tools.shared_allometry_tool_utils.cli_paths import (
    finalize_cli_config,
    resolve_config_path,
)

# Reused unchanged from the single-year tool: filters, candidate/stand
# building, species aggregation, growth-table construction, CSV writing,
# per-stand orchestration (process_stand already includes peat_type after
# the patch applied to that file).
from tools.shared_allometry_tool_utils.dense_young_stand_scaling import (
    DenseYoungStandScalingConfig,
    decide_scaling_factors,
)
from tools.metsakeskus_to_allometry.metsakeskus_to_allometry import (
    TREESTAND_MEASURED_TYPE,
    ParsedStand,
    StandOutcome,
    StandWritten,
    ExtractionConfig,
    load_gpkg_layers,
    stand_layer_in_source_crs,
    filter_stands_by_site_attributes,
    with_parsed_measurement_columns,  # renamed from _with_parsed_measurement_columns, see patch (d)
    attach_stand_attributes,
    filter_by_developmentclass,
    build_stand_candidates,
    partition_viable_candidates,
    build_stands,
    process_stand,
    csv_counts,
    print_dry_run_plan,
    plan_stand_outputs,
)


# %% Config


class StratifiedExtractionConfig(AllometryGenerationDefaults):
    """Like ExtractionConfig, but with no target_year (each stand uses its
    own latest measurement) and with the stratified-sampling budget instead."""

    # Required, no defaults
    altitude: float
    ddy: float

    # Optional -- same defaults as ExtractionConfig
    developmentclass_filter: tuple[int, ...] = (1, 2, 3)
    fertilityclass_filter: tuple[int, ...] = (2, 3, 4, 5)

    # Stratified-sampling parameters (see MK_to_susi_allometry_v2.ipynb
    # TOTAL_BUDGET_PER_DC / MIN_PER_STRATUM). Keyed by developmentclass.
    sample_budget_per_developmentclass: dict[int, int] = {1: 30, 2: 90, 3: 90}
    min_stands_per_stratum: int = 3

    # NOTE (2026-10-04): added when merging in create-input-structure's
    # dense young stand scaling work (PR #312). Same field, same default,
    # same rationale as ExtractionConfig's -- process_stand() now requires
    # a scaling_factor argument regardless of which tool calls it.
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


# %% Latest-measurement selection (replaces select_target_year_snapshot)


def select_latest_measured_snapshot(
    treestand: pd.DataFrame, stand_ids: set[int]
) -> pd.DataFrame:
    """One treestand row per stand: the measured (type=1) record with the
    latest date, whatever year that falls in -- not one shared target_year.
    Ported from MK_to_susi_allometry_v2.ipynb cell 3."""
    candidates = with_parsed_measurement_columns(treestand, stand_ids)
    measured = candidates[candidates["type_num"] == TREESTAND_MEASURED_TYPE]
    return (
        measured.sort_values(["standid", "date_dt"], ascending=[True, False])
        .drop_duplicates(subset="standid", keep="first")
        .reset_index(drop=True)
    )


def print_year_distribution(latest_snapshot: pd.DataFrame) -> None:
    counts = (
        latest_snapshot.groupby("year")["standid"]
        .count()
        .reset_index()
        .rename(columns={"standid": "n_stands"})
        .sort_values("year")
    )
    print("Selected snapshot years (each stand's own latest measurement):")
    print(counts.to_string(index=False))
    print()


# %% Stratified sampling (ported from MK_to_susi_allometry_v2.ipynb cell 7)

StratumKey = tuple[str, int, int]  # (peat_type, fertilityclass, developmentclass)


def stratum_key(stand: ParsedStand) -> StratumKey:
    peat_type_value = stand.peat_type.value if stand.peat_type is not None else None
    return (peat_type_value, stand.fertilityclass, stand.developmentclass)


def allocate_sample_sizes(
    stands: list[ParsedStand],
    budget_per_dc: dict[int, int],
    min_per_stratum: int,
) -> dict[StratumKey, int]:
    """{(peat_type, fertilityclass, developmentclass): n_target}, with
    n_target proportional to that stratum's share of total stand AREA within
    its developmentclass, floored at min_per_stratum."""
    allocation: dict[StratumKey, int] = {}
    for dc, budget in budget_per_dc.items():
        subset = [s for s in stands if s.developmentclass == dc]
        area_by_stratum: dict[tuple[str, int], float] = {}
        for stand in subset:
            peat_type_value = stand.peat_type.value if stand.peat_type is not None else None
            key = (peat_type_value, stand.fertilityclass)
            area_by_stratum[key] = area_by_stratum.get(key, 0.0) + (stand.area or 0.0)
        total_area = sum(area_by_stratum.values())
        if total_area <= 0:
            continue
        for (peat_type_value, fc), area in area_by_stratum.items():
            n_target = max(min_per_stratum, round(budget * area / total_area))
            allocation[(peat_type_value, fc, dc)] = int(n_target)
    return allocation


def select_weighted_quantile_representatives(
    stands: list[ParsedStand], n: int
) -> list[ParsedStand]:
    """Up to n stands at evenly-spaced quantiles of the AREA-WEIGHTED
    stand_basalarea distribution, so a handful of large stands don't get the
    same quantile "vote" as many small fragmented ones."""
    ordered = sorted(stands, key=lambda s: s.stand_basalarea)
    actual_n = min(n, len(ordered))
    if actual_n == 0:
        return []
    if actual_n == len(ordered):
        return ordered

    weights = [max(s.area or 0.0, 0.0) for s in ordered]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(ordered)
        total_weight = float(len(ordered))

    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative.append(running / total_weight)

    target_quantiles = [i / (actual_n + 1) for i in range(1, actual_n + 1)]
    indices = sorted(
        {
            next(i for i, cw in enumerate(cumulative) if cw >= q)
            for q in target_quantiles
        }
    )
    if len(indices) < actual_n:
        remaining = [i for i in range(len(ordered)) if i not in indices]
        indices += remaining[: actual_n - len(indices)]
        indices = sorted(set(indices))
    return [ordered[i] for i in indices]


def select_stratified_sample(
    stands: list[ParsedStand],
    budget_per_dc: dict[int, int],
    min_per_stratum: int,
) -> tuple[list[ParsedStand], list[StandSkipped]]:
    """Splits out stands with no peat_type (sampling can't place them into a
    stratum) as StandSkipped, then applies allocate_sample_sizes +
    select_weighted_quantile_representatives per stratum."""
    with_peat_type = [s for s in stands if s.peat_type is not None]
    skipped = [
        StandSkipped(stand_id=s.id, reason="no peat_type (unrecognized soiltype)")
        for s in stands
        if s.peat_type is None
    ]

    allocation = allocate_sample_sizes(with_peat_type, budget_per_dc, min_per_stratum)

    groups: dict[StratumKey, list[ParsedStand]] = {}
    for stand in with_peat_type:
        groups.setdefault(stratum_key(stand), []).append(stand)

    selected: list[ParsedStand] = []
    for key, group in sorted(groups.items()):
        n_target = allocation.get(key, min_per_stratum)
        selected.extend(select_weighted_quantile_representatives(group, n_target))

    return selected, skipped


# %% CLI


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Metsakeskus .gpkg stand data to SUSI allometry CSV inputs, "
            "using each stand's own latest measurement year and a representative, "
            "area-weighted stratified sample instead of every filtered stand."
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
        help="Report what the run would sample and write, then exit having written nothing.",
    )

    args = parser.parse_args()
    config_path = resolve_config_path(
        args.config, args.project_dir, parser, CONFIG_FILENAME
    )
    config = load_extraction_config(config_path)
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
    output_dir = allometry_dir_for_project(cli_args.project_dir)

    print_section("Reading")
    print("Tool initialized with:")
    print(f"    - input_gpkg  = {cli_args.input_gpkg.resolve()}")
    print(f"    - project_dir = {cli_args.project_dir.resolve()}")
    print(f"    - config      = {cli_args.config_path.resolve()}")
    print(f"    - output_dir  = {output_dir.resolve()}")
    print(f"    - altitude    = {cli_args.config.altitude}")
    print(f"    - ddy         = {cli_args.config.ddy}")
    print(f"    - sample_budget_per_developmentclass = {cli_args.config.sample_budget_per_developmentclass}")
    print(f"    - min_stands_per_stratum = {cli_args.config.min_stands_per_stratum}")
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

    print("2. Latest-measurement selection -- each stand's own latest type=1 record:")
    snapshot = select_latest_measured_snapshot(layers.treestand, stand_ids)
    n_excluded = len(stand_ids) - len(snapshot)
    print(f"   -> {len(snapshot):,} / {len(stand_ids):,} stands kept")
    if n_excluded > 0:
        print(f"      ({n_excluded} stand(s) had no type=1 record at all and are excluded)")
    print()
    print_year_distribution(snapshot)

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

    print("4. Structural + species-data checks -- geometry, treestandid, diameter/height:")
    candidates, structural_skips = build_stand_candidates(merged_filtered, layers.treestratum)
    print(f"   -> {len(candidates):,} / {len(merged_filtered):,} stands kept")
    print_skips(structural_skips, "   Skipped (invalid or missing data)")
    print()

    print("5. Viability check -- nonzero total basal area:")
    viable_candidates, ba_skips = partition_viable_candidates(candidates)
    print(f"   -> {len(viable_candidates):,} / {len(candidates):,} stands kept")
    print_skips(ba_skips, "   Skipped (zero basal area)")

    parsed_stands, build_skips = build_stands(viable_candidates)
    print_skips(build_skips, "   Skipped (could not build stand record)")
    print()
    print(f"Stands surviving filters: {len(parsed_stands):,}")

    print_section("Stratified sampling")
    selected_stands, sampling_skips = select_stratified_sample(
        parsed_stands,
        cli_args.config.sample_budget_per_developmentclass,
        cli_args.config.min_stands_per_stratum,
    )
    print_skips(sampling_skips, "Skipped (no peat_type)")
    print(f"Sampled for allometry generation: {len(selected_stands):,} / {len(parsed_stands):,}")

    if cli_args.dry_run:
        print_section("Writing (dry run -- nothing is written)")
        plans = [plan_stand_outputs(stand, output_dir) for stand in selected_stands]
        print_dry_run_plan(plans, output_dir, cli_args.project_dir)
        return

    print_section("Writing")
    print(f"Destination folder: {output_dir.resolve()}")
    output_dir.mkdir(parents=True)
    print()

    # NOTE (2026-10-04): process_stand() now requires a scaling_factor
    # (create-input-structure's dense young stand scaling, PR #312). Decided
    # here, scoped to selected_stands (the sampled subset actually written),
    # mirroring metsakeskus_to_allometry.py's own call site.
    strata_per_stand = {stand.id: stand.strata for stand in selected_stands}
    scaling_factors = decide_scaling_factors(
        strata_per_stand, cli_args.config.dense_young_stand_scaling
    )

    outcomes: list[StandOutcome] = [
        process_stand(
            stand,
            cli_args.config,
            output_dir,
            scaling_factor=scaling_factors[stand.id],
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
            altitude=cli_args.config.altitude,
            ddy=cli_args.config.ddy,
            stands=final_stands,
        ),
    )
    print(f"Informational JSON written: {json_path}")


if __name__ == "__main__":
    main()