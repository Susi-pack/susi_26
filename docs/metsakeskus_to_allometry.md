---
icon: lucide/table-2
---

# Metsäkeskus data --> allometry files

`susi-metsakeskus-to-allometry` (`src/tools/metsakeskus_to_allometry/metsakeskus_to_allometry.py`) converts a Metsäkeskus forest inventory
GeoPackage (`.gpkg`) into the allometry CSVs SUSI reads.
Every stand that survives the filters below is converted: there is no sampling
and no grouping of stands.

This page is the reference for the tool's parameters, inputs and outputs.
For a walkthrough of an actual run, see
[How to generate allometry files from Metsäkeskus data](how_to_generate_allometry_from_metsakeskus.md).

To generate the same kind of file from a Finnish forestry XML stand export
instead, see [XML data --> allometry files](xml_to_allometry.md).

## Synopsis

```bash
susi-metsakeskus-to-allometry INPUT_GPKG \
    --project-dir PROJECT_DIR [--config CONFIG.toml] \
    [--allow-out-of-range-values] [--dry-run]
```

## Command-line arguments

| Argument | Required | Description |
|---|---|---|
| `INPUT_GPKG` | yes | The Metsäkeskus GeoPackage. Must exist and end in `.gpkg`. |
| `--project-dir` | yes | Path to the project's folder. Decides where the output goes — `<project-dir>/allometry/` — where the config file is looked up by default. |
| `--config` | no | Path to the TOML config file (see below). Defaults to `<project-dir>/config.toml`. Must exist and end in `.toml`. |
| `--allow-out-of-range-values` | no | Downgrade an out-of-range `altitude`/`ddy` from an error to a warning. `NaN` is rejected either way. |
| `--dry-run` | no | Report what the run would produce and exit, writing nothing at all. See [Dry runs](#dry-runs). |

The output folder is not selectable: `<project-dir>/allometry/` is the
only place this tool writes. It must **not** already exist — the tool refuses
to run into a previous run's output rather than overwrite it.
So rename or move the existing `allometry/` first. (If it holds a
`new_growth/` subfolder from `new_growth_allometry.py`, move that back into
the fresh `allometry/` afterwards.) The folder is created just before the files are written, so
a run that fails while reading or filtering leaves nothing behind.

The tool also prints the fully-resolved (absolute) path it read the config
from and the path it writes to, so what ends up on disk is never ambiguous
relative to the directory you happened to run it from.

## Config file

A TOML file. By default the tool looks for `config.toml` directly inside
`--project-dir`; pass `--config` to use a different name or location. Copy
[`src/tools/metsakeskus_to_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/metsakeskus_to_allometry/default_config.toml)
and edit it: the optional fields are listed with the values the tool applies
when they are absent, while the three required ones carry deliberately invalid
placeholders for you to replace. Unknown fields are rejected, and every missing
required field is reported at once.

### Required

Neither altitude nor temperature sum is part of the Metsäkeskus data, so you
supply both. One value applies to every stand in the run.

| Field | Unit | Enforced range | Description |
|---|---|---|---|
| `target_year` | year | — | Which inventory snapshot to use. Only stands measured in exactly this year are converted. |
| `altitude` | m above sea level | 0 – 1000 | Site elevation. |
| `ddy` | degree days per year | 500 – 2000 | Temperature sum. |

The enforced ranges are deliberately wider than Finnish forest land, to catch a
mistyped value (metres vs feet, say) without rejecting real ones. A value
outside the range blocks the run unless you pass `--allow-out-of-range-values`.

### Optional

| Field | Default | Description |
|---|---|---|
| `developmentclass_filter` | `[1, 2, 3]` | Metsäkeskus development classes to keep: 1 = open/seedling, 2 = young growing, 3 = grown-up. |
| `fertilityclass_filter` | `[2, 3, 4, 5]` | Peatland fertility classes to keep. The default excludes the richest and poorest extremes. |
| `n_trees` | `20` | Reference trees per species in the growth model. |
| `start_year` | `5` | First projected step, in years after the inventory snapshot. |
| `end_year` | `80` | Last projected step, in years after the snapshot. |
| `step_years` | `5` | Interval between projected steps. |

The last four are passed straight to `Growth_and_Yield_Table`, and set how far
forward each stand's growth is projected and at what resolution. Every table
also opens with a row for the snapshot itself, at `Year` 0, so the defaults
produce 17 rows per file. `Year` counts from the snapshot; the `Age` column
adds the stand's measured age to it.

#### `[dense_young_stand_scaling]`

An optional table that scales down the stem count and basal area of dense
young stands before the growth model runs. It is off by default, and explained
in full on its own page: [Dense young stand scaling](dense_young_stand_scaling.md).
A TOML table has to come after every top-level key, so keep it at the end of
the file.

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Turns the scaling on. |
| `max_mean_diameter` | `8.0` | Only stands with a mean diameter below this are scaled, cm. |
| `stem_count_threshold_spruce` | `2200` | A spruce-dominated stand is scaled when its stem count is above this, stems/ha. |
| `stem_count_threshold_other` | `2500` | The same, for a pine- or deciduous-dominated stand. |
| `target_stem_count_spruce` | `1800` | The stem count a scaled spruce-dominated stand starts from, stems/ha. |
| `target_stem_count_other` | `2000` | The same, for a pine- or deciduous-dominated stand. |

## Input data

Three layers of the GeoPackage are read:

| Layer | Columns used |
|---|---|
| `stand` | `standid`, `maingroup`, `subgroup`, `drainagestate`, `fertilityclass`, `developmentclass`, `soiltype`, `geometry` |
| `treestand` | `standid`, `treestandid`, `date`, `type` |
| `treestratum` | `treestandid`, `treespecies`, `basalarea`, `stemcount`, `age`, `meandiameter`, `meanheight` |

??? info "How `stand`, `treestand` and `treestratum` relate"

    One **stand** (*kuvio*) is one mapped forest polygon, and carries the site
    attributes: soil, drainage, fertility, development class.

    A stand's trees are not stored on the stand itself. Each row of `treestand`
    is one *snapshot* of that stand's trees on one date, so a stand measured
    repeatedly, or projected forward, has several. `type` says which kind of
    snapshot it is: `1` is measured data, while `2` and `3` are Metsäkeskus's
    own grown-forward projections.

    Each snapshot in turn has several rows in `treestratum`, joined by
    `treestandid`: one row per species and diameter cohort. This is the
    Metsäkeskus *stand stratum*, and it is not SUSI's canopy layer. Several
    strata of the same species collapse into one of SUSI's three species slots,
    and those slots are then ranked into the dominant and subdominant canopy
    layers.

    `treestandsummary` is deliberately not read: it is only populated for the
    projected snapshots, not for measured ones.

## What the tool does

The run is printed in three sections — reading, filtering, and writing.
Each filter reports how many stands it kept.

**1. Site filter.** Keeps stands that are forest land (`maingroup` = 1), on
peatland (`subgroup` 2 *korpi* or 3 *räme*), already drained (`drainagestate`
7 *ojikko*, 8 *muuttuma* or 9 *turvekangas*), and whose `fertilityclass` is in
`fertilityclass_filter`. Everything but the fertility class is hard-coded.

??? info "Why only drained peatland forest land?"

    Because that is what SUSI simulates. Mineral soils (`subgroup` = 1) are
    outside the model altogether, and so are pristine, undrained mires
    (`drainagestate` = 6) — the model's water and peat dynamics assume a
    drainage network exists. These stands are not merely unwanted here: SUSI
    has nothing meaningful to say about them, so they are excluded in code
    rather than left to the config.

**2. Measured-snapshot filter.** Keeps one `treestand` row per stand: the
measured one (`type` = 1) whose date falls in `target_year`. A stand without
one is dropped. Before applying the filter, the tool prints every measured
year present in your data and how many stands it covers.

??? info "Why an exact year, and only measured data?"

    `type` 2 and 3 snapshots are Metsäkeskus's own projections, grown forward
    to a common date and roughly ten years beyond it. Feeding one of those into
    SUSI would model the same growth twice — once in Metsäkeskus's model, again
    in SUSI's — so only measured data is used.

    The year match is exact, with no "nearest available year" fallback. The
    benefit is that every stand in a run shares one inventory date, so the
    stands are comparable to each other; the cost is that stands measured in
    other years are lost. That is usually the largest single drop in the run,
    which is why the year table is printed just before it.

**3. Development-class filter.** Keeps stands whose `developmentclass` is in
`developmentclass_filter`.

**4. Structural and species-data checks.** Drops a stand with no usable
geometry or no `treestandid`, then collapses its `treestratum` rows into SUSI's
three species slots:

| SUSI slot | `treespecies` codes |
|---|---|
| 1, pine | 1 |
| 2, spruce | 2 |
| 3, deciduous | 3, 4, 5, 6, 7, 8, 9, 15, 20, 29 |

Within a slot, age, mean diameter and mean height are averaged weighted by
basal area, and stem counts are summed — or estimated from basal area and mean
diameter when the inventory did not record them. A species carrying real basal
area but no usable diameter or height drops the whole stand (see
[Skipped stands](#skipped-stands)).

**5. Viability check.** Drops stands whose basal area is zero across all three
species: there is no growth to model.

**Dense young stand scaling.** Not a filter: it removes no stand. After the
last check, the tool reports
[dense young stand scaling](dense_young_stand_scaling.md#with-metsakeskus-data):
with the option on, one line per stand it scales down; and in any case a
warning naming the dense young stands that are about to be grown with more
stems than the default limits allow. A scaled stand has every species' stem
count and basal area multiplied by its scaling factor before the growth model
runs, and before the stand is split into its two layers.

**Writing.** For each surviving stand, the polygon centroid is transformed from
EPSG:3067 (ETRS-TM35FIN) to EPSG:2393 (YKJ) for the growth model, the species
are ranked by basal area into a dominant and a subdominant, and one growth
trajectory is computed per layer. `stand_data.json` records the stand as the
inventory reports it, and says whether it was scaled and by what factor.

## Dry runs

`--dry-run` stops the run immediately before that last stage. Everything above
it still happens: the GeoPackage is read, every filter runs, the year table and
every skipped stand are reported exactly as in a real run. The tool then prints
the files it would have written, and exits.

Nothing at all is created — no CSVs, no `stand_data.json`, and not
even the output folder, so a dry run does not claim a `--project-dir` that the
real run then has to work around. The one thing it does still enforce is the
refusal to run into an existing output folder: whether the real run could start
is part of what a dry run is for.
The file names printed are the ones a real run would produce, decided by the same code.

The counts are an **upper bound**. Computing the growth trajectories is the
slow part of a run and the part `--dry-run` skips, so a stand that would fail
inside the growth model is still counted here. Every other skip — geometry,
missing data, zero basal area — is reported in full, because those stages all
ran.

## Output files

| File | Written |
|---|---|
| `<standid>_dominant.csv` | Always, one per surviving stand. |
| `<standid>_subdominant.csv` | Only when the second-ranked species carries basal area above zero. |
| `stand_data.json` | Always, in the project's `inputs/`. One `StandData` entry per converted stand — fertility class, allometry files, YKJ coordinates, polygon, stand-level means, stem count, and whether [dense young stand scaling](dense_young_stand_scaling.md) was applied. Read by `build_stand_params` and by `new_growth_allometry.py`'s sourced mode. |

Each CSV follows the canonical allometry schema.
The columns declared in `susi.core.allometry_columns.ALLOMETRY_COLUMNS` and validated on read by `read_allometry_info_from_csv`.

??? info "Why two single-species files per stand?"

    `Growth_and_Yield_Table` pools every species with a nonzero stem count into
    one shared list of reference trees. A curve for a single canopy layer
    therefore only exists if the other species are zeroed, so the tool calls the
    growth model once per layer, each time with only that layer's species alive.
    The result maps directly onto `CanopyLayerAllometry`'s `dominant` and
    `subdominant` layers.

    For a monoculture, the subdominant file is simply not written: the
    second-ranked species is genuinely absent, and duplicating the dominant
    layer would double-count its basal area for anything that sums across
    layers.

## Skipped stands

A stand that fails a check is reported with its id and a reason, and the run
continues. One bad stand never aborts the others.

| Reason | Meaning |
|---|---|
| `no usable geometry` | The stand polygon is missing or empty. |
| `no treestandid` | The snapshot has no id to join `treestratum` on, so there are no trees to read. |
| `<species>: basal area X m2/ha but no usable mean diameter` (or `mean height`) | A species with real, measured basal area has no diameter or height to go with it. |
| `zero basal area across all species` | No growth data at all. Expected for recently cleared or seedling stands. |
| Anything else | The growth model or the coordinate transform raised for this stand; the message quotes the underlying error. |

??? info "Why stands are skipped rather than patched"

    A missing diameter could be replaced with a plausible number, and the stand
    would go through. But that number would then be compounded: the stem count
    is derived from basal area and diameter, so a real basal area and an
    invented diameter produce an invented stem count that is indistinguishable
    from a measured one, in a file that looks exactly like every other output.

    The tool would rather lose the stand and say so. The same rule applies to
    `soiltype`: when the export doesn't record it, it stays absent all the way
    through — `null` in the JSON—
    rather than being filled in with a number.
