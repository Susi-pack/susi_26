---
icon: lucide/table-2
---

# Metsäkeskus data --> allometry files

`susi-metsakeskus-to-allometry` (`src/tools/metsakeskus_to_allometry/metsakeskus_to_allometry.py`) converts a Metsäkeskus forest inventory
GeoPackage (`.gpkg`) into the allometry CSVs SUSI reads.
Every stand that survives the filters below is converted: there is no sampling
and no grouping of stands.

The tool reads the GeoPackage in **Metsäkeskus's 2026 data model** (version
2.1.x), the one their open forest data has been distributed in since 2026. A file
in the old, pre-2026 format is refused with a message saying so: convert it
with the deprecated
[old-format tool](metsakeskus_to_allometry_old_format.md) instead, or download
the data again.

This page is the reference for the tool's parameters, inputs and outputs.
For a walkthrough of an actual run, see
[How to generate allometry files from Metsäkeskus data](how_to_generate_allometry_from_metsakeskus.md).

To generate the same kind of file from a Finnish forestry XML stand export
instead, see [XML data --> allometry files](xml_to_allometry.md).

## Synopsis

```bash
susi-metsakeskus-to-allometry INPUT_GPKG \
    --project-dir PROJECT_DIR [--config CONFIG.toml] \
    [--allometry-set-name NAME] [--allow-out-of-range-values] [--dry-run]
```

## Command-line arguments

| Argument | Required | Description |
|---|---|---|
| `INPUT_GPKG` | yes | The Metsäkeskus GeoPackage, in the 2026 data model. Must exist and end in `.gpkg`. |
| `--project-dir` | yes | Path to the project's folder. Decides where the output goes — `<project-dir>/inputs/allometry/` — and where the config file is looked up by default. |
| `--config` | no | Path to the TOML config file (see below). Defaults to `<project-dir>/inputs/config.toml`. Must exist and end in `.toml`. |
| `--allometry-set-name` | no | Write a named allometry set: `inputs/allometry_NAME/` and `inputs/stand_data_NAME.json`. See [Several allometry sets in one project](#several-allometry-sets-in-one-project). |
| `--allow-out-of-range-values` | no | Downgrade an out-of-range `altitude`/`ddy` from an error to a warning. `NaN` is rejected either way. |
| `--dry-run` | no | Report what the run would produce and exit, writing nothing at all. See [Dry runs](#dry-runs). |

The tool writes into `<project-dir>/inputs/allometry/` and
`<project-dir>/inputs/stand_data.json`, or the named set's pair with
`--allometry-set-name`. Neither may already exist — the tool refuses to write
over a previous run's output rather than overwrite it.
So rename or move the existing `allometry/` and `stand_data.json` first. (If it holds a
`new_growth/` subfolder from `new_growth_allometry.py`, move that back into
the fresh `allometry/` afterwards.) The folder is created just before the files are written, so
a run that fails while reading or filtering leaves nothing behind.

The tool also prints the fully-resolved (absolute) path it read the config
from and the path it writes to, so what ends up on disk is never ambiguous
relative to the directory you happened to run it from.

## Config file

A TOML file. By default the tool looks for `config.toml` inside
`--project-dir`'s `inputs/` folder; pass `--config` to use a different name or location. Copy
[`src/tools/metsakeskus_to_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/metsakeskus_to_allometry/default_config.toml)
and edit it: the optional fields are listed with the values the tool applies
when they are absent, while the four required ones carry deliberately invalid
placeholders for you to replace. Unknown fields are rejected, and every missing
required field is reported at once.

A config written for the old-format tool does not work here as it is: its
`developmentclass_filter` is rejected as an unknown field, and
`mean_diameter_range_filter` is missing. See
[Mean-diameter range](#mean-diameter-range) for why one replaced the other.

### Required

Neither altitude nor temperature sum is part of the Metsäkeskus data, so you
supply both. One value applies to every stand in the run.

| Field | Unit | Enforced range | Description |
|---|---|---|---|
| `target_year` | year | — | Which inventory snapshot to use. Only stands with an inventory snapshot dated in exactly this year are converted. |
| `altitude` | m above sea level | 0 – 1000 | Site elevation. |
| `ddy` | degree days per year | 500 – 2000 | Temperature sum. |
| `mean_diameter_range_filter` | cm | `[minimum, maximum]`, minimum below maximum | Keep only stands whose mean diameter is in this range, both ends included. No default: see [Mean-diameter range](#mean-diameter-range). |

The enforced ranges are deliberately wider than Finnish forest land, to catch a
mistyped value (metres vs feet, say) without rejecting real ones. A value
outside the range blocks the run unless you pass `--allow-out-of-range-values`.

### Optional

| Field | Default | Description |
|---|---|---|
| `fertilityclass_filter` | `[2, 3, 4, 5]` | Peatland fertility classes to keep. The default excludes the richest and poorest extremes. The codes are listed in the `.gpkg`'s own `codeschema_mapping` table (see [Code meanings](#code-meanings)). |
| `n_trees` | `20` | Reference trees per species in the growth model. |
| `start_year` | `5` | First projected step, in years after the inventory snapshot. |
| `end_year` | `80` | Last projected step, in years after the snapshot. |
| `step_years` | `5` | Interval between projected steps. |

The last four are passed straight to `Growth_and_Yield_Table`, and set how far
forward each stand's growth is projected and at what resolution. Every table
also opens with a row for the snapshot itself, at `Year` 0, so the defaults
produce 17 rows per file. `Year` counts from the snapshot; the `Age` column
adds the stand's recorded age to it.

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

### Mean-diameter range

`mean_diameter_range_filter` leaves out stands too young or too old to grow
into a useful allometry curve. A stand is kept when its **mean diameter** — the
basal-area-weighted mean of its species' mean diameters, the same figure
recorded as `mean_diameter` in `stand_data.json` — is within the range.

There is **no default**, because what counts as too young or too old depends on
the region: trees in the south reach a given diameter at a different age than
in Lapland. As an example only, **`[8, 21]` suits Pirkanmaa**: it keeps roughly
what Metsäkeskus classes there as young and advanced thinning stands (`02` and
`03`: from about the 10th percentile of `02`'s mean diameter, to the 90th of
`03`'s), and leaves out most seedling stands and mature stands.

To choose a range for your own region, make a [dry run](#dry-runs) with a wide
range, e.g. `[0, 100]`. Before the range is applied, the tool prints a table of
mean diameters (10th percentile, median, 90th percentile) for each development
class, over the stands that reached that point:

```text
   developmentclass  n_stands  p10  median  p90
             (none)         1 14.2    14.2 14.2
                 02         2  9.0    13.1 17.2
                 03        17 13.4    15.7 17.4
                 04         8 17.7    18.8 22.7
```

`(none)` collects the stands whose inventory snapshot records no development
class, which is every remote-sensed one.

!!! warning "A minimum of 8 cm or more turns off dense young stand scaling"

    [Dense young stand scaling](dense_young_stand_scaling.md) only acts on
    stands with a mean diameter below its `max_mean_diameter` (8 cm by default).
    With a range starting at 8 cm, those stands are already gone by the time it
    runs, so it could never scale anything: the tool refuses a config file that
    turns the scaling on with such a range. Lower the minimum, or leave the
    scaling off.

??? info "Why a diameter range, not the development class?"

    Metsäkeskus classes every stand into a development class
    (*kehitysluokka*): `A0` open area, `S0` seed-tree stand, `T1`/`T2`
    seedling stand below/above 1.3 m, `Y1` seedling stand with an overstorey,
    `02` young thinning stand, `03` advanced thinning stand, `04` mature stand,
    `05` shelterwood stand, `ER` uneven-aged stand. The old-format tool filtered
    on it. This tool doesn't, for three reasons:

    - **Remote-sensed snapshots have no class.** From 2020 on, nearly every
      inventory in the data is laser scanning, and Metsäkeskus records no
      development class on those snapshots. A class filter would silently keep
      only field-measured stands.
    - **The class and the trees often disagree.** In the Pirkanmaa data,
      field-measured snapshots classed as seedling stands (`T1`/`T2`) that
      record any basal area have a median mean diameter of about 22 cm and
      height of 19 m, probably trees left standing after a cutting. The growth
      model grows those trees, whatever the label says.
    - **A stand's current class is not its class at the inventory date.** The
      old format had the class on the stand, as it was when the data was
      exported. For a 2005 snapshot that is a class twenty years later, and in
      the Pirkanmaa data it differs from the snapshot's own class for about
      half the stands.

    The mean diameter has none of these problems: every inventory snapshot with
    trees has one, it describes exactly the trees the growth model is given, on
    the snapshot's own date. The official boundaries between seedling, young
    thinning and advanced thinning stands are themselves diameters (about 8 and
    16 cm). The boundary to a mature stand is not: it depends on species, site
    and region, which is why there is no single default.

    The old filter, `developmentclass_filter = [1, 2, 3]`, never did what its
    description said either. The codes are text (`02`, `03`, `T1`...), not the
    numbers 1, 2, 3, so it kept exactly `02` and `03` and dropped every other
    class.

    The development class is still recorded in `stand_data.json`, for
    information only.

## Input data

Four layers of the GeoPackage are read, and only these columns of each:

| Layer | One row per | Columns used |
|---|---|---|
| `stand` | stand | `standid`, `area`, `geometry` |
| `growthplacedata` | stand | `standid`, `maingroup`, `subgroup`, `fertilityclass`, `soiltype`, `drainagestate` |
| `treestand` | tree-stand record of a stand | `standid`, `treestandid`, `treestandclass`, `treestanddatasource`, `treedatadate`, `developmentclass` |
| `treestratum` | stratum of a tree-stand record | `treestandid`, `treespecies`, `basalarea`, `stemcount`, `age`, `meandiameter`, `meanheight` |

Before reading anything else, the tool checks that the file has every one of
these layers and columns. A file without the `growthplacedata` layer and with
the site columns on `stand` is in the old format, and the tool says so and
stops; any other missing layer or column is named in the error.

??? info "How the four layers relate"

    One **stand** (*kuvio*) is one mapped forest polygon. Its site — soil,
    drainage, fertility — is in **growthplacedata** (*kasvupaikkatieto*), one
    row per stand, joined on `standid`.

    A stand's trees are not stored on the stand itself. Each row of
    `treestand` is one *record* of that stand's trees on one date
    (`treedatadate`), so a stand inventoried repeatedly, or projected forward,
    has several. `treestandclass` says which kind of record it is:

    | `treestandclass` | Meaning | Used |
    |---|---|---|
    | `1` | Inventory (*inventointipuusto*): the trees as an inventory found them | yes |
    | `2` | Current state (*nykytilan puusto*): Metsäkeskus's projection to the current date | no |
    | `3` | Development forecast (*kehitysennusteen puusto*): ten years ahead | no |
    | `4` | Dead trees | no |

    Each record in turn has several rows in `treestratum`, joined by
    `treestandid`: one row per species and diameter cohort. This is the
    Metsäkeskus *stand stratum*, and it is not SUSI's canopy layer. Several
    strata of the same species collapse into one of SUSI's three species slots,
    and those slots are then ranked into the dominant and subdominant canopy
    layers.

    `treestandsummary` is deliberately not read: the tool aggregates the strata
    itself, the same way for every stand.

### Inventory snapshots and their sources

An **inventory snapshot** is a `treestand` record with `treestandclass` = 1.
Metsäkeskus counts several sources as an inventory, and so does this tool: it
uses a snapshot whatever its source (`treestanddatasource`):

| `treestanddatasource` | Source |
|---|---|
| `1` | Measured in the field. Dated by the measurement. |
| `2` | Remote-sensed (laser scanning). Dated by the scan. |
| `9` | Combined from several sources. |
| `21`, `22`, `23` | Reported by the forest owner, an authority, or an operator. |

The source matters for what you are converting. In the Pirkanmaa data, almost
every inventory up to 2019 is field-measured, and from 2020 on almost every one
is remote-sensed. That is why the tool prints, before choosing the snapshots,
how many stands have an inventory snapshot in each year **and from each
source**:

```text
 year  n_stands  source 1  source 2  source 9  source 21
 2005        35        31         0         4          0
 2020     20222       170     20037        52          0
```

Metsäkeskus describes the inventory classes and sources in section 3 of its
GeoPackage content guide (*GeoPackagen sisältöohjeet*), available on
[Metsäkeskus's open forest and nature data pages](https://www.metsakeskus.fi/fi/avoin-metsa-ja-luontotieto).

### Code meanings

Every Metsäkeskus GeoPackage carries its own code list: the
`codeschema_mapping` table. Each row gives a table, a field, a code and the
code's meaning in Finnish, for every coded field in the file — `maingroup`,
`subgroup`, `fertilityclass`, `drainagestate`, `treestandclass`,
`treestanddatasource`, `developmentclass`, `treespecies` and the rest. Open it
in QGIS, or read it with GeoPandas:

```python
import geopandas as gpd

codes = gpd.read_file("MV_Pirkanmaa.gpkg", layer="codeschema_mapping")
codes[codes["field_name"] == "developmentclass"]
```

The data model itself, and the rest of Metsäkeskus's documentation, are on
[Metsäkeskus's open forest and nature data pages](https://www.metsakeskus.fi/fi/avoin-metsa-ja-luontotieto).

## What the tool does

The run is printed in three sections — reading, filtering, and writing.
Each filter reports how many stands it kept.

**1. Site filter.** Keeps stands that are forest land (`maingroup` = 1), on
peatland (`subgroup` 2 *korpi* or 3 *räme*), already drained (`drainagestate`
7 *ojikko*, 8 *muuttuma* or 9 *turvekangas*), and whose `fertilityclass` is in
`fertilityclass_filter`. Everything but the fertility class is hard-coded. All
of these come from `growthplacedata`.

??? info "Why only drained peatland forest land?"

    Because that is what SUSI simulates. Mineral soils (`subgroup` = 1) are
    outside the model altogether, and so are pristine, undrained mires
    (`drainagestate` = 6) — the model's water and peat dynamics assume a
    drainage network exists. These stands are not merely unwanted here: SUSI
    has nothing meaningful to say about them, so they are excluded in code
    rather than left to the config.

**2. Inventory-snapshot filter.** Keeps one `treestand` row per stand: the
inventory snapshot (`treestandclass` = 1, any source) dated in `target_year`,
the earliest one if there are several that year. A stand without one is
dropped. Before applying the filter, the tool prints every inventory year
present in your data, with how many stands it covers per source.

??? info "Why an exact year, and only inventories?"

    `treestandclass` 2 and 3 records are Metsäkeskus's own projections, grown
    forward to the current date and ten years beyond it. Feeding one of those
    into SUSI would model the same growth twice — once in Metsäkeskus's model,
    again in SUSI's — so only inventory snapshots are used.

    The year match is exact, with no "nearest available year" fallback. The
    benefit is that every stand in a run shares one inventory date, so the
    stands are comparable to each other; the cost is that stands inventoried in
    other years are lost. That is usually the largest single drop in the run,
    which is why the year table is printed just before it.

**3. Structural and species-data checks.** Drops a stand with more than one
row in `growthplacedata` or `stand` (the data model has one per stand), then
one with no usable geometry or no `treestandid`, then collapses its `treestratum` rows into SUSI's
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

**4. Viability check.** Drops stands whose basal area is zero across all three
species: there is no growth to model.

**5. Mean-diameter filter.** Prints the mean-diameter table by development
class, then keeps the stands whose mean diameter is within
`mean_diameter_range_filter` (see [Mean-diameter range](#mean-diameter-range)).

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
it still happens: the GeoPackage is read, every filter runs, the year table,
the mean-diameter table and every skipped stand are reported exactly as in a
real run. The tool then prints the files it would have written, and exits.

Nothing at all is created — no CSVs, no `stand_data.json`, and not
even the output folder, so a dry run does not claim a `--project-dir` that the
real run then has to work around. The one thing it does still enforce is the
refusal to run into an existing output folder: whether the real run could start
is part of what a dry run is for.
The file names printed are the ones a real run would produce, decided by the same code.

The counts are an **upper bound**. Computing the growth trajectories is the
slow part of a run and the part `--dry-run` skips, so a stand that would fail
inside the growth model is still counted here. Every other skip — geometry,
missing data, zero basal area, mean diameter — is reported in full, because
those stages all ran.

## Several allometry sets in one project

A project whose stands come from several inventory exports — one per region,
say — keeps one **allometry set** per export: a stand-data document plus the
folder of allometry files it references. `--allometry-set-name NAME` writes a
named set beside the default one:

| | Default set | `--allometry-set-name pirkanmaa` |
|---|---|---|
| Allometry files | `inputs/allometry/` | `inputs/allometry_pirkanmaa/` |
| Stand-data document | `inputs/stand_data.json` | `inputs/stand_data_pirkanmaa.json` |

The name may contain only letters, digits, underscores and dashes. Each set
gets its own run of the tool, usually with its own `--config` (each export has
its own `target_year`, `altitude` and `ddy`). The same refusal applies to each
set: if either its folder or its document already exists, the run stops.
Renaming a set by hand afterwards breaks it, because the document records its
files' paths, folder name included. Regenerate the set under the new name
instead.

A run script loads each set's document and builds the stands from all of them:

```python
from susi.io.project_layout import stand_data_path_for_project
from susi.io.stand_data import build_stand_params, load_stand_data_document_from_json

for region in ("pirkanmaa", "pohjois_pohjanmaa"):
    document = load_stand_data_document_from_json(
        stand_data_path_for_project(PROJECT_DIR, set_name=region)
    )
    for stand_id in document.stands:
        stand_params = build_stand_params(document, stand_id, n=N_SOIL_COLS)
        ...  # plus this region's weather, photosynthesis parameters, ...
```

Two limitations:

- **Stand IDs must be unique across the sets used in one run.** A stand's
  output folder is named by its stand ID alone, so two stands with the same ID
  would collide. SUSI refuses such a run before it starts ("Duplicate
  simulation folder paths detected").
- **The optimization reads only the default set's `stand_data.json`** for
  stand areas. Runs built from named sets can't be optimized yet.

## Output files

| File | Written |
|---|---|
| `<standid>_dominant.csv` | Always, one per surviving stand. |
| `<standid>_subdominant.csv` | Only when the second-ranked species carries basal area above zero. |
| `stand_data.json` | Always, in the project's `inputs/` (`stand_data_NAME.json` for a named set). One `StandData` entry per converted stand — fertility class, allometry files, YKJ coordinates, polygon, stand-level means, stem count, the snapshot's development class (`null` when it records none), and whether [dense young stand scaling](dense_young_stand_scaling.md) was applied. Read by `build_stand_params` and by `new_growth_allometry.py`'s sourced mode. |

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
| `standid appears N times in the <layer> layer` | The stand has more than one row in `growthplacedata` or `stand`, so its site, area or polygon is ambiguous. |
| `no usable geometry` | The stand polygon is missing or empty. |
| `no treestandid` | The snapshot has no id to join `treestratum` on, so there are no trees to read. |
| `<species>: basal area X m2/ha but no usable mean diameter` (or `mean height`) | A species with real, recorded basal area has no diameter or height to go with it. |
| `zero basal area across all species` | No growth data at all. Expected for recently cleared or seedling stands. |
| `mean diameter X cm, outside A-B cm` | The stand's mean diameter is outside `mean_diameter_range_filter`. |
| Anything else | The growth model or the coordinate transform raised for this stand; the message quotes the underlying error. |

??? info "Why stands are skipped rather than patched"

    A missing diameter could be replaced with a plausible number, and the stand
    would go through. But that number would then be compounded: the stem count
    is derived from basal area and diameter, so a real basal area and an
    invented diameter produce an invented stem count that is indistinguishable
    from a recorded one, in a file that looks exactly like every other output.

    The tool would rather lose the stand and say so. The same rule applies to
    `soiltype` and `developmentclass`: when the export doesn't record them,
    they stay absent all the way through — `null` in the JSON —
    rather than being filled in.

## Known warnings

Reading a Metsäkeskus GeoPackage with other tools (QGIS, GeoPandas) can print
two warnings this tool silences or avoids, and both are harmless:

- *Measured (M) geometry types are not supported*: the stand polygons are
  stored with a measure value per vertex, which is dropped on reading. Only the
  x/y coordinates are used.
- *Error parsing datetimes … `T20:02:60Z`*: a few `stand.updatetime` values
  carry a leap second. This tool doesn't read that column.
