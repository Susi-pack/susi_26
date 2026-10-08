---
icon: lucide/file-input
---

# XML data --> allometry files

`susi-xml-to-allometry` (`src/tools/xml_to_allometry/xml_to_allometry.py`) converts a Finnish national forest XML stand export (metsätietostandardit) into the allometry CSVs SUSI needs.
Every stand in the file that carries tree strata data is converted: there is no sampling and no grouping of stands.

This page is the reference for the tool's parameters, inputs and outputs.
For a walkthrough of an actual run, see
[How to generate allometry files from XML data](how_to_generate_allometry_from_xml.md).

To generate the same kind of file from a Metsäkeskus GeoPackage instead, see
[Metsäkeskus data --> allometry files](metsakeskus_to_allometry.md).

## Synopsis

```bash
susi-xml-to-allometry INPUT_XML \
    --project-dir PROJECT_DIR [--config CONFIG.toml] \
    [--allow-out-of-range-values] [--dry-run]
```

## Command-line arguments

| Argument | Required | Description |
|---|---|---|
| `INPUT_XML` | yes | The XML stand export. Must exist and end in `.xml`. |
| `--project-dir` | yes | Path to the project's folder. Decides where the output goes — `<project-dir>/allometry/` — and where the config file is looked up by default. |
| `--config` | no | Path to the TOML config file (see below). Defaults to `<project-dir>/config.toml`. Must exist and end in `.toml`. |
| `--allow-out-of-range-values` | no | Downgrade an out-of-range `altitude`/`ddy` from an error to a warning. `NaN` is rejected either way. |
| `--dry-run` | no | Report what the run would produce and exit, writing nothing at all. See [Dry runs](#dry-runs). |

The output folder is not selectable: `<project-dir>/allometry/` is the
only place this tool writes. It must **not** already exist — the tool refuses
to run into a previous run's output rather than overwrite it, so to
regenerate it, rename or move the existing `allometry/` first. (If it holds a
`new_growth/` subfolder from `new_growth_allometry.py`, move that back into
the fresh `allometry/` afterwards.) The folder is created just before the files are written, so
a run that fails while reading or filtering leaves nothing behind.

The tool also prints the fully-resolved (absolute) path it read the config
from and the path it writes to, so what ends up on disk is never ambiguous
relative to the directory you happened to run it from.

## Config file

A TOML file. By default the tool looks for `config.toml` directly inside
`--project-dir`; pass `--config` to use a different name or location. Copy
[`src/tools/xml_to_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/xml_to_allometry/default_config.toml)
and edit it: the optional fields are listed with the values the tool applies
when they are absent, while the two required ones carry deliberately invalid
placeholders for you to replace. Unknown fields are rejected, and every missing
required field is reported at once.

### Required

Neither altitude nor temperature sum is part of the XML standard, so you
supply both. One value applies to every stand in the run.

| Field | Unit | Enforced range | Description |
|---|---|---|---|
| `altitude` | m above sea level | 0 – 1000 | Site elevation. |
| `ddy` | degree days per year | 500 – 2000 | Temperature sum. |

The enforced ranges are deliberately wider than Finnish forest land, to catch a
mistyped value (metres vs feet, say) without rejecting real ones. A value
outside the range blocks the run unless you pass `--allow-out-of-range-values`.

### Optional

| Field | Default | Description |
|---|---|---|
| `n_trees` | `20` | Reference trees per species in the growth model. |
| `start_year` | `5` | First projected step, in years after the stand's measured data. |
| `end_year` | `80` | Last projected step, in years after that. |
| `step_years` | `5` | Interval between projected steps. |

These four are passed straight to `Growth_and_Yield_Table`, and set how far
forward each stand's growth is projected and at what resolution.

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

## Input XML structure

The input XML files follow the **Finnish national forest information
standards** (metsätietostandardit), managed by the Finnish Forest Centre.
These standards define XML schema specifications for exchanging forest
resource data between different operators in the forestry sector.

Key characteristics of the format:

- Uses **GML (Geography Markup Language)** for geospatial data (polygon
  geometries)
- Structured around forest **stands** (kuvio) as the basic unit
- Contains tree stratum data, stand statistics, and metadata

The XML file must follow this hierarchical structure:

```
<ForestPropertyData>
  <Stands>
    <Stand id="...">
      <StandBasicData>
        <FertilityClass>...</FertilityClass>
        <MainGroup>...</MainGroup>
        <SubGroup>...</SubGroup>
        <SoilType>...</SoilType>
        <Area>...</Area>
        <PolygonGeometry>
          <polygonProperty>
            <Polygon>
              <exterior>
                <LinearRing>
                  <coordinates> x1,y1 x2,y2 ... </coordinates>
                </LinearRing>
              </exterior>
            </Polygon>
          </polygonProperty>
        </PolygonGeometry>
      </StandBasicData>
      <TreeStandData>
        <TreeStandDataDate>
          <TreeStandSummary>
            <MeanDiameter>...</MeanDiameter>
            <BasalArea>...</BasalArea>
            <MeanHeight>...</MeanHeight>
            <Volume>...</Volume>
          </TreeStandSummary>
          <TreeStrata>
            <TreeStratum>
              <TreeSpecies>...</TreeSpecies>
              <Age>...</Age>
              <BasalArea>...</BasalArea>
              <StemCount>...</StemCount>
              <MeanDiameter>...</MeanDiameter>
              <MeanHeight>...</MeanHeight>
            </TreeStratum>
            <!-- Up to two more TreeStratum elements -->
          </TreeStrata>
        </TreeStandDataDate>
      </TreeStandData>
    </Stand>
  </Stands>
</ForestPropertyData>
```

### Required elements

| Element | Description |
|---------|-------------|
| `Stand/@id` | Unique stand identifier |
| `StandBasicData/FertilityClass` | Site fertility class (integer) |
| `StandBasicData/MainGroup`, `SubGroup`, `Area` | Parsed unconditionally; a stand missing any of these aborts the whole run rather than being skipped. |
| `StandBasicData/PolygonGeometry` | Stand boundary as GML polygon |
| `TreeStandSummary/MeanDiameter` | Mean diameter at breast height (cm) |
| `TreeStandSummary/BasalArea` | Basal area (m²/ha) |
| `TreeStandSummary/MeanHeight` | Mean height (m) |
| `TreeStandSummary/Volume` | Total volume (m³/ha) |
| `TreeStrata/TreeStratum` | One to three tree strata, each with a species code plus age, basal area, stem count, diameter, and height (see [What the tool does](#what-the-tool-does)). A stand with no `TreeStrata` container at all is not malformed — it is skipped, see [Skipped stands](#skipped-stands). |

### Optional elements

Genuinely optional — a missing `SoilType` tag is read as absent (`null`)
rather than aborting the run:

- `StandBasicData/SoilType` — soil type

## What the tool does

The run is printed in three sections — reading, filtering, and writing.

**Filtering.** The one check applied is presence of tree strata: a stand
whose `TreeStandDataDate` has no `TreeStrata` container at all is skipped
(reported by stand id and reason) rather than aborting the run; every other
piece of missing required data aborts immediately, since it signals a
malformed file rather than an unremarkable gap.

The same section then reports
[dense young stand scaling](dense_young_stand_scaling.md): with the option on,
one line per scaled stand; and in any case a warning naming the dense young
stands that are about to be grown with more stems than the default limits
allow. Neither removes a stand from the run.

**Writing.** For each stand that survives filtering:

- Its up to three `TreeStratum` entries are mapped into fixed species slots —
  species 1 (pine) into slot 0, species 2 (spruce) into slot 1, any other
  species code into slot 2 (deciduous) — with a slot left at zero when the
  stand has no stratum for that species.
- The **main species** is the slot with the largest basal area.
- If [dense young stand scaling](dense_young_stand_scaling.md) is on and the
  stand is a dense young stand, every species' stem count and basal area are
  multiplied by the stand's scaling factor before the growth model runs.
  `stand_data.json` still records the stand as the XML reports it, and says
  whether it was scaled and by what factor.
- The stand's first coordinate pair is transformed from EPSG:3067
  (ETRS-TM35FIN) to EPSG:2393 (YKJ) for the growth model.
- One growth trajectory is computed by `Growth_and_Yield_Table`, covering all
  three species slots together (unlike `metsakeskus_to_allometry.py`, which
  isolates the dominant and subdominant species into separate files — see
  [issue #276](https://github.com/Susi-pack/susi_26/issues/276)).

## Dry runs

`--dry-run` stops the run immediately before the writing stage. Reading and
filtering still happen in full — every stand is parsed and the TreeStrata
check still runs and reports its skips — since that is precisely what a dry
run exists to show. The tool then prints the files it would have written, and
exits.

Nothing at all is created — no CSVs, no `stand_data.json`, and not even
the output folder, so a dry run does not claim a `--project-dir` that the
real run then has to work around. The one thing it does still enforce is the
refusal to run into an existing output folder: whether the real run could
start is part of what a dry run is for.

The counts are an **upper bound**: computing the growth trajectory is the
slow part of a run and the part `--dry-run` skips, so a stand that would fail
inside the growth model or the coordinate transform is still counted here.
The TreeStrata skip is reported in full, because that stage did run.

## Output files

| File | Written |
|---|---|
| `allometry/{stand_id}.csv` | Always, one per stand that survives filtering. |
| `stand_data.json` | Always, in the project's `inputs/`. One `StandData` entry per converted stand — fertility class, allometry file, YKJ coordinates, polygon, stand-level and per-species metadata. Read by `build_stand_params` and by `new_growth_allometry.py`'s sourced mode. |

Each CSV is the allometric road map follows the canonical allometry schema, i.e., the columns declared in `susi.core.allometry_columns.ALLOMETRY_COLUMNS`.

## Skipped stands

A stand that fails a check is reported with its id and a reason, and the run
continues. One bad stand never aborts the others.

| Reason | Meaning |
|---|---|
| `Stand <id> has no TreeStrata` | The stand's `TreeStandDataDate` has no `TreeStrata` container at all. |

Every other piece of missing or malformed required data — a missing
`TreeStandData`, `TreeStandDataDate`, `TreeStandSummary`, or `TreeStratum`
list once a `TreeStrata` container is present — aborts the whole run instead
of being skipped stand-by-stand, since it signals the file itself doesn't
follow the expected structure rather than one ordinary gap in the data.
