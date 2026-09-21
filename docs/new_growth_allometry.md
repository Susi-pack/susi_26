---
icon: lucide/sprout
---


# New-growth allometry file

`src/tools/new_growth_allometry/new_growth_allometry.py` generates the
allometry file for one freshly regenerated stand: the single-species stand
that starts growing, at age 1, immediately after a clear cut. It is meant
for `ClearCut.new_growth_allometry`, not for a stand's initial pre-cut
growth.

This tool processes exactly one stand, and it has two execution modes:

* **Standalone**. Every fact about the stand, including the species and its stem count, is hand-typed into the config.
* **Sourced** (`--stand-data stand_data.json --stand-id <id>`, provide both or
  neither). The following data is read from a project's `stand_data.json`: `altitude`, `ddy`, `fertility_class` and
  the stand's coordinates. The file `stad_data.json` is the document [`xml_to_allometry.py`](how_to_generate_allometry_from_xml.md) and [`metsakeskus_to_allometry.py`](how_to_generate_allometry_from_metsakeskus.md) write alongside their own output. `new_growth_allometry.py` config then must only state what that document does not know: the species, its stem count, and the projection settings.

Both modes produce the same file, from the same code.
They only differ in where the input data/parameters come from.

This page is the reference for the tool's parameters, inputs and outputs.
For a walkthrough of an actual run, see
[How to generate a new-growth allometry file](how_to_generate_new_growth_allometry.md).

## Synopsis

```bash
python src/tools/new_growth_allometry/new_growth_allometry.py \
    --project-dir PROJECT_DIR [--config CONFIG.toml] \
    [--stand-data STAND_DATA.json --stand-id STAND_ID] \
    [--allow-out-of-range-values] [--dry-run]
```

## Command-line arguments

| Argument | Required | Description |
|---|---|---|
| `--project-dir` | yes | Path to the project's folder. Decides where the output goes — `<project-dir>/allometry/` — and where the config file is looked up by default. |
| `--config` | no | Path to the TOML config file (see below). Defaults to `<project-dir>/new_growth_config.toml`. Must exist and end in `.toml`. |
| `--stand-data` | no, but required with `--stand-id` | Path to a project's `stand_data.json`. Selects [sourced mode](#sourced-mode). Must exist and end in `.json`. |
| `--stand-id` | no, but required with `--stand-data` | Which stand of `--stand-data` to read the site values from. |
| `--allow-out-of-range-values` | no | Downgrade an out-of-range `altitude`/`ddy`, and a standalone run's converted `x_ykj`/`y_ykj`, from an error to a warning. `NaN` altitude/ddy is rejected either way, and so is an out-of-range coordinate in a stand-data document — see [Enforced ranges](#enforced-ranges). |
| `--dry-run` | no | Report what the run would produce and exit, writing nothing at all. See [Dry runs](#dry-runs). |

`--stand-data` and `--stand-id` must be given **together**: either one alone
is an error, since a document with no stand named, or a stand name with no
document to look it up in, means nothing. Give neither and the tool runs in
standalone mode.

There is no positional input file. In standalone mode the config **is** the
input; in sourced mode the config plus `--stand-data` are. The output
folder is not selectable either: `<project-dir>/allometry/` is the only
place this tool writes, and it must **not** already exist — the tool refuses
to run into a previous run's output rather than overwrite it, so a repeat
run needs a different `--project-dir`, or a fresh `allometry/` folder
underneath the existing one.

The tool prints the fully-resolved (absolute) path it read the config from
and the path it writes to, along with every value it actually used —
including ones not literally in the config file, like the converted YKJ
coordinates — so nothing about the run is left ambiguous or hidden.

## Config file

A TOML file. By default the tool looks for `new_growth_config.toml`
directly inside `--project-dir`; pass `--config` to use a different name or
location.

That name is this tool's own, not a shared `config.toml`:
`xml_to_allometry.py` and `metsakeskus_to_allometry.py` default to
`config.toml`, so all three tools' configs can sit in one project folder
without colliding. A `config.toml` left there by another tool is ignored —
this one looks only for `new_growth_config.toml`.

Copy
[`src/tools/new_growth_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/new_growth_allometry/default_config.toml)
and edit it: the optional fields are listed with the values the tool applies
when they are absent, while the required ones carry deliberately invalid
placeholders for you to replace. Unknown fields are rejected, and every
missing required field is reported at once.

The same template covers both modes — every field in it is marked either
standalone-only or required in both. Which mode you are in is decided by the
command line, not by the file.

### Required, standalone mode only

These five have no equivalent in a sourced run: it reads them from the
stand-data document instead. In sourced mode they must be **deleted** from
the config — unknown fields are rejected, so leaving one behind is an
error rather than a silently ignored, possibly stale value.

| Field | Unit | Enforced range | Description | Sourced mode reads instead |
|---|---|---|---|---|
| `altitude` | m above sea level | 0 – 1000 | Site elevation. | the document's root `altitude` |
| `ddy` | degree days per year | 500 – 2000 | Temperature sum. | the document's root `ddy` |
| `fertility_class` | — | — | Fertility class of the site. | `stands[<stand-id>].site_fertility_class` |
| `x`, `y` | m (ETRS-TM35FIN, EPSG:3067) | — | Stand coordinates. Converted internally to YKJ (EPSG:2393) for the growth model, the same conversion `xml_to_allometry.py`/`metsakeskus_to_allometry.py` apply to their own coordinates. | `stands[<stand-id>].x_ykj`/`.y_ykj`, which are **already** YKJ — see [Sourced mode](#sourced-mode) |

### Required, both modes

A stand-data document says nothing about these, so they are hand-written
whichever mode you run in.

| Field | Unit | Enforced range | Description |
|---|---|---|---|
| `species` | — | one of `"pine"`, `"spruce"`, `"birch"` | The one species growing in this new-growth stand. See [Why only one species?](#why-only-one-species). |
| `stems_count` | stems/ha | > 0 | Stem count of that species. |

### Enforced ranges

| Value | Range | Units |
|---|---|---|
| `altitude` | 0 – 1000 | m above sea level |
| `ddy` | 500 – 2000 | degree days per year |
| `x_ykj` | 200 – 550 | YKJ grid easting, 10 km units |
| `y_ykj` | 6500 – 7900 | YKJ grid northing, 1 km units |

All four are deliberately wider than Finnish forest land, to catch a
mistyped value (metres vs feet, say, or a coordinate that never went
through the EPSG:3067 → YKJ conversion) without rejecting real ones. A
value outside its range blocks the run unless you pass
`--allow-out-of-range-values`; a `NaN` `altitude`/`ddy` is rejected even
then.

`x_ykj`/`y_ykj` are checked against the **converted** YKJ coordinates, and
there is exactly one definition of that range: `X_YKJ_MIN/MAX` and
`Y_YKJ_MIN/MAX` in
`src/tools/shared_allometry_tool_utils/shared_utils.py`, which sit next to
the `point_to_ykj` call that produces coordinates in those units.
`StandData.x_ykj`/`.y_ykj` take their field bounds from the same constants,
so a coordinate is held to the same range however it reaches the tool. What
differs between the modes is only *when* a bad one is caught:

* **Standalone.** The config's `x`/`y` are converted, then range-checked at
  the CLI. `--allow-out-of-range-values` downgrades a violation to a
  warning, the same as for `altitude`/`ddy`.
* **Sourced.** The coordinates are validated while the stand-data document
  is being read, so a bad one is rejected before the CLI check is reached.
  `--allow-out-of-range-values` does **not** apply there — a document that
  fails its own validation is simply unreadable — and the run stops with an
  error naming the file and the offending field.

`species`/`stems_count` have no range and no override: an invalid species is
rejected outright, since `pine`/`spruce`/`birch` is this tool's entire
vocabulary, and a non-positive `stems_count` would mean nothing is actually
growing.

### Optional, both modes

| Field | Default | Description |
|---|---|---|
| `n_trees` | `20` | Reference trees in the growth model. |
| `start_year` | `5` | First projected step, in years after establishment. |
| `end_year` | `80` | Last projected step, in years after that. |
| `step_years` | `5` | Interval between projected steps. |

These four are passed straight to `Growth_and_Yield_Table`, and set how far
forward the stand's growth is projected and at what resolution. A stand-data
document has nothing to say about them either, so they stay in the config in
sourced mode too.

The stand's **starting height** (per species: 0.1 m for pine, 0.2 m for
spruce and birch) is not configurable — it is a fixed model parameter in
`new_growth_allometry.py` (`STARTING_HEIGHT`), not something to tune per
project.

## Sourced mode

Pass `--stand-data <stand_data.json> --stand-id <id>` and the tool takes
its five site values from that document instead of the config:

| Value | Read from |
|---|---|
| `altitude` | the document's root `altitude` — it is project-global, one value for every stand in the document |
| `ddy` | the document's root `ddy`, likewise project-global |
| `fertility_class` | `stands[<stand-id>].site_fertility_class` |
| `x_ykj` | `stands[<stand-id>].x_ykj` |
| `y_ykj` | `stands[<stand-id>].y_ykj` |

Note the coordinates: **sourced mode performs no `point_to_ykj`
conversion.** `StandData.x_ykj`/`.y_ykj` are already YKJ grid units,
converted once by whichever tool wrote the document, and are used as-is.
This is unlike standalone mode, whose `x`/`y` are ETRS-TM35FIN (EPSG:3067)
metres that the tool converts itself. The two are not interchangeable — the
same numbers mean completely different places.

A `--stand-id` that is not in the document is an error, and the message
lists the ids the document does hold. A document that fails its own
validation — a coordinate outside the [shared YKJ
range](#enforced-ranges), a missing required field — is likewise a plain
error naming the file and the offending field, not a traceback.

Nothing else about the stand is read: not its existing allometry files, not
its area, not any of the document's other bookkeeping fields. Sourced mode
saves you from re-typing five numbers; it does not change what the tool
produces.

## What the tool does

There is no filtering stage — this tool processes exactly one stand, so
there is nothing to filter. The run is printed in two sections instead of
the usual three: reading, then writing.

**Reading.** Prints every config value, plus the values computed from it:
the YKJ `x`/`y` and the starting height that applies to the configured
species. In sourced mode it also prints the stand-data path, the stand id,
and, next to each sourced value, where it came from — with no informational
JSON for this tool, this is the only place that is ever shown. Nothing the
run actually uses is left unprinted.

**Writing.** The configured species is placed into its own species slot
(pine → slot 1, spruce → slot 2, birch → slot 3 — the same positional
convention `xml_to_allometry.py`/`metsakeskus_to_allometry.py` use), age 1,
zero diameter, zero basal area, and its starting height. The other two
slots are left at zero. One growth trajectory is computed by
`Growth_and_Yield_Table`, tagged with the configured species' own code —
there is no dominant/subdominant ranking or tie-break to perform, since
only one species is ever non-zero.

### Why only one species?

`xml_to_allometry.py` accepts a stand with up to three species strata and
pools every non-zero one into a single combined growth table, tagged only
with whichever species carries the most basal area — the resulting file can
silently represent a mixed stand under one species' label.
`new_growth_allometry.py` deliberately does not repeat that: since the
config already states the species directly (there is no inventory row to
extract several from), the tool enforces that exactly one species is ever
modeled.

## Dry runs

`--dry-run` stops the run immediately before the writing stage. There is no
filtering stage to run either way. The config, and in sourced mode the
stand-data document, are read and validated first, so a dry run still
catches a mistyped `--stand-id` or an out-of-range value. The tool prints the file it would have
written, and exits.

Nothing at all is created — no CSV, and not even the output folder — so a
dry run does not claim a `--project-dir` that the real run then has to work
around. The one thing it does still enforce is the refusal to run into an
existing output folder: whether the real run could start is part of what a
dry run is for.

## Output files

| File | Written |
|---|---|
| `new_growth_<species>.csv` | Always. `<species>` is exactly the configured `species` value, e.g. `new_growth_pine.csv`. |

There is no informational JSON dump for this tool, unlike
`xml_to_allometry.py`'s or `metsakeskus_to_allometry.py`'s `stand_data.json`:
there is no extraction step here to audit. Standalone mode's complete input
is the config file; sourced mode's is the config file plus the one named
stand of an existing `stand_data.json` — and either way the Reading section
prints every value used, including which document and stand each sourced
one came from.

The CSV follows the canonical allometry schema — the columns declared in
`susi.core.allometry_columns.ALLOMETRY_COLUMNS` — and is read directly by
`read_allometry_info_from_csv`. Its first row has `Age` = 1, satisfying
`ClearCut.new_allometry_includes_age_one`.
