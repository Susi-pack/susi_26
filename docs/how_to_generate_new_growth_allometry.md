---
icon: lucide/sprout
---

# How to generate a new-growth allometry file

`new_growth_allometry.py` generates the allometry file for one freshly
regenerated (a.k.a. "new growth") stand: a single species, at age 1, immediately
after a clear cut.
This is the allometry file that `SusiParams.ClearCut.new_growth_allometry` expects.

This guide walks through a run, from writing the config to checking the
output file. For every flag, config field and output-file detail mentioned
below, see the [reference page](new_growth_allometry.md).

## 1. Write the config file

The tool has two modes, and step 1 is where they differ. Pick one before you
start:

* **Standalone** — everything about the new-growth stand, including its
  species and stem count, is written directly into the config. There is no
  input data file to place first, unlike the other two allometry tools.
* **Sourced** — you already have a project's `stand_data.json`, written by
  [`xml_to_allometry.py`](how_to_generate_allometry_from_xml.md) or
  [`metsakeskus_to_allometry.py`](how_to_generate_allometry_from_metsakeskus.md)
  into their own `<project-dir>/allometry/` folder. That document supplies
  the five site values, so the config is shorter.

Sourced mode is the one to use when the clear cut is happening in a stand for which you already have a `stand_data.json`:
it keeps the new growth's altitude, temperature sum, fertility class and location exactly consistent with the pre-cut stand, instead of re-typing them and hoping they match.

Pick (or create) a project folder. The CLI argument `--project-dir` points
there; for example:

```
inputs/my_project/
```

Next, copy the `.toml` template
[`src/tools/new_growth_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/new_growth_allometry/default_config.toml)
into that folder. Call it `new_growth_config.toml`:

```
inputs/my_project/new_growth_config.toml
```

Named and placed this way, the tool finds it automatically from
`--project-dir` alone, with no separate `--config` flag needed (see step 2).
`--config` still accepts any path, if you'd rather keep the file elsewhere
or under a different name.

!!! note "The name matters"
    This tool looks for `new_growth_config.toml`, while`xml_to_allometry.py` and `metsakeskus_to_allometry.py` look for `config.toml`. That way all three can share one project folder.

### Standalone mode: seven required fields

Seven fields are required and have no default. Say the clear cut area is
regenerating naturally to pine, with an expected 2,000 stems/ha:

```toml
altitude = 100          # metres above sea level
ddy = 1250               # temperature sum, degree days per year
fertility_class = 3
x = 379930.3             # ETRS-TM35FIN (EPSG:3067) metres
y = 7039150.8            # ETRS-TM35FIN (EPSG:3067) metres
species = "pine"         # exactly one of "pine", "spruce", "birch"
stems_count = 2000       # stems/ha
```

### Sourced mode: two required fields

In sourced mode only two fields are required, and the other five must be **deleted** from your copy of the template config.
Leaving `altitude = 100` behind is an error rather than a silently ignored value.
Working example:

```toml
species = "pine"         # exactly one of "pine", "spruce", "birch"
stems_count = 2000       # stems/ha
```

`altitude`, `ddy`, `fertility_class` and the coordinates come from
`stand_data.json` instead (see step 2 for where that file is and how to
point at it).

`species` is deliberately singular: this tool models one species growing
alone, not a mix — see
[Why only one species?](new_growth_allometry.md#why-only-one-species) for
why.

There are other parameters that modify how far forward the growth is
projected and at what resolution; they are optional, and documented in the
[reference](new_growth_allometry.md#config-file).

## 2. Run the tool

### Standalone mode

```bash
python src/tools/new_growth_allometry/new_growth_allometry.py \
    --project-dir inputs/my_project # <-- your project's folder, from step 1
```

Since `new_growth_config.toml` lives directly inside `--project-dir`, the
tool finds it there automatically.

### Sourced mode

Add `--stand-data` and `--stand-id`. The `stand_data.json` is the one
`xml_to_allometry.py`/`metsakeskus_to_allometry.py` wrote into *their*
output folder, and `--stand-id` is the id of the stand being clear cut —
the same key those tools used, which you can read straight out of the JSON:

```bash
python src/tools/new_growth_allometry/new_growth_allometry.py \
    --project-dir inputs/my_project \
    --stand-data inputs/my_earlier_project/allometry/stand_data.json \
    --stand-id 12345
```

The two flags go together: passing one without the other is an error, not a
half-sourced run. Pass a `--stand-id` the document does not contain and the
tool says so, listing the ids it does hold.

Note that `--project-dir` still points at *this* run's folder — the one
holding the short config from step 1, and the one the new-growth CSV is
written into. It is unrelated to wherever the `stand_data.json` came from,
and the two are usually different folders.

The output file is saved to `<project-dir>/allometry/`. In the example
above, that's `inputs/my_project/allometry/`. There is no way to send it
anywhere else: `--project-dir` is what decides the folder. That `allometry/`
folder must not already exist: the tool refuses to run into a previous
run's output rather than overwrite it, so a repeat run needs a new
`--project-dir` (or a fresh `allometry/` folder underneath the existing
one).

!!! tip "Try it with `--dry-run` first"

    Add `--dry-run` to the command above and the tool reads the config and
    prints exactly what it would produce, then exits without writing
    anything at all.

    This is worth doing before any real run, for the same reason it's worth
    doing for the other two allometry tools: a real run creates a folder you
    may have wanted only as a test, and it's a fast way to check that
    `--project-dir` and the config are set up the way you expect.

## 3. Read the progress report

The run prints what it read — including values computed from the config,
like the converted YKJ coordinates and the starting height that applies to
your chosen species — and what it wrote. For the standalone config above:

```
Reading
-------
Tool initialized with:
    - project_dir     = /path/to/inputs/my_project
    - config          = /path/to/inputs/my_project/new_growth_config.toml
    - output_dir      = /path/to/inputs/my_project/allometry
    - species         = pine
    - stems_count     = 2000 stems/ha
    - fertility_class = 3
    - altitude        = 100.0
    - ddy             = 1250.0
    - x, y (input)    = 379930.3, 7039150.8 (ETRS-TM35FIN metres)
    - x, y (YKJ)      = 338, 7042
    - starting_height = 0.1 m (fixed model parameter)

Writing
-------
Destination folder: /path/to/inputs/my_project/allometry

Assuming this site is a peatland site!
New-growth allometry file written: /path/to/inputs/my_project/allometry/new_growth_pine.csv
```

A sourced run prints the same two sections, but names the origin of every
value it did not get from the config — so a wrong `--stand-id` is visible
here rather than only in the numbers:

```
Reading
-------
Tool initialized with:
    - project_dir     = /path/to/inputs/my_project
    - config          = /path/to/inputs/my_project/new_growth_config.toml
    - output_dir      = /path/to/inputs/my_project/allometry
    - species         = pine
    - stems_count     = 2000 stems/ha
    - stand_data      = /path/to/inputs/my_earlier_project/allometry/stand_data.json
    - stand_id        = 12345
    - fertility_class = 3 (from stand)
    - altitude        = 100.0 (from document root)
    - ddy             = 1250.0 (from document root)
    - x, y (YKJ)      = 338, 7042 (from stand, already YKJ -- no conversion)
    - starting_height = 0.1 m (fixed model parameter)
```

`altitude` and `ddy` say "from document root" because they are
project-global in a `stand_data.json`: one value shared by every stand in
it. `fertility_class` and the coordinates belong to the one stand you named.
And there is no `x, y (input)` line, because there is nothing to convert —
`x_ykj`/`y_ykj` are stored as YKJ already.

There is no "Filtering" section in either mode, unlike the other two
allometry tools: a single stand has nothing to filter.

```
inputs/my_project/allometry/
└── new_growth_pine.csv
```

There is also no informational JSON alongside it — unlike
`xml_to_allometry.py`/`metsakeskus_to_allometry.py`, there is no extraction
step here to audit. Everything the run used is already in the config file
(plus, in sourced mode, the stand-data document it named) and the Reading
section above.

## Next steps

You now have the allometry file `ClearCut.new_growth_allometry` expects for
one canopy layer. Register it in a `CanopyLayerAllometry`'s
`allometry_file_registry` and point the relevant layer's `pointers` at it.
See [Building a `SusiParams` instance from your own data](how_to_handle_input_data.md#building-a-susiparams-instance-from-your-own-data).
