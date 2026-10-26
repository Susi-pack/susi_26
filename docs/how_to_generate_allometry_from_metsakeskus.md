---
icon: lucide/map
---

# How to generate allometry files from Metsäkeskus data

Metsäkeskus publishes some its forest inventory as open data.
`susi-metsakeskus-to-allometry` turns each stand in that dataset into the allometry file SUSI reads.

The tool reads Metsäkeskus's **2026 data model**, the format the files are downloaded in today.
For an older download, in the pre-2026 format, see the deprecated
[old-format tool](metsakeskus_to_allometry_old_format.md); the current tool recognises such a
file and tells you so.

This guide walks through that process, from downloading the data to checking the output allometry files.
For every flag, config field and filter rule mentioned below, see the [reference page](metsakeskus_to_allometry.md).

To generate the same kind of file from a Finnish forestry XML stand export instead, see
[How to generate allometry files from XML data](how_to_generate_allometry_from_xml.md).

## 1. Download the inventory data

Metsäkeskus publishes one GeoPackage per region (*maakunta*), zipped:

```
https://avoin.metsakeskus.fi/aineistot/Metsavarakuviot/Maakunta/MV_Pirkanmaa.zip
```

Swap `MV_Pirkanmaa` for the region you need.

We recommend you unzip it into the root-level `inputs/` folder.
That way, it won't be tracked by git, and it will be located in a logical place.
( For more information on this, see [How to handle input data](how_to_handle_input_data.md)).

```
inputs/
└──  MV_Pirkanmaa.gpkg
```

!!! warning "These files are large"

    The Pirkanmaa export is 2.9 GB unzipped, covers about 793,000 stands (each one a site, not all are forests!), and needs several GB of memory to read. Everything inside `inputs/` is gitignored, so it will not end up in a commit.

## 2. Write the config file

Create a new folder under `inputs/`.
That's where the input data for your SUSI simulations will live.
For lack of a better term, we call this the `project` directory.
The CLI argument `--project-dir` points there.
In our case, this could be:
```
inputs/pirkanmaa/
```
Next, copy the `.toml` template [`src/tools/metsakeskus_to_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/metsakeskus_to_allometry/default_config.toml) into that folder.
Call it `config.toml`:

```
inputs/pirkanmaa/config.toml
```

Named and placed this way, the tool finds it automatically from `--project-dir` alone, with no separate `--config` flag needed (see step 3).
`--config` still accepts any path, if you'd rather keep the file elsewhere or under a different name.


Four fields are required and have no default.
We choose the following values in this case:

```toml
target_year = 2015                    # which inventory snapshot to use
altitude = 100                        # metres above sea level
ddy = 1300                            # temperature sum, degree days per year
mean_diameter_range_filter = [8, 21]  # keep stands with a mean diameter of 8-21 cm
```

`target_year` picks which inventory of each stand to convert; only stands with an inventory dated in exactly that year are used.
Altitude and temperature sum are not part of the Metsäkeskus data, so you supply them yourself, and the same pair applies to every stand in the run.

`mean_diameter_range_filter` leaves out stands too young or too old to grow into a useful allometry curve.
`[8, 21]` suits Pirkanmaa, but what is too young or too old depends on the region, so there is no default:
the [reference](metsakeskus_to_allometry.md#mean-diameter-range) explains how to choose a range for yours with a dry run.

There are other parameters that modify, e.g., which fertility classes to keep, how far forward each stand's growth is projected etc., is optional, and documented in the [reference](metsakeskus_to_allometry.md#config-file).

The config file can also turn on [dense young stand scaling](dense_young_stand_scaling.md), which scales down young stands recorded with a very large number of stems before their growth is computed. It is off by default.


## 3. Run the tool

```bash
susi-metsakeskus-to-allometry \
    inputs/pirkanmaa/MV_Pirkanmaa.gpkg \ # <-- .gpkg with Metsäkeskus region data
    --project-dir inputs/pirkanmaa # <-- your project's folder, from steps 1-2
```

`--config` is left out here: since `config.toml` lives directly inside `--project-dir`, the tool finds it there automatically.

The output files land in `<project-dir>/allometry/`.
In the example above, that's `inputs/pirkanmaa/allometry/`.
`--project-dir` is what decides the folder; `--allometry-set-name` writes a named set beside it instead, for a project with several exports (e.g. one per region).
Neither that `allometry/` folder nor `stand_data.json` may already exist: the tool refuses to run into a previous run's output rather than overwrite it.
So rename or move the existing `allometry/` and `stand_data.json` first (and move any `new_growth/` subfolder back into the fresh one afterwards).

The tool prints the fully-resolved (absolute) path of everything it reads and writes, so it's never ambiguous what "the output folder" refers to.

!!! tip "Try it with `--dry-run` first"

    Add `--dry-run` to the command above and the tool reads the data, applies every filter, and prints exactly what it would produce.
    Then, exits without writing anything at all.

    This is worth doing before any real run. Two reasons:

    - A real run creates a folder of allometry files that you may have wanted only as a test.
    - It is faster. Reading and filtering is the fast part of a run; computing one growth trajectory per canopy layer (what `--dry-run` skips) is what takes the time.

    The flip side is that failures inside the growth model cannot be seen in a dry run, so the file count it reports is an upper bound.


## 4. Read the progress report

The run prints what it read, how each filter narrowed the stands down, and what it wrote.
For the run above:

| Stage | Stands left |
|---|---|
| Read from the `growthplacedata` layer | 792,854 |
| 1. Site filter — drained peatland forest, fertility class 2–5 | 64,282 |
| 2. Inventory snapshot from 2015 | 634 |
| 3. Structural and species-data checks | 634 |
| 4. Viability — basal area above zero | 163 |
| 5. Mean diameter 8–21 cm | 96 |

Those 96 stands produced 159 CSV files — 96 dominant layers and 63
subdominant ones:

```
inputs/pirkanmaa/allometry/
├── 13051239_dominant.csv
├── 13051239_subdominant.csv
├── ...
└── extra_gpkg_info.json
```

Each stand gets one file per canopy layer: `_dominant.csv` always, and `_subdominant.csv` when a second species is present.
The JSON alongside them records what was extracted for every converted stand; it is the place to look when you want to know why a stand came out the way it did.

Just before step 5, the run also prints the mean diameter of the stands for each development class.
That table is what to look at when choosing `mean_diameter_range_filter`; the
[reference](metsakeskus_to_allometry.md#mean-diameter-range) explains it.

If the data holds dense young stands, the Filtering section says so after the "Stands ready for allometry" line.
With a minimum mean diameter of 8 cm, as above, there are none left by then: dense young stands are all thinner than that.
With a lower minimum, and the option off (the default), it is a warning naming the stands. For example, from an Uusimaa run:

```
Warning: 12 dense young stand(s): 32175706, 32175707, 32175794, 32176520, 32176550, 32177051, 32177293, 32344874, 32484519, 32487471, 32559296, 32559664
  These are young stands with more stems than the default limits of dense young stand scaling, and they will be grown that way. [dense_young_stand_scaling] in the config file scales such stands down: https://susi-pack.github.io/susi_26/dense_young_stand_scaling/
```

With `enabled = true` in the `[dense_young_stand_scaling]` table, it is one line per scaled stand instead:

```
Dense young stand scaling -- 12 stand(s) scaled down before the growth model runs:
  32175706: 5398 -> 2000 stems/ha (scaling factor 0.371)
  32175707: 2684 -> 2000 stems/ha (scaling factor 0.745)
  32175794: 2668 -> 1800 stems/ha (scaling factor 0.675)
  ...
```

See [Dense young stand scaling](dense_young_stand_scaling.md) for what both mean, and its [Metsäkeskus section](dense_young_stand_scaling.md#with-metsakeskus-data) for what is particular to this tool.

??? question "Why did I get so few stands?"

    Almost the entire drop happens in the first two filters, and neither is a
    sign that anything went wrong.

    The site filter keeps only what SUSI can simulate: drained peatland forest.
    In the run above that alone took 792,854 stands down to 64,282.

    The second filter then keeps only stands with an inventory dated in exactly
    `target_year`, which excluded a further 63,648. This is the one to reach
    for. Just before applying it, the tool prints every inventory year in your
    data, how many stands each one covers, and from which source: field
    measurement, remote sensing, and so on. In Pirkanmaa, the years up to 2019
    each cover a few hundred stands, mostly measured in the field; from 2020 on,
    tens of thousands, nearly all remote-sensed.

    `--dry-run` (see step 3) is the cheap way to try another year: it prints
    that same table, and the stand counts each filter leaves behind, without
    writing any files or claiming a `--project-dir`.

    The viability check is the next largest drop here: 471 of the 634 are
    seedling stands, whose inventory records stems but no basal area, so there
    is nothing to grow. Every dropped stand is listed with a reason, and the
    [reference](metsakeskus_to_allometry.md#skipped-stands) explains each one.

## Next steps

You now have allometry files in the layout `SusiParams` expects: a folder of CSVs, one per canopy layer.
Pointing SUSI at them means building a `CanopyLayerAllometry` whose `allometry_dir_path` is that folder, registering the stand's files, and mapping them onto the `dominant` and `subdominant` layers.
See [Building a `SusiParams` instance from your own data](how_to_handle_input_data.md#building-a-susiparams-instance-from-your-own-data).
