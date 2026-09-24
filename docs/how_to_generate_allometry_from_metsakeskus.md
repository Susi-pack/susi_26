---
icon: lucide/map
---

# How to generate allometry files from Metsäkeskus data

Metsäkeskus publishes some its forest inventory as open data.
`metsakeskus_to_allometry.py` turns each stand in that dataset into the allometry file SUSI reads.

This guide walks through that process, from downloading the data to checking the output allometry files.
For every flag, config field and filter rule mentioned below, see the [reference page](metsakeskus_to_allometry.md).

To generate the same kind of file from a Finnish forestry XML stand export instead, see
[How to generate allometry files from XML data](how_to_generate_allometry_from_xml.md).

## 1. Download the inventory data

Metsäkeskus publishes one GeoPackage per region (*maakunta*), zipped:

```
https://avoin.metsakeskus.fi/aineistot/Metsavarakuviot/Maakunta/MV_Uusimaa.zip
```

Swap `MV_Uusimaa` for the region you need.

We recommend you unzip it into the root-level `inputs/` folder.
That way, it won't be tracked by git, and it will be located in a logical place.
( For more information on this, see [How to handle input data](how_to_handle_input_data.md)).

```
inputs/
└──  MV_Uusimaa.gpkg
```

!!! warning "These files are large"

    The Uusimaa export is 1.8 GB, covers about 429,000 rows (each row = a site, not all are forests!), and needs a few GB of memory to read. Everything inside `inputs/` is gitignored, so it will not end up in a commit.

## 2. Write the config file

Create a new folder under `inputs/`.
That's where the input data for your SUSI simulations will live.
For lack of a better term, we call this the `project` directory.
The CLI argument `--project-dir` points there.
In our case, this could be:
```
inputs/uusimaa/
```
Next, copy the `.toml` template [`src/tools/metsakeskus_to_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/metsakeskus_to_allometry/default_config.toml) into that folder.
Call it `config.toml`:

```
inputs/uusimaa/config.toml
```

Named and placed this way, the tool finds it automatically from `--project-dir` alone, with no separate `--config` flag needed (see step 3).
`--config` still accepts any path, if you'd rather keep the file elsewhere or under a different name.


Three fields are required and have no default.
We choose the following values in this case:

```toml
target_year = 2018   # which inventory snapshot to use
altitude = 50        # metres above sea level
ddy = 1250           # temperature sum, degree days per year
```

`target_year` picks which measurement of each stand to convert; only stands measured in exactly that year are used.
Altitude and temperature sum are not part of the Metsäkeskus data, so you supply them yourself, and the same pair applies to every stand in the run.

There are other parameters that modify, e.g., which fertility and development classes to keep, how far forward each stand's growth is projected etc., is optional, and documented in the [reference](metsakeskus_to_allometry.md#config-file).


## 3. Run the tool

```bash
python src/tools/metsakeskus_to_allometry/metsakeskus_to_allometry.py \
    inputs/uusimaa/MV_Uusimaa.gpkg \ # <-- .gpkg with Metsäkeskus region data
    --project-dir inputs/uusimaa # <-- your project's folder, from steps 1-2
```

`--config` is left out here: since `config.toml` lives directly inside `--project-dir`, the tool finds it there automatically.

The output files land in `<project-dir>/allometry/`.
In the example above, that's `inputs/uusimaa/allometry/`.
There is no way to send them anywhere else: `--project-dir` is what decides the folder.
That `allometry/` folder must not already exist: the tool refuses to run into a previous run's output rather than overwrite it.
So rename or move the existing `allometry/` first (and move any `new_growth/` subfolder back into the fresh one afterwards).

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
| Read from the `stand` layer | 429,164 |
| 1. Site filter — drained peatland forest, fertility class 2–5 | 13,488 |
| 2. Measured snapshot from 2018 | 917 |
| 3. Development class 1–3 | 804 |
| 4. Structural and species-data checks | 804 |
| 5. Viability — basal area above zero | 786 |

Those 786 stands produced 1,481 CSV files — 786 dominant layers and 695
subdominant ones:

```
inputs/uusimaa/allometry/
├── 13051239_dominant.csv
├── 13051239_subdominant.csv
├── ...
└── extra_gpkg_info.json
```

Each stand gets one file per canopy layer: `_dominant.csv` always, and `_subdominant.csv` when a second species is present.
The JSON alongside them records what was extracted for every converted stand; it is the place to look when you want to know why a stand came out the way it did.

??? question "Why did I get so few stands?"

    Almost the entire drop happens in the first two filters, and neither is a
    sign that anything went wrong.

    The site filter keeps only what SUSI can simulate: drained peatland forest.
    In the run above that alone took 429,164 stands down to 13,488.

    The second filter then keeps only stands measured in exactly `target_year`,
    which excluded a further 12,571. This is the one to reach for. Just before
    applying it, the tool prints every measured inventory year in your data and
    how many stands each one covers.

    `--dry-run` (see step 3) is the cheap way to try another year: it prints
    that same table, and the stand counts each filter leaves behind, without
    writing any files or claiming a `--project-dir`.

    Only 18 stands were lost to data quality here — it is the filters, not the
    state of the data, that decide your yield. Every dropped stand is listed
    with a reason, and the
    [reference](metsakeskus_to_allometry.md#skipped-stands) explains each one.

## Next steps

You now have allometry files in the layout `SusiParams` expects: a folder of CSVs, one per canopy layer.
Pointing SUSI at them means building a `CanopyLayerAllometry` whose `allometry_dir_path` is that folder, registering the stand's files, and mapping them onto the `dominant` and `subdominant` layers.
See [Building a `SusiParams` instance from your own data](how_to_handle_input_data.md#building-a-susiparams-instance-from-your-own-data).
