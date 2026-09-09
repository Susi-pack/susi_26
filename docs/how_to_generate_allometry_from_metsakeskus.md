---
icon: lucide/map
---

# How to generate allometry files from Metsäkeskus data

Metsäkeskus, the Finnish Forest Centre, publishes its forest inventory as open
data: every mapped stand in Finland, with its site attributes and its trees.
`metsakeskus_to_allometry.py` turns that into the allometry files SUSI reads,
for every stand in the export that SUSI can actually simulate.

This guide walks through one run, from downloading the data to checking what
came out. For every flag, config field and filter rule, see the
[reference page](metsakeskus_to_allometry.md).

## 1. Download the inventory data

Metsäkeskus publishes one GeoPackage per region (*maakunta*), zipped:

```
https://avoin.metsakeskus.fi/aineistot/Metsavarakuviot/Maakunta/MV_Uusimaa.zip
```

Swap `MV_Uusimaa` for the region you need.

Unzip it into your project folder, the same way as the rest of your input data
(see [How to handle input data](how_to_handle_input_data.md)):

```
inputs/
└── uusimaa/
    └── MV_Uusimaa.gpkg
```

!!! warning "These files are big"

    The Uusimaa export is 1.8 GB, covers about 429,000 stands, and needs a few
    GB of memory to read. Everything inside `inputs/` is gitignored, so it will
    not end up in a commit.

## 2. Write the config file

Copy the template,
[`src/tools/metsakeskus_to_allometry.default.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/metsakeskus_to_allometry.default.toml),
next to the data it describes:

```
inputs/uusimaa/metsakeskus.toml
```

Three fields are required and have no default:

```toml
target_year = 2018   # which inventory snapshot to use
altitude = 50        # metres above sea level
ddy = 1250           # temperature sum, degree days per year
```

`target_year` picks which measurement of each stand to convert; only stands
measured in exactly that year are used. Altitude and temperature sum are not
part of the Metsäkeskus data, so you supply them yourself, and the same pair
applies to every stand in the run.

Everything else — which fertility and development classes to keep, and how far
forward each stand's growth is projected — is optional, and documented in the
[reference](metsakeskus_to_allometry.md#config-file).

!!! tip "Don't know your region's altitude and temperature sum?"

    `src/scripts/metsakeskus.py` carries a rough value per region: Uusimaa 50 m
    and 1250 degree days, Lappi 120 m and 800, and so on. Good enough to get a
    first run out; replace them with your own site's figures when you have them.

`--config` accepts any path, so keeping the file next to the data is only a
convention.

## 3. Run the tool

```bash
python src/tools/metsakeskus_to_allometry.py \
    inputs/uusimaa/MV_Uusimaa.gpkg \
    --config inputs/uusimaa/metsakeskus.toml \
    --project-name uusimaa
```

The files land in `inputs/<project-name>/allometry/` — here
`inputs/uusimaa/allometry/`. Pass a second positional argument if you want them
somewhere else. Either way the folder must not already exist: the tool refuses
to run into a previous run's output rather than overwrite it, so a repeat run
needs a new `--project-name` (or a new output folder).

Expect the writing to dominate the runtime. In the Uusimaa run below, reading
and filtering the whole 1.8 GB export took under a minute, and writing the
files took about twenty more — one growth trajectory per canopy layer, computed
and written one file at a time.

## 4. Read the progress report

The run prints what it read, how each filter narrowed the stands down, and what
it wrote. For the run above:

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

Each stand gets one file per canopy layer: `_dominant.csv` always, and
`_subdominant.csv` when a second species is genuinely present. The JSON
alongside them records what was extracted for every converted stand; nothing in
SUSI reads it, but it is the place to look when you want to know why a stand
came out the way it did.

??? question "Why did I get so few stands?"

    Almost the entire drop happens in the first two filters, and neither is a
    sign that anything went wrong.

    The site filter keeps only what SUSI can simulate: drained peatland forest.
    In the run above that alone took 429,164 stands down to 13,488.

    The second filter then keeps only stands measured in exactly `target_year`,
    which excluded a further 12,571. This is the one to reach for. Just before
    applying it, the tool prints every measured inventory year in your data and
    how many stands each one covers. Pick a `target_year` from that table: the
    Uusimaa data spans 1990 to 2025, and 2018 is nowhere near its richest year —
    2023 covers 2,995 of these stands, against 917 for 2018.

    Only 18 stands were lost to data quality here — it is the filters, not the
    state of the data, that decide your yield. Every dropped stand is listed
    with a reason, and the
    [reference](metsakeskus_to_allometry.md#skipped-stands) explains each one.

## Next steps

You now have allometry files in the layout `CanopyLayerAllometry` expects: a
folder of CSVs, one per canopy layer. Pointing SUSI at them means building a
`CanopyLayerAllometry` whose `allometry_dir_path` is that folder, registering
the stand's files, and mapping them onto the `dominant` and `subdominant`
layers — see
[Building a `SusiParams` instance from your own data](how_to_handle_input_data.md#building-a-susiparams-instance-from-your-own-data).
