---
icon: lucide/map
---

# How to generate allometry files from XML data

`xml_to_allometry.py` turns each stand in a Finnish national forest XML export (metsätietostandardit) into the allometry file SUSI reads.

This guide walks through that process, from placing the XML file to checking the output allometry files.
For every flag, config field and skip reason mentioned below, see the [reference page](xml_to_allometry.md).

To generate the same kind of file from a Metsäkeskus GeoPackage instead, see
[How to generate allometry files from Metsäkeskus data](how_to_generate_allometry_from_metsakeskus.md).

## 1. Obtain and place the XML file

The tool reads an XML file following the Finnish national forest information standards.

!!! info "shared XML standard"
    This XML format is considered a forestry standard in Finland.
    The Metsäkeskus data, however, lack this standard.
    For those, we have the separate tool `metsakeskus_to_allometry.py`.

    That tool already produces the allometry files required by SUSI.
    But if for any reason you also need the XML files, you can get them by using the `--emit-xml` flag, see its [reference page](metsakeskus_to_allometry.md#output-files).

We recommend you place it inside the root-level `inputs/` folder, in a subfolder named after your project.
That way, it won't be tracked by git, and it will be located in a logical place.
(For more information on this, see [How to handle input data](how_to_handle_input_data.md)).

```
inputs/
└── my_project/
    └── stands.xml
```

## 2. Write the config file

That same folder doubles as the **project** directory: the CLI argument `--project-dir` points there. In our case, this would be:

```
inputs/my_project/
```

Next, copy the `.toml` template
[`src/tools/xml_to_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/xml_to_allometry/default_config.toml)
into that folder. Call it `config.toml`:

```
inputs/my_project/config.toml
```

Named and placed this way, the tool finds it automatically from `--project-dir` alone, with no separate `--config` flag needed (see step 3).
`--config` still accepts any path, if you'd rather keep the file elsewhere or
under a different name.

Two fields are required and have no default.
Let's say we pick the following numbers and add them to the config file:

```toml
altitude = 100   # metres above sea level
ddy = 1250       # temperature sum, degree days per year
```

Neither is part of the XML standard, so you supply them yourself, and the same pair applies to every stand in the run.

There are other parameters that modify how far forward each stand's growth is projected and at what resolution; they are optional, and documented in the [reference](xml_to_allometry.md#config-file).

The config file can also turn on [dense young stand scaling](dense_young_stand_scaling.md), which scales down young stands recorded with a very large number of stems before their growth is computed. It is off by default.

## 3. Run the tool

```bash
susi-xml-to-allometry \
    inputs/my_project/stands.xml \ # <-- the XML file from step 1
    --project-dir inputs/my_project # <-- your project's folder, from steps 1-2
```

Since `config.toml` lives directly inside `--project-dir`, the tool finds it there automatically.

The output files are saved to `<project-dir>/allometry/`.
In the example above, that's `inputs/my_project/allometry/`.
There is no way to send them anywhere else: `--project-dir` is what decides the folder.
That `allometry/` folder must not already exist: the tool refuses to run into a previous run's output rather than overwrite it.
So rename or move the existing `allometry/` first (and move any `new_growth/` subfolder back into the fresh one afterwards).

!!! tip "Try it with `--dry-run` first"

    Add `--dry-run` to the command above and the tool reads the file, applies the TreeStrata check, and prints exactly what it would produce.
    Then exits without writing anything at all.

    This is worth doing before any real run. Two reasons:

    - A real run creates a folder of allometry files that you may have wanted
      only as a test.
    - It is faster. Reading and filtering is the fast part of a run;
      computing the growth trajectory (what `--dry-run` skips) is what takes
      the time.

    The flip side is that failures inside the growth model or the coordinate
    transform cannot be seen in a dry run, so the file count it reports is an
    upper bound.

## 4. Read the progress report

The run prints what it read, how the TreeStrata check narrowed the stands
down, and what it wrote. For a 21-stand XML file with every stand carrying
tree strata data:

```
Reading
-------
Tool initialized with:
    - xml_file    = /path/to/inputs/my_project/stands.xml
    - project_dir = /path/to/inputs/my_project
    - config      = /path/to/inputs/my_project/config.toml
    - output_dir  = /path/to/inputs/my_project/allometry
    - altitude    = 100.0
    - ddy         = 1250.0

Filtering
---------
1. TreeStrata check -- stands with no recorded tree strata are skipped:
   -> 21 / 21 stands kept

Stands ready for allometry: 21

Writing
-------
Destination folder: /path/to/inputs/my_project/allometry

Assuming all sites are peatland sites!
Allometric road map successfully generated for stand 1
...
Allometry files written: 21 stand(s) -- 21 CSV(s)
Informational JSON written: /path/to/inputs/my_project/allometry/extra_xml_info.json
```

If the file holds a dense young stand, the Filtering section says so after the "Stands ready for allometry" line.
With the option off (the default), it is a warning naming the stand:

```
Warning: 1 dense young stand(s): 20
  These are young stands with more stems than the default limits of dense young stand scaling, and they will be grown that way. [dense_young_stand_scaling] in the config file scales such stands down: https://susi-pack.github.io/susi_26/dense_young_stand_scaling/
```

With `enabled = true` in the `[dense_young_stand_scaling]` table, it is one line per scaled stand instead:

```
Dense young stand scaling -- 1 stand(s) scaled down before the growth model runs:
  20: 2809 -> 1800 stems/ha (scaling factor 0.641)
```

See [Dense young stand scaling](dense_young_stand_scaling.md) for what both mean.

Those 21 stands produced 21 CSV files, one per stand.

```
inputs/my_project/allometry/
├── 1.csv
├── 2.csv
├── ...
└── extra_xml_info.json
```

The JSON alongside them records what was extracted for every converted
stand; it is the place to look when you want to know why a stand came out
the way it did.

??? question "What if a stand is missing from the output?"

    A stand with no `TreeStrata` container at all is skipped rather than aborting the run, and reported with its id.
    Look for a line under "Skipped (no TreeStrata)" in the printed Filtering section.

    Any other structural problem (a missing `TreeStandSummary`, a missing required `StandBasicData` field, etc.) aborts the whole run instead, since it signals the file itself doesn't follow the expected structure.

## Next steps

You now have allometry files in the layout `SusiParams` expects: a folder of CSVs, one per stand.
Point SUSI at them through `SusiParams`: `allometry_dir_path` is that same folder.
See [Building a `SusiParams` instance from your own data](how_to_handle_input_data.md#building-a-susiparams-instance-from-your-own-data).
