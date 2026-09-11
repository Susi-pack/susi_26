---
icon: lucide/map
---

# How to generate allometry files from XML data

`xml_to_allometry.py` turns each stand in a Finnish national forest XML export
(metsätietostandardit) into the allometry file SUSI reads.

This guide walks through that process, from placing the XML file to checking
the output allometry files.
For every flag, config field and skip reason mentioned below, see the
[reference page](xml_to_allometry.md).

To generate the same kind of file from a Metsäkeskus GeoPackage instead, see
[How to generate allometry files from Metsäkeskus data](how_to_generate_allometry_from_metsakeskus.md).

## 1. Obtain and place the XML file

The tool reads a `ForestPropertyData` XML file following the Finnish national
forest information standards. You typically already have one of these from a
forest-management plan export; alternatively, `metsakeskus_to_allometry.py`
can produce one for you from a Metsäkeskus GeoPackage with its `--emit-xml`
flag — see its [reference page](metsakeskus_to_allometry.md#output-files).

We recommend you place it inside the root-level `inputs/` folder, in a
subfolder named after your project. That way, it won't be tracked by git, and
it will be located in a logical place. (For more information on this, see
[How to handle input data](how_to_handle_input_data.md)).

```
inputs/
└── my_project/
    └── stands.xml
```

## 2. Write the config file

That same folder doubles as the **project** directory: the CLI argument
`--project-dir` points there. In our case, this could be:

```
inputs/my_project/
```

Next, copy the `.toml` template
[`src/tools/xml_to_allometry/default_config.toml`](https://github.com/Susi-pack/susi_26/blob/main/src/tools/xml_to_allometry/default_config.toml)
into that folder. Call it `config.toml`:

```
inputs/my_project/config.toml
```

Named and placed this way, the tool finds it automatically from
`--project-dir` alone, with no separate `--config` flag needed (see step 3).
`--config` still accepts any path, if you'd rather keep the file elsewhere or
under a different name.

Two fields are required and have no default. We choose the following values
in this case:

```toml
altitude = 100   # metres above sea level
ddy = 1250       # temperature sum, degree days per year
```

Neither is part of the XML standard, so you supply them yourself, and the
same pair applies to every stand in the run.

There are other parameters that modify how far forward each stand's growth is
projected and at what resolution; they are optional, and documented in the
[reference](xml_to_allometry.md#config-file).

## 3. Run the tool

```bash
python src/tools/xml_to_allometry/xml_to_allometry.py \
    inputs/my_project/stands.xml \ # <-- the XML file from step 1
    --project-dir inputs/my_project # <-- your project's folder, from steps 1-2
```

`--config` is left out here: since `config.toml` lives directly inside
`--project-dir`, the tool finds it there automatically.

The output files land in `<project-dir>/allometry/`. In the example above,
that's `inputs/my_project/allometry/`. There is no way to send them anywhere
else: `--project-dir` is what decides the folder. That `allometry/` folder
must not already exist: the tool refuses to run into a previous run's output
rather than overwrite it, so a repeat run needs a new `--project-dir` (or a
fresh `allometry/` folder underneath the existing one).

The tool prints the fully-resolved (absolute) path of everything it reads and
writes, so it's never ambiguous what "the output folder" refers to.

!!! tip "Try it with `--dry-run` first"

    Add `--dry-run` to the command above and the tool reads the file, applies
    the TreeStrata check, and prints exactly what it would produce. Then,
    exits without writing anything at all.

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

Those 21 stands produced 21 CSV files, one per stand — unlike
`metsakeskus_to_allometry.py`, this tool writes one file per stand rather than
one per canopy layer (see [issue #276](https://github.com/Susi-pack/susi_26/issues/276)):

```
inputs/my_project/allometry/
├── susi_input_1.csv
├── susi_input_2.csv
├── ...
└── extra_xml_info.json
```

The JSON alongside them records what was extracted for every converted
stand; it is the place to look when you want to know why a stand came out
the way it did.

??? question "What if a stand is missing from the output?"

    A stand with no `TreeStrata` container at all is skipped rather than
    aborting the run, and reported with its id: look for a line under
    "Skipped (no TreeStrata)" in the Filtering section.

    Any other structural problem — a missing `TreeStandSummary`, a missing
    required `StandBasicData` field, and so on — aborts the whole run instead,
    since it signals the file itself doesn't follow the expected structure.
    The [reference](xml_to_allometry.md#skipped-stands) explains the
    distinction.

## Next steps

You now have allometry files in the layout `SusiParams` expects: a folder of
CSVs, one per stand. Pointing SUSI at them means building a
`CanopyLayerAllometry` whose `allometry_dir_path` is that folder and
registering the stand's file. See
[Building a `SusiParams` instance from your own data](how_to_handle_input_data.md#building-a-susiparams-instance-from-your-own-data).
