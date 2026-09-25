"""
Convert the Motti files that PARAMS_PER_SITE references into CSV allometry files.

A Motti file (see "Motti file" in CONTEXT.md) is a legacy .xls workbook with
Finnish headers. Its first sheet (`Puustotunnukset`) is the allometry table and
its second (`Kertymät`) records the species (`id Puulaji`). SUSI reads only the
canonical CSV allometry schema (ALLOMETRY_COLUMNS), so this script, run by hand
once the raw data is in place, writes one CSV per referenced Motti file:

    data/motti/ansa21_A.xls  ->  inputs/allometry/ansa21_A.csv

File names are kept as they are, never translated. The species is not written
into the CSV: it lives in each site's `species_id` in parameters_2021.py, and
this script fails if the Motti file disagrees with it.

Only the referenced files are converted; the other Motti files are #291.

Usage (from the repo root):
    uv run python test_projects/susi_2021/convert_motti.py
"""

import os
from pathlib import Path

import pandas as pd

from susi.core.allometry_columns import ALLOMETRY_COLUMNS

from parameters_2021 import ALLOMETRY_DIR, MOTTI_DIR, PARAMS_PER_SITE

# Sheets are read by position: in these files the first sheet's name has a
# leading space (" Puustotunnukset"), so reading by name would be brittle.
ALLOMETRY_SHEET = 0  # Puustotunnukset
ACCUMULATION_SHEET = 1  # Kertymät
SPECIES_COLUMN = "id Puulaji"


def _read_sheet(motti_path: Path, sheet_index: int) -> pd.DataFrame:
    # These old .xls files make xlrd print harmless OLE2 size warnings to
    # stdout; send them to devnull so the report stays readable.
    with open(os.devnull, "w") as devnull:
        return pd.read_excel(
            motti_path,
            sheet_name=sheet_index,
            engine="xlrd",
            engine_kwargs={"logfile": devnull},
        )


def rename_to_canonical_columns(
    allometry_sheet: pd.DataFrame, motti_path: Path
) -> pd.DataFrame:
    """
    Check that the first sheet's non-empty columns are the canonical columns'
    former Finnish names, in order, and rename them to the canonical names.
    ALLOMETRY_COLUMNS stays the only Finnish-to-English header mapping.
    """
    # The sheets carry ~150 columns, of which only the first ones hold data;
    # the rest are blank and come in with an "Unnamed: N" header. A blank
    # column is dropped; a column with data but no header survives and then
    # fails the header check below.
    table = allometry_sheet.dropna(axis="columns", how="all")

    expected = [c.former_finnish_name for c in ALLOMETRY_COLUMNS]
    found = list(table.columns)
    if found != expected:
        raise ValueError(
            f"{motti_path.name}: first-sheet headers do not match the canonical "
            f"allometry columns' Finnish names.\n  expected: {expected}\n"
            f"  found:    {found}"
        )
    if table.isna().any().any():
        raise ValueError(f"{motti_path.name}: first sheet has empty cells.")

    return table.rename(
        columns={c.former_finnish_name: c.name for c in ALLOMETRY_COLUMNS}
    )


def species_from_accumulation_sheet(
    accumulation_sheet: pd.DataFrame, motti_path: Path
) -> int:
    """
    Read the species from the second sheet's `id Puulaji`, which must hold
    exactly one distinct value: one allometry file describes one species.
    """
    if SPECIES_COLUMN not in accumulation_sheet.columns:
        raise ValueError(
            f"{motti_path.name}: second sheet has no '{SPECIES_COLUMN}' column."
        )
    species = accumulation_sheet[SPECIES_COLUMN].dropna().unique()
    if len(species) == 0:
        raise ValueError(
            f"{motti_path.name}: second sheet records no species "
            f"('{SPECIES_COLUMN}' is empty)."
        )
    if len(species) > 1:
        raise ValueError(
            f"{motti_path.name}: second sheet records more than one species "
            f"in '{SPECIES_COLUMN}': {sorted(species.tolist())}."
        )
    return int(species[0])


def convert_motti_file(
    motti_path: Path, csv_path: Path, expected_species_id: int
) -> None:
    """
    Write the CSV allometry file, after checking that the species the Motti
    file records is the site's `species_id`. Every check runs before the
    write, so a failing file never leaves a CSV behind.
    """
    table = rename_to_canonical_columns(
        _read_sheet(motti_path, ALLOMETRY_SHEET), motti_path
    )
    species_id = species_from_accumulation_sheet(
        _read_sheet(motti_path, ACCUMULATION_SHEET), motti_path
    )
    if species_id != expected_species_id:
        raise ValueError(
            f"{motti_path.name}: records species {species_id}, but the site's "
            f"species_id is {expected_species_id}."
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)


def main() -> None:
    for site_label, site_params in PARAMS_PER_SITE.items():
        csv_path = ALLOMETRY_DIR / site_params.allometry_file
        motti_path = MOTTI_DIR / Path(site_params.allometry_file).with_suffix(".xls")

        convert_motti_file(motti_path, csv_path, site_params.species_id)
        print(
            f"{site_label:<10} {motti_path.name} -> {csv_path.name}  "
            f"species {site_params.species_id}"
        )


if __name__ == "__main__":
    main()
