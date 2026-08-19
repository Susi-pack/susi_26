"""Run paired dynamic- and fixed-growth SUSI simulations for problem stands.

Each requested stand receives the standard reference, ditch-depth, and spacing
scenarios used by MK_susi_resume.py.  The two growth modes use separate
scenario IDs and a new experiment ID, so existing output is not overwritten.
"""

import argparse
import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from inputs.parameters import sample_parameters
from susi.core.susi_main import Susi
from susi.io.execution_config import MultipleSusis, SimulationParams
from susi.io.metadata_model import SimulationMetaData
from susi.io.susi_parameter_model import SusiParams


parser = argparse.ArgumentParser(
    description="Run paired dynamic and fixed growth simulations for problem stands."
)
parser.add_argument("n_parallel_processes", type=int)
cli_args = parser.parse_args()

ALLOMETRY_DIR = Path(
    "/Users/sandeep/mNFI/mNFIprocessing/DATA/susi_allometry/"
    "Uusimaa_dc12_volmedsplit_20260713_111857"
)
WEATHER_FILE = Path(
    "/Users/sandeep/mNFI/weather_data/susi_csv/"
    "susi_weather_E309560_N6747150_1980_2025.csv"
)
EXPERIMENT_ID = "problem_stands_fixed_vs_dynamic_growth_combined_effect_same_rew"
SIM_START = datetime.datetime(1995, 1, 1)
SIM_END = datetime.datetime(2025, 12, 31)

TARGET_STANDS = [
    "Uusimaa_sg2_fc2_dc2_vgdense__dom_10699335",
    "Uusimaa_sg2_fc2_dc2_vgdense__dom_25951700",
    "Uusimaa_sg2_fc3_dc2_vgsparse__dom_13605171",
    "Uusimaa_sg2_fc4_dc2_vgdense__dom_36001660",
    "Uusimaa_sg2_fc4_dc2_vgdense__dom_36113509",
    "Uusimaa_sg3_fc3_dc2_vgdense__dom_27757688",
    "Uusimaa_sg3_fc3_dc2_vgdense__dom_27868196",
    "Uusimaa_sg3_fc4_dc2_vgdense__dom_35201938",
    "Uusimaa_sg3_fc5_dc2_vgdense__dom_32486561",
]


def peat_type_from_filename(filename: str) -> str:
    label = Path(filename).stem.rsplit("_", 1)[-1].lower()
    return "S" if label == "sphagnum" else "A"


def target_entries() -> list[dict[str, object]]:
    entries = []
    for target in TARGET_STANDS:
        file_stem, dom_sheet = target.split("__", maxsplit=1)
        workbook = ALLOMETRY_DIR / f"{file_stem}.xlsx"
        if not workbook.is_file():
            raise FileNotFoundError(f"Allometry workbook not found: {workbook}")

        stand_table = pd.read_excel(workbook, sheet_name=dom_sheet, engine="openpyxl")
        age_column = next(
            (column for column in stand_table.columns if str(column).lower() == "age"),
            None,
        )
        if age_column is None:
            raise ValueError(f"No age column in {workbook.name}::{dom_sheet}")

        entries.append(
            {
                "target": target,
                "workbook": workbook,
                "dom_sheet": dom_sheet,
                "initial_age": float(stand_table[age_column].iloc[0]),
            }
        )
    return entries


def create_scenario(base_params: SusiParams, entry: dict[str, object], growth_mode: str, **overrides) -> SusiParams:
    data = base_params.model_dump(exclude_computed_fields=True)
    n_cols = len(data["site_parameters"]["canopylayers"]["dominant"])
    workbook = entry["workbook"]
    assert isinstance(workbook, Path)

    data["allometry_parameters"]["dominant"] = {1: f"{workbook}::{entry['dom_sheet']}"}
    data["allometry_parameters"]["subdominant"] = {0: ""}
    data["allometry_parameters"]["under"] = {0: ""}
    data["site_parameters"]["canopylayers"]["subdominant"] = [0] * n_cols
    data["site_parameters"]["canopylayers"]["under"] = [0] * n_cols
    data["site_parameters"]["initial_dominant_stand_age_years"] = entry["initial_age"]
    data["weather_parameters"]["FMI_weather_filepath"] = str(WEATHER_FILE)
    data["simulation_config"]["start_date"] = SIM_START.isoformat()
    data["simulation_config"]["end_date"] = SIM_END.isoformat()
    data["simulation_config"]["growth_mode"] = growth_mode

    peat_type = peat_type_from_filename(workbook.name)
    data["site_parameters"]["peat_type"] = [peat_type] * len(data["site_parameters"]["peat_type"])
    data["site_parameters"]["peat_type_bottom"] = [peat_type] * len(
        data["site_parameters"]["peat_type_bottom"]
    )

    for key, value in overrides.items():
        if key == "ditch_spacing":
            data["site_parameters"]["L"] = value
            n_cols = int(value / 2)
            data["site_parameters"]["canopylayers"]["dominant"] = [1] * n_cols
            data["site_parameters"]["canopylayers"]["subdominant"] = [0] * n_cols
            data["site_parameters"]["canopylayers"]["under"] = [0] * n_cols
        else:
            data["site_parameters"][key] = [value]

    return SusiParams.model_validate(data)


def build_simulations() -> list[SimulationParams]:
    base = sample_parameters.PARAMETERS
    base_depth = float(base.site_parameters.ditch_depth_west[0])
    base_spacing = float(base.site_parameters.L)
    templates = [("ref", {})]
    templates += [
        (
            f"ditch_{abs(int(depth * 10)):02d}dm",
            {
                "ditch_depth_west": depth,
                "ditch_depth_east": depth,
                "ditch_depth_20y_west": depth,
                "ditch_depth_20y_east": depth,
            },
        )
        for depth in (round(-level * 0.1, 1) for level in range(1, 16))
        if depth != base_depth
    ]
    templates += [
        (f"spacing_{spacing}m", {"ditch_spacing": float(spacing)})
        for spacing in (20, 30, 40, 50)
        if float(spacing) != base_spacing
    ]

    simulations = []
    for entry in target_entries():
        for growth_mode in ("dynamic", "fixed"):
            for scenario_id, overrides in templates:
                simulations.append(
                    SimulationParams(
                        metadata=SimulationMetaData(
                            experiment_id=EXPERIMENT_ID,
                            stand_id=entry["target"],
                            scenario_id=f"{growth_mode}_{scenario_id}",
                        ),
                        susi_params=create_scenario(base, entry, growth_mode, **overrides),
                    )
                )
    return simulations


def run_susi(simulation_parameters: SimulationParams) -> None:
    susi = Susi(simulation_parameters)
    susi.run()
    susi.write_params_and_metadata()


if __name__ == "__main__":
    simulations = build_simulations()
    print(f"Running {len(simulations)} simulations in {EXPERIMENT_ID}")
    execution_config = MultipleSusis(
        simulation_parameter_list=simulations,
        n_parallel_processes=min(cli_args.n_parallel_processes, len(simulations)),
    )
    with Pool(processes=execution_config.n_parallel_processes) as pool:
        pool.map(run_susi, execution_config.simulation_parameter_list)