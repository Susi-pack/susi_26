from functools import lru_cache
import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Self, Union
import numpy as np
import pandas as pd

from pydantic import (
    DirectoryPath,
    Field,
    FilePath,
    SkipValidation,
    field_validator,
    PrivateAttr,
    model_validator,
    NonNegativeInt,
)

from susi.io.extra_pydantic_types import (
    StrictFrozenModel,
    PositiveFloat,
    NonNegativeFloat,
    NonPositiveFloat,
    PositiveInt,
)


def mass_mor_from_drainage_Pitkanen(drain_age: float) -> float:
    """
    Pitkänen et al. 2012 Forest Ecology and Management 284 (2012) 100–106
    mass of humus layer in kg m-2
    """
    return 1.616 * np.log(drain_age) - 1.409


def h_mor_from_drainage_and_mass_mor_Pitkanen(
    drain_age: float, rho_mor: float
) -> float:
    """
    Depth of humus layer (in meters) from the hmuslayer mass (in kg m-2)
    """
    return mass_mor_from_drainage_Pitkanen(drain_age) / rho_mor


@lru_cache()
def read_allometry_info_from_excel(filepath: Path) -> tuple[pd.DataFrame, int]:
    """
    Read allometry file, return allometry dataframe and species id.
    It is cached so that the same file is not read twice.
    """

    cnames = [
        "yr",
        "age",
        "N",
        "BA",
        "Hg",
        "Dg",
        "hdom",
        "vol",
        "logs",
        "pulp",
        "loss",
        "yield",
        "mortality",
        "stem",
        "stemloss",
        "branch_living",
        "branch_dead",
        "leaves",
        "stump",
        "roots_coarse",
        "roots_fine",
    ]
    df = pd.read_excel(
        filepath, sheet_name=0, usecols=range(22), skiprows=1, header=None
    )
    df = df.drop([0], axis=1)
    df.columns = cnames
    cname = ["idSpe"]
    df2 = pd.read_excel(filepath, sheet_name=1, usecols=[4], skiprows=1, header=None)
    df2.columns = cname

    # ---- find thinnings and add a small time to lines with the age to enable interpolation---------
    df = df.loc[df["age"] != 0]

    steps = np.array(np.diff(df["age"]), dtype=float)
    idx = np.ravel(np.argwhere(steps < 1.0)) + 1
    df.loc[idx, "age"] = df.loc[idx, "age"] + 5.0 / 365.0

    return df, df2["idSpe"][0]


class SimulationConfig(StrictFrozenModel):
    # Time
    start_date: datetime.datetime = Field(description="Simulation start date.")
    end_date: datetime.datetime = Field(description="Simulation end date.")


class WeatherParams(StrictFrozenModel):
    """
    Weather parameters
    """

    # FMI weather
    FMI_weather_filepath: FilePath = Field(description="Path to weather files.")


class AllometryParams(StrictFrozenModel):
    """
    Allometry .xlsx files to read
    """

    allometry_dir_path: DirectoryPath = Field(
        description="Folder where to look for the allometry files."
    )
    dominant: dict[int, str] = Field(
        description="int: 0 if not in use. str: Name of the allometry file in `allometry_dir_path` for the dominant layer."
    )
    subdominant: dict[int, str] = Field(
        description="int: 0 if not in use. str: Name of the allometry file in `allometry_dir_path` for the subdominant layer."
    )
    under: dict[int, str] = Field(
        description="int: 0 if not in use. str: Name of the allometry file in `allometry_dir_path` for the understorey layer."
    )

    # Information read from the excel file, not serialized
    _dominant_data: dict[int, pd.DataFrame] = PrivateAttr()
    _dominant_species_id: dict[int, int] = PrivateAttr()

    _subdominant_data: dict[int, pd.DataFrame] = PrivateAttr()
    _subdominant_species_id: dict[int, int] = PrivateAttr()

    _under_data: dict[int, pd.DataFrame] = PrivateAttr()
    _under_species_id: dict[int, int] = PrivateAttr()

    @model_validator(mode="after")
    def parse_excels(self) -> Self:
        _dominant_data = {}
        _dominant_species_id = {}

        _subdominant_data = {}
        _subdominant_species_id = {}

        _under_data = {}
        _under_species_id = {}

        for id, filename in self.dominant.items():
            if id != 0:
                df, species_id = read_allometry_info_from_excel(
                    filepath=self.allometry_dir_path / filename
                )

                _dominant_data[id] = df
                _dominant_species_id[id] = species_id

        for id, filename in self.subdominant.items():
            if id != 0:
                df, species_id = read_allometry_info_from_excel(
                    filepath=self.allometry_dir_path / filename
                )

                _subdominant_data[id] = df
                _subdominant_species_id[id] = species_id

        for id, filename in self.under.items():
            if id != 0:
                df, species_id = read_allometry_info_from_excel(
                    filepath=self.allometry_dir_path / filename
                )

                _under_data[id] = df
                _under_species_id[id] = species_id

        self._dominant_data = _dominant_data
        self._dominant_species_id = _dominant_species_id

        self._subdominant_data = _subdominant_data
        self._subdominant_species_id = _subdominant_species_id

        self._under_data = _under_data
        self._under_species_id = _under_species_id

        return self

    @property
    def dominant_data(self) -> dict[int, pd.DataFrame]:
        return self._dominant_data

    @property
    def dominant_species_id(self) -> dict[int, int]:
        return self._dominant_species_id

    @property
    def subdominant_data(self) -> dict[int, pd.DataFrame]:
        return self._subdominant_data

    @property
    def subdominant_species_id(self) -> dict[int, int]:
        return self._subdominant_species_id

    @property
    def under_data(self) -> dict[int, pd.DataFrame]:
        return self._under_data

    @property
    def under_species_id(self) -> dict[int, int]:
        return self._under_species_id


class CanopyStateParams(StrictFrozenModel):
    """
    Canopy state parameters
    """

    lai_conif: float = Field(default=3.0, description="conifer 1-sided LAI (m2 m-2)")
    lai_decid_max: float = Field(
        default=0.01, description="maximum annual deciduous 1-sided LAI (m2 m-2):"
    )
    hc: float = Field(default=16.0, description="canopy height (m)")
    cf: float = Field(default=0.7, description="canopy closure fraction (-)")

    w: float = Field(default=0.0, description="Initial state of canopy storage (mm)")
    swe: float = Field(
        default=0.0, description="Initial state of snow water equivalent (mm)"
    )


class CanopyStateParamsArray:
    """
    Canopy and moss for each soil column (0, and n-1 are ditches)
    Same as CanopyStateParameters, but with all array elements.
    """

    lai_conif: np.ndarray
    lai_decid_max: np.ndarray
    hc: np.ndarray
    cf: np.ndarray
    # initial state of canopy storage [mm] and snow water equivalent [mm]
    w: np.ndarray
    swe: np.ndarray

    def __init__(self, canopy_state_parameters: CanopyStateParams, array_length: int):
        for name, value in canopy_state_parameters.model_dump().items():
            setattr(self, name, value * np.ones(array_length))


class Flow(StrictFrozenModel):
    """
    Canopy flow field parameters
    """

    # Flow field
    zmeas: float = 2.0
    zground: float = Field(
        default=0.5, description="Reference height above ground (m)."
    )
    zo_ground: float = Field(default=0.01, description="ground roughness length (m).")


class Interception(StrictFrozenModel):
    """
    Canopy interception parameters
    """

    # interception
    wmax: float = 0.5
    wmaxsnow: float = 4.0


class Snow(StrictFrozenModel):
    """
    Snow parameters
    """

    # degree-day snow model
    kmelt: float = Field(
        default=2.8934e-05, description="melt coefficient in open (mm/s)"
    )
    kfreeze: float = Field(default=5.79e-6, description="freezing coefficient (mm/s)")
    r: float = Field(
        default=0.05, description="maximum fraction of liquid water in snow (-)"
    )


class Physpara(StrictFrozenModel):
    """
    Physpara parameters: canopy conductance and soil evaporation.
    """

    # canopy conductance
    amax_init: float = Field(
        frozen=False,
        default=10.0,
        description="Initial maximum photosynthetic rate (umolm-2(leaf)s-1)",
    )
    g1_conif: float = Field(default=2.1, description="stomatal parameter, conifers")
    g1_decid: float = Field(default=3.5, description="stomatal parameter, deciduous")
    q50: float = Field(default=50.0, description="light response parameter (Wm-2)")
    kp: float = Field(default=0.6, description="light attenuation parameter (-)")
    rw: float = Field(default=0.20, description="critical value for REW (-),")
    rwmin: float = Field(default=0.02, description="minimum relative conductance (-)")
    # soil evaporation
    gsoil: float = Field(
        default=1e-2,
        description="Soil surface conductance if soil is fully wet (m/s)",
    )


class Phenology(StrictFrozenModel):
    """
    Canopy phenology parameters. Seasonal cycle of physiology.
    """

    # seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
    smax: float = Field(default=18.5, description="degC")
    tau: float = Field(default=13.0, description="days")
    xo: float = Field(default=-4.0, description="degC")
    fmin: float = Field(
        default=0.05, description="minimum photosynthetic capacity in winter (-)"
    )


class CanopyParams(StrictFrozenModel):
    """
    Canopy parameters
    """

    dt: PositiveFloat = Field(default=86400.0, description="Canopy model timestep (s).")

    flow: Flow = Flow()
    interception: Interception = Interception()
    snow: Snow = Snow()
    physpara: Physpara = Physpara()
    phenology: Phenology = Phenology()
    state: CanopyStateParams = CanopyStateParams()


class OrganicLayerParams(StrictFrozenModel):
    """
    Parameters for the organic layer
    """

    org_depth: PositiveFloat = Field(
        default=0.04, description="depth of organic top layer (m)"
    )
    org_poros: PositiveFloat = Field(default=0.9, description="porosity (-)")
    org_fc: PositiveFloat = Field(default=0.3, description="field capacity (-)")
    org_rw: PositiveFloat = Field(
        default=0.24,
        description="critical vol. moisture content (-) for decreasing phase in Ef.",
    )
    pond_storage_max: PositiveFloat = Field(
        default=0.01, description="max ponding allowed (m)"
    )

    # initial values
    org_sat: PositiveFloat = Field(
        default=1.0, description="organic top layer saturation ratio (-)"
    )
    pond_storage: NonNegativeFloat = Field(default=0.0, description="pond storage")


class OrganicLayerParamsArray:
    """
    Same class as OrganicLayerParams, but with all fields a numpy array for each soil column.
    """

    org_depth: np.ndarray
    org_poros: np.ndarray
    org_fc: np.ndarray
    org_rw: np.ndarray
    pond_storage_max: np.ndarray
    org_sat: np.ndarray
    pond_storage: np.ndarray

    def __init__(self, organic_layer_parameters: OrganicLayerParams, array_length: int):
        for name, value in organic_layer_parameters.model_dump().items():
            setattr(self, name, value * np.ones(array_length))


class OutputParams(StrictFrozenModel):
    """
    Output file IO parameters
    """

    startday: int = 1
    startmonth: int = 7  # Päivä josta keskiarvojen laskenta alkaa
    endday: int = 31
    endmonth: int = 8  # Päivä johon keskiarvojen laskenta loppuu


class PhotoParameters(StrictFrozenModel):
    """
    Photosynthesis parameters for assimilation model (Mäkelä et al. 2008)
    """

    beta: float
    gamma: float
    kappa: float
    tau: float
    X0: float
    Smax: float
    alfa: float
    nu: float


class LocationsForPhotoParams(str, Enum):
    """
    Gives all options for the location of the photosynthesis parameters
    """

    all_data = "All_data"
    sodankyla = "Sodankyla"
    hyytiala = "Hyytiala"
    norunda = "Norunda"
    tharandt = "Tharandt"
    bray = "Bray"


PRESET_PHOTO_PARAMETERS: dict[str, PhotoParameters] = {
    "All_data": PhotoParameters(
        beta=0.513,
        gamma=0.0196,
        kappa=-0.389,
        tau=7.2,
        X0=-4.0,
        Smax=17.3,
        alfa=1.0,
        nu=5.0,
    ),
    "Sodankyla": PhotoParameters(
        beta=0.831,
        gamma=0.065,
        kappa=-0.150,
        tau=10.2,
        X0=-0.9,
        Smax=16.4,
        alfa=1.0,
        nu=5.0,
    ),
    "Hyytiala": PhotoParameters(
        beta=0.504,
        gamma=0.0303,
        kappa=-0.235,
        tau=11.1,
        X0=-3.1,
        Smax=17.3,
        alfa=1.0,
        nu=5.0,
    ),
    "Norunda": PhotoParameters(
        beta=0.500,
        gamma=0.0220,
        kappa=-0.391,
        tau=5.7,
        X0=-4.0,
        Smax=17.6,
        alfa=1.062,
        nu=11.27,
    ),
    "Tharandt": PhotoParameters(
        beta=0.742,
        gamma=0.0267,
        kappa=-0.512,
        tau=1.8,
        X0=-5.2,
        Smax=18.5,
        alfa=1.002,
        nu=442.0,
    ),
    "Bray": PhotoParameters(
        beta=0.459,
        gamma=-0.000669,
        kappa=-0.560,
        tau=2.6,
        X0=-17.6,
        Smax=45.0,
        alfa=0.843,
        nu=2.756,
    ),
}


def get_photo_parameters_by_location(
    location: LocationsForPhotoParams,
) -> PhotoParameters:
    return PRESET_PHOTO_PARAMETERS[location.value]


class TreeSpecies(str, Enum):
    """
    Possible tree species options
    """

    pine = "Pine"
    spruce = "Spruce"
    birch = "Birch"


class PeatTypes(str, Enum):
    """
    Possible choices for peat types
    """

    generic = "A"
    sphagnum = "S"


class NutrientFertilizationParameters(StrictFrozenModel):
    """
    Nutrient fertilization parameters
    """

    dose: NonNegativeFloat = Field(
        description="Dose of compound in fertilizer, kg ha-1"
    )
    decay_k: NonNegativeFloat = Field(description="Decay rate, yr-1")
    eff: NonNegativeFloat = Field(description="Nutrient use efficiency")


class StandardNPKFertilizationParameters(StrictFrozenModel):
    """
    Fertilization parameters for all fertilizers except wood ash.
    First order decay function.
    """

    application_year: int
    N: NutrientFertilizationParameters
    P: NutrientFertilizationParameters
    K: NutrientFertilizationParameters
    pH_increment: NonNegativeFloat = 0.0


class AshFertilizationParameters(StrictFrozenModel):
    """
    Ash fertilization parameters
    """

    application_year: int
    grain_radius: NonNegativeFloat = Field(
        default=0.005, description="Radius of ash grains (m)."
    )
    particle_cracking_rate: NonNegativeFloat = Field(
        default=2.0, description="How fast particles break down"
    )
    dissolution_rate: NonNegativeFloat = Field(
        default=0.005, description="Liukoisuusvakiot. (kg/m^2/year)"
    )
    K_dissolution_rate: NonNegativeFloat = Field(
        default=0.00012, description="Potassium release rate. (kg/m^2/year)"
    )
    P_dissolution_rate: NonNegativeFloat = Field(
        default=0.000045, description="Phosphorus release rate. (kg/m^2/year)"
    )
    density: NonNegativeFloat = Field(
        default=1000, description="Density of ash grains (kg/m^3)"
    )
    fertilizer_dose: NonNegativeFloat = Field(
        description="Mass of the ash fertilizer (kg/ha)"
    )
    K_in_ash: NonNegativeFloat = Field(
        description="Amount of potassium in the fertilizer (kg/ha)"
    )
    P_in_ash: NonNegativeFloat = Field(
        description="Amount of phosphorus in the fertilizer (kg/ha)"
    )
    time_exp: NonNegativeFloat = Field(
        description="Exponent in the grain cracking function."
    )


FertilizationParameters = Union[
    StandardNPKFertilizationParameters | AshFertilizationParameters
]


class PeatTemperatureParams(StrictFrozenModel):
    """
    Peat soil temperature parameters
    """

    heat_of_vaporization: PositiveFloat = Field(default=2467700.0, description="J/kg")
    timestep: PositiveFloat = Field(
        default=86400.0, description="timestep in seconds (s)"
    )

    n_subtimesteps: PositiveInt = Field(
        default=24,
        description="number of subtimesteps in the time step (here every 2 hrs",
    )

    D: PositiveFloat = Field(
        default=1e-7, description="Thermal diffusivity of peat (m2 s-1) de Vries 1975"
    )


class CanopyLayerAllometryPointers(StrictFrozenModel):
    """
    Pointers to the allometry files for all canopy layers: dominant, subdominant, understorey
    Each list must contain one non-negative integer per soil column.
    Integers in the list reference the keys in the `AllometryParams` dictionaries.
    0 implies that layer is not present for that soil column.
    """

    dominant: list[NonNegativeInt]
    subdominant: list[NonNegativeInt]
    under: list[NonNegativeInt]


class SiteParams(StrictFrozenModel):
    """
    Soil and stand parameters
    """

    # Forest
    # Age of different forest layers at the beginning of the simulation
    initial_dominant_stand_age_years: NonNegativeFloat = Field(
        description="Age of the dominant stand at the beginning of the simulation. This is set to all nodes in the strip."
    )
    initial_subdominant_stand_age_years: NonNegativeFloat = Field(
        description="Age of the subdominant layer at the beginning of the simulation. This is set to all nodes in the strip."
    )
    initial_understorey_age_years: NonNegativeFloat = Field(
        description="Age of the understorey at the beginning of the simulation. This is set to all nodes in the strip."
    )
    canopylayers: CanopyLayerAllometryPointers

    L: float = Field(description="Strip width, i.e., distance between ditches, m")

    n: NonNegativeInt = Field(
        description="Number of computation nodes, a.k.a. number of soil columns. It is usually `int(L/2)`."
    )

    site_fertility_class: PositiveInt = Field(
        description="Site fertility class. This is set to all nodes in the strip."
    )

    sitename: str
    species: TreeSpecies
    sfc_specification: float
    hdom: float | None
    vol: list[float] | None
    smc: str
    nLyrs: int
    dzLyr: float
    ditch_depth_west: list[NonPositiveFloat] = Field(
        description="ditch depth at the beginning of simulation (m). If given several values SUSI calculates scenarios for each ditch depth."
    )
    ditch_depth_east: list[NonPositiveFloat] = Field(
        description="ditch depth at the beginning of simulation (m). If given several values SUSI calculates scenarios for each ditch depth."
    )
    ditch_depth_20y_west: list[NonPositiveFloat] = Field(
        description="Ditch depth after 20 yrs, m, negative down"
    )
    ditch_depth_20y_east: list[NonPositiveFloat] = Field(
        description="Ditch depth after 20 yrs, m, negative down"
    )
    scenario_name: list[str] = Field(
        description="Scenario names, equal nmber of names than ditch depth scenarios."
    )

    drain_age: PositiveFloat = Field(description="Time since drainage (yrs).")
    initial_h: float
    slope: float
    peat_type: list[PeatTypes] = Field(
        description="'S' if Sphagnum, 'A' if woody or carex peat"
    )
    peat_type_bottom: list[PeatTypes]
    anisotropy: float = Field(description="Anisotropy of peat hydraulic conductivity")
    vonP: bool = Field(description="degree of decomposition, vonPost scale, int")
    vonP_top: list[int]
    vonP_bottom: int
    bd_top: float | None = Field(description="Bulk density (g/cm3).")
    bd_bottom: float
    peatN: float | None
    peatP: float | None
    peatK: float | None
    enable_peattop: bool
    enable_peatmiddle: bool
    enable_peatbottom: bool
    rho_mor: float = Field(description="bulk density of mor layer, kg m-3")
    h_mor: NonNegativeFloat | Callable[..., float] = Field(
        description="depth of mor layer, m"
    )
    cutting_yr: int = Field(
        description="Year for cutting. Not used if year is outside the simulation period."
    )
    cutting_to_ba: float = Field(description="basal area after cutting, m2/ha")
    depoN: float
    depoP: float
    depoK: float
    fertilization: FertilizationParameters | None = Field(
        description="Use None for no fertilization.", default=None
    )
    peat_temperature: PeatTemperatureParams

    @property
    def age(self) -> SkipValidation[dict[str, np.ndarray]]:
        """Age of stand for all nodes along the strip"""
        return {
            "dominant": self.initial_dominant_stand_age_years * np.ones(self.n),
            "subdominant": self.initial_subdominant_stand_age_years * np.ones(self.n),
            "under": self.initial_understorey_age_years * np.ones(self.n),
        }

    @property
    def sfc(self) -> SkipValidation[np.ndarray]:
        """site fertility class for all nodes in the strip"""
        return np.ones(self.n, dtype=int) * self.site_fertility_class

    @field_validator("h_mor", mode="before")
    @classmethod
    def compute_if_callable(cls, hmor, info):
        if callable(hmor):
            drain_age = info.data.get("drain_age")
            rho_mor = info.data.get("rho_mor")
            if drain_age is None:
                raise ValueError("`drain_age` must be provided to compute hmor")
            if rho_mor is None:
                raise ValueError("`rho_mor` must be provided to compute hmor")
            try:
                return hmor(drain_age, rho_mor)
            except Exception as e:
                raise ValueError(f"Failed to compute h_mor: {e}")
        return hmor

    @model_validator(mode="after")
    def canopy_layer_elements(self) -> Self:
        n = self.n
        for layer_name in ["dominant", "subdominant", "under"]:
            layer_list = getattr(self.canopylayers, layer_name)
            if len(layer_list) != n:
                raise ValueError(
                    f"CanopyLayerAllometryPointers.{layer_name} has {len(layer_list)} elements, "
                    f"but must have {n} elements"
                )
        return self


class SusiParams(StrictFrozenModel):
    """
    Parameter class to be stantiated.
    """

    params_schema_version: int = 1
    weather_parameters: WeatherParams
    allometry_parameters: AllometryParams
    simulation_config: SimulationConfig
    canopy_parameters: CanopyParams
    organic_layer_parameters: OrganicLayerParams
    output_parameters: OutputParams
    photo_parameters: PhotoParameters
    site_parameters: SiteParams

    @model_validator(mode="after")
    def check_fertilization_within_bounds(self) -> Self:
        # Now 'self' is SusiParams, which CAN see both children
        config = self.simulation_config
        site = self.site_parameters

        if site.fertilization is not None:
            app_year = site.fertilization.application_year
            if not (config.start_date.year <= app_year <= config.end_date.year):
                raise ValueError(f"Fertilization year {app_year} is out of bounds!")
        return self

    @model_validator(mode="after")
    def stand_age_vs_allometry_pathway(self) -> Self:
        """
        Validates the following:
        - Initial stand age is not below the minimum in the allometry file
        - initial stand age + simulation time is not above the maximum age in the allometry file
        """
        simulation_duration_years = (
            self.simulation_config.end_date.year
            - self.simulation_config.start_date.year
        )

        self._validate_layer_age(
            layer_name="dominant",
            initial_age=self.site_parameters.initial_dominant_stand_age_years,
            allometry_data=self.allometry_parameters.dominant_data,
            simulation_duration=simulation_duration_years,
        )

        self._validate_layer_age(
            layer_name="subdominant",
            initial_age=self.site_parameters.initial_subdominant_stand_age_years,
            allometry_data=self.allometry_parameters.subdominant_data,
            simulation_duration=simulation_duration_years,
        )

        self._validate_layer_age(
            layer_name="under",
            initial_age=self.site_parameters.initial_understorey_age_years,
            allometry_data=self.allometry_parameters.under_data,
            simulation_duration=simulation_duration_years,
        )

        return self

    @model_validator(mode="after")
    def allometry_files_pointers(self) -> Self:
        for layer_name in ["dominant", "subdominant", "under"]:
            layer_pointers = getattr(self.site_parameters.canopylayers, layer_name)
            layer_keys = getattr(self.allometry_parameters, layer_name).keys()
            non_zero_pointers = set(p for p in layer_pointers if p != 0)
            # only check non-zero pointers. 0 means no tree in the layer.
            if non_zero_pointers:
                missing_keys = non_zero_pointers - layer_keys
                if missing_keys:
                    raise ValueError(
                        f"Allometry pointer(s) {missing_keys} in {layer_name} layer "
                        f"do not exist in AllometryParams.{layer_name} keys {set(layer_keys)}"
                    )
        return self

    def _validate_layer_age(
        self,
        layer_name: str,
        initial_age: float,
        allometry_data: dict[int, pd.DataFrame],
        simulation_duration: float,
    ) -> None:
        """Helper method to validate age for a single canopy layer."""
        if not allometry_data:
            # This convers the case of no stand in the canopy layer
            return

        # Compute the minimum and maximum of all dataframes
        min_age = float("inf")
        max_age = float("-inf")

        for df in allometry_data.values():
            min_age = min(min_age, df["age"].min())
            max_age = max(max_age, df["age"].max())

        if initial_age < min_age:
            raise ValueError(
                f"Initial {layer_name} stand age ({initial_age}) is below "
                f"minimum age ({min_age}) in allometry file"
            )

        if initial_age + simulation_duration > max_age:
            raise ValueError(
                f"Initial {layer_name} stand age ({initial_age}) plus simulation "
                f"duration ({simulation_duration:.1f} years) exceeds maximum age "
                f"({max_age}) in allometry file"
            )

    def dump_json_to_file(self, filepath: Path) -> None:
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))
