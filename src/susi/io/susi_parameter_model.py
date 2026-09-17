from functools import lru_cache, cached_property
import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Self, Union, TypeAlias
import numpy as np
import pandas as pd

from pydantic import (
    Field,
    FilePath,
    SkipValidation,
    field_validator,
    PrivateAttr,
    model_validator,
)

from susi.io.extra_pydantic_types import (
    StrictFrozenModel,
    PositiveFloat,
    NonNegativeFloat,
    NonPositiveFloat,
    PositiveInt,
)
from susi.core.allometry_columns import ALLOMETRY_COLUMNS


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
def read_allometry_info_from_csv(filepath: Path) -> pd.DataFrame:
    """
    Read allometry file and return the allometry dataframe.
    It is cached so that the same file is not read twice.
    """
    column_names = [c.name for c in ALLOMETRY_COLUMNS]
    df = pd.read_csv(filepath)
    missing = set(column_names) - set(df.columns)
    if missing:
        raise ValueError(
            f"Allometry file {filepath} is missing expected columns: {missing}"
        )
    extra = set(df.columns) - set(column_names)
    if extra:
        raise ValueError(f"Allometry file {filepath} has unexpected columns: {extra}")

    # ---- find thinnings and add a small time to lines with the age to enable interpolation---------
    df = df.loc[df["Age"] != 0]

    steps = np.array(np.diff(df["Age"]), dtype=float)
    idx = np.ravel(np.argwhere(steps < 1.0)) + 1
    df.loc[idx, "Age"] = df.loc[idx, "Age"] + 5.0 / 365.0

    return df


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


class CanopyLayerName(str, Enum):
    dominant = "dominant"
    subdominant = "subdominant"
    under = "under"


class AllometryFileAndSpecies(StrictFrozenModel):
    file_path: Path
    species_id: PositiveInt


AllometryRegistryNumber: TypeAlias = PositiveInt


class CanopyLayerAllometry(StrictFrozenModel):
    """
    Allometry parameters
    """

    allometry_file_registry: dict[AllometryRegistryNumber, AllometryFileAndSpecies] = (
        Field(
            description=(
                "Map: allometry registry number -> allometry file. Example: "
                "{1:AllometryFileAndSpecies(file_path=Path('pines.csv'), "
                "species_id=1), 2:AllometryFileAndSpecies("
                "file_path=Path('spruces.csv'), species_id=2)}"
            )
        )
    )
    pointers: dict[CanopyLayerName, list[AllometryRegistryNumber] | None] = Field(
        description="Map: canopy layer name-> list of pointers, length ncols. Example: {'dominant': [1,1,1,1,2,2,2], 'subdominant': None, 'under': None}"
    )

    # Information read from the CSV file, not serialized
    # Map: Allometry registry number -> allometry path dataframe read from file
    _zones_data: dict[AllometryRegistryNumber, pd.DataFrame] = PrivateAttr()
    # Map: Allometry registry number -> species ID
    _zones_species_id: dict[AllometryRegistryNumber, int] = PrivateAttr()

    @model_validator(mode="after")
    def pointers_must_reference_declared_zones(self) -> Self:
        for layer_name, layer_pointers in self.pointers.items():
            if layer_pointers is None:
                continue

            declared = self.allometry_file_registry.keys()
            missing = set(layer_pointers) - declared
            if missing:
                raise ValueError(
                    f"Allometry pointer(s) {missing} in '{layer_name}' layer do not exist "
                    f"in allometry_file_registry keys {set(declared)}"
                )
        return self

    @model_validator(mode="after")
    def parse_allometry_files(self) -> Self:
        _zones_data = {}
        _zones_species_id = {}

        for (
            allometry_registry_number,
            filepath_and_species,
        ) in self.allometry_file_registry.items():
            df = read_allometry_info_from_csv(filepath=filepath_and_species.file_path)
            _zones_data[allometry_registry_number] = df
            _zones_species_id[allometry_registry_number] = (
                filepath_and_species.species_id
            )

        self._zones_data = _zones_data
        self._zones_species_id = _zones_species_id

        return self

    @cached_property
    def zones_data(
        self,
    ) -> dict[AllometryRegistryNumber, pd.DataFrame]:
        return {
            reg_number: read_allometry_info_from_csv(
                filepath=fpath_and_species.file_path
            )
            for reg_number, fpath_and_species in self.allometry_file_registry.items()
        }

    @cached_property
    def zones_species_id(self) -> dict[AllometryRegistryNumber, int]:

        return {
            reg_number: fpath_and_species.species_id
            for reg_number, fpath_and_species in self.allometry_file_registry.items()
        }


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


class ClearCut(StrictFrozenModel):
    """
    Parameters to define a clear- or strip-cut management intervention.
    """

    new_growth_allometry: CanopyLayerAllometry = Field(
        description="Allometry data to specify growth after clear cut."
    )
    strips_to_cut: list[bool] = Field(
        description="Boolean mask specifying which strips to cut. The length of the list must be the number of columns `n`. Cutting all strips, specified by [True, ..., True], implements clearcutting. Choosing only some strips implements strip cutting."
    )

    @field_validator("strips_to_cut")
    @classmethod
    def must_have_at_least_one_true(cls, v: list[bool]) -> list[bool]:
        if not any(v):
            raise ValueError(
                "strips_to_cut must contain at least one True entry. If you genuinely do not want to cut any strip, then set `cutting=None`."
            )
        return v

    @model_validator(mode="after")
    def new_allometry_includes_age_one(self) -> Self:
        for zid, df in self.new_growth_allometry.zones_data.items():
            if df["Age"].min() > 1:
                raise ValueError(
                    f"Post-clearcut allometry zone {zid} has a minimum age of {df['Age'].min()} years, but must start from age=1 year."
                )
        return self

    @model_validator(mode="after")
    def new_growth_allometry_matches_cut_column_count(self) -> Self:
        """new_growth_allometry provides one allometry pointer per *cut* soil
        column (True entries in strips_to_cut), not one per soil column in
        the full stand -- uncut columns keep growing under the pre-cut
        allometry and don't need a post-clearcut pointer."""
        n_cut_columns = sum(self.strips_to_cut)
        for layer, post_cut_pointers in self.new_growth_allometry.pointers.items():
            if post_cut_pointers is None:
                continue
            if len(post_cut_pointers) != n_cut_columns:
                raise ValueError(
                    f"ClearCut.new_growth_allometry.pointers['{layer.value}'] has "
                    f"{len(post_cut_pointers)} elements, but must have {n_cut_columns} "
                    f"elements, i.e., one per cut soil column (True entries in "
                    f"strips_to_cut) -- not one per soil column in the full stand."
                )
        return self


class Thinning(StrictFrozenModel):
    """
    Parameters to define a thinning intervention.
    """

    target_basal_area: dict[CanopyLayerName, PositiveFloat] = Field(
        description="Basal area after thinning (m2/ha) for each canopy layer. If a given canopy layer does not appear in the dictionary, it is omitted and no thinning is applied to it. Example: {'dominant': 2.0, 'subdominant': 3.0} -> This omits thinning to the 'under' layer."
    )

    @field_validator("target_basal_area")
    @classmethod
    def thinning_is_not_clearcut(
        cls, target_basal_area: dict[CanopyLayerName, float]
    ) -> dict[CanopyLayerName, float]:
        for layer_name, to_ba in target_basal_area.items():
            if to_ba < 1.0:
                raise ValueError(
                    f"`target_basal_area['{layer_name.value}']` must be higher than 1.0 m2/ha. "
                    "A thinning that reaches a basal area of less than 1.0 m2/ha is here "
                    "considered to be better modeled by a clear cut."
                )
        return target_basal_area


class ContinuousCover(StrictFrozenModel):
    def __init__(self, **data):
        raise NotImplementedError("Not yet implemented")


class CuttingManagementParams(StrictFrozenModel):
    """
    Parameters to define a thinning, clear cut (including selected strip cut),
    continuous cover management intervention.
    """

    application_yr: int = Field(
        description="Year for cutting management application. Must be inside the simulation period."
    )
    management_type: Union[ClearCut | ContinuousCover | Thinning] = Field(
        description="Type of cutting management selected."
    )


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


class StandParams(StrictFrozenModel):
    site_fertility_class: PositiveInt = Field(
        description="Site fertility class. This is set to all nodes in the strip."
    )

    canopy_layer_allometry: CanopyLayerAllometry
    # TODO:
    # age: dict[int, float] and soil_type: int are explicitly NOT in this ticket's scope --
    # soil_type's mapping to peat_type/peat_type_bottom is open work tracked by #280, and
    # age has no consumer until that lands either. Both stay future work, not invented here.


class SiteParams(StrictFrozenModel):
    """
    Soil and stand parameters
    """

    stand_params: StandParams

    # Forest
    # Age of different forest layers at the beginning of the simulation
    initial_canopylayer_age_years: dict[CanopyLayerName, NonNegativeFloat] = Field(
        description="Age of the different canopy layers at the beginning of the simulation. This is set to all nodes in the strip. Example: {'dominant': 20, 'subdominant': 0, 'under': 20 }."
    )

    L: float = Field(description="Strip width, i.e., distance between ditches, m")

    n: int = Field(
        description="Number of computation nodes, a.k.a. number of soil columns. It used to be`int(L/2)`. It must be at least 3, because a 2-node strip has no interior columns",
        ge=3,
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
        description="ditch depth at the beginning of simulation (m). Must have exactly one element unless `allow_multiple_ditch_scenarios` is True (see its description)."
    )
    ditch_depth_east: list[NonPositiveFloat] = Field(
        description="ditch depth at the beginning of simulation (m). Must have exactly one element unless `allow_multiple_ditch_scenarios` is True (see its description)."
    )
    ditch_depth_20y_west: list[NonPositiveFloat] = Field(
        description="Ditch depth after 20 yrs, m, negative down. Must have exactly one element unless `allow_multiple_ditch_scenarios` is True (see its description)."
    )
    ditch_depth_20y_east: list[NonPositiveFloat] = Field(
        description="Ditch depth after 20 yrs, m, negative down. Must have exactly one element unless `allow_multiple_ditch_scenarios` is True (see its description)."
    )
    scenario_name: list[str] = Field(
        description="Scenario names, one per ditch depth scenario (same length as the ditch_depth_* lists). Must have exactly one element unless `allow_multiple_ditch_scenarios` is True (see its description)."
    )
    allow_multiple_ditch_scenarios: bool = Field(
        default=False,
        description=(
            "Advanced/legacy escape hatch. By default, ditch_depth_west, ditch_depth_east, "
            "ditch_depth_20y_west, ditch_depth_20y_east and scenario_name must each contain "
            "exactly one element, and SUSI runs a single ditch-depth scenario. Set this to "
            "True to allow those lists to hold more than one element, in which case SUSI "
            "loops over them and computes one scenario per element within a single run. "
            "Most users should leave this False and instead call SUSI once per scenario; "
            "this is kept for backward compatibility with the historical multi-scenario "
            "workflow (see issue #30)."
        ),
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
    cutting_management: CuttingManagementParams | None = Field(
        description="Implement management interventions involving cutting, such as clearcutting and thinning. Use `None` for no cutting during the simulation.",
        default=None,
    )
    depoN: float
    depoP: float
    depoK: float
    fertilization: FertilizationParameters | None = Field(
        description="Use None for no fertilization.", default=None
    )
    peat_temperature: PeatTemperatureParams

    @property
    def age(self) -> SkipValidation[dict[CanopyLayerName, np.ndarray]]:
        """Age of stand for all nodes along the strip"""
        return {
            layer_name: self.initial_canopylayer_age_years[layer_name] * np.ones(self.n)
            for layer_name in CanopyLayerName
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

    @model_validator(mode="before")
    @classmethod
    def check_old_cutting_api(cls, data: dict) -> dict:
        if isinstance(data, dict) and ("cutting_yr" in data or "cutting_to_ba" in data):
            raise ValueError(
                "The `cutting_yr` and `cutting_to_ba` fields have been replaced by the "
                "`cutting_management` field.\n"
                "Use `cutting_management=CuttingManagementParams(application_yr=..., "
                "management_type=Thinning(target_basal_area=...))` for thinning or\n"
                "`cutting_management=CuttingManagementParams(application_yr=..., "
                "management_type=ClearCut(new_growth_allometry=..., strips_to_cut=...))` "
                "for clear-cutting."
            )
        return data

    @model_validator(mode="after")
    def single_ditch_scenario_by_default(self) -> Self:
        lists = {
            "ditch_depth_west": self.ditch_depth_west,
            "ditch_depth_east": self.ditch_depth_east,
            "ditch_depth_20y_west": self.ditch_depth_20y_west,
            "ditch_depth_20y_east": self.ditch_depth_20y_east,
            "scenario_name": self.scenario_name,
        }
        lengths = {name: len(values) for name, values in lists.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "ditch_depth_west, ditch_depth_east, ditch_depth_20y_west, "
                "ditch_depth_20y_east and scenario_name must all have the same "
                f"number of elements, got {lengths}"
            )
        n_scenarios = next(iter(lengths.values()))
        if n_scenarios != 1 and not self.allow_multiple_ditch_scenarios:
            raise ValueError(
                f"Got {n_scenarios} ditch-depth scenarios, but SiteParams only "
                "accepts one by default. Call SUSI once per scenario instead. "
                "If you really need multiple ditch-depth scenarios computed "
                "within a single run, set allow_multiple_ditch_scenarios=True "
                "(advanced/legacy usage, see issue #30)."
            )
        return self

    @model_validator(mode="after")
    def clear_cut_elements_same_as_soil_columns(self) -> Self:
        if self.cutting_management is not None:
            if isinstance(self.cutting_management.management_type, ClearCut):
                if len(self.cutting_management.management_type.strips_to_cut) != self.n:
                    raise ValueError(
                        f"ClearCut.strips_to_cut has {len(self.cutting_management.management_type.strips_to_cut)} elements, "
                        f"but must have {self.n} elements (equal to the number of soil columns)"
                    )
        return self


class SusiParams(StrictFrozenModel):
    """
    Parameter class to be stantiated.
    """

    params_schema_version: int = 1
    weather_parameters: WeatherParams
    allometry_parameters: CanopyLayerAllometry
    simulation_config: SimulationConfig
    canopy_parameters: CanopyParams
    organic_layer_parameters: OrganicLayerParams
    output_parameters: OutputParams
    photo_parameters: PhotoParameters
    site_parameters: SiteParams

    @model_validator(mode="after")
    def check_cutting_within_years(self) -> Self:
        config = self.simulation_config
        site = self.site_parameters

        match site.cutting_management:
            case None:
                pass
            case _any_other:
                cut_year = site.cutting_management.application_yr
                if not (config.start_date.year <= cut_year <= config.end_date.year):
                    raise ValueError(
                        f"Cutting management year {cut_year} is out of bounds wrt the simulation years ({config.start_date.year} -- {config.end_date.year})! You cannot cut a forest before or after the simulation period."
                    )
        return self

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

        # Each layer might have a different initial stand age
        for canopy_layer in CanopyLayerName:
            layer_pointers = self.allometry_parameters.pointers.get(canopy_layer)
            if layer_pointers is None:
                continue
            layer_zones = {
                zone_id: self.allometry_parameters.zones_data[zone_id]
                for zone_id in set(layer_pointers)
            }
            self._validate_layer_age(
                layer_name=canopy_layer.value,
                initial_age=self.site_parameters.initial_canopylayer_age_years[
                    canopy_layer
                ],
                allometry_data=layer_zones,
                simulation_duration=simulation_duration_years,
            )

        return self

    @model_validator(mode="after")
    def canopy_layers_must_have_number_of_soil_columns(self) -> Self:
        n_soil_cols = self.site_parameters.n
        for (
            layer_name,
            pointer_column_list,
        ) in self.allometry_parameters.pointers.items():
            if pointer_column_list is None:
                continue

            pointer_list_length = len(pointer_column_list)
            if pointer_list_length != n_soil_cols:
                raise ValueError(
                    f"CanopyLayerAllometry.pointers[{layer_name}] has {pointer_list_length} elements, "
                    f"but must have {n_soil_cols} elements, i.e., same elements as number of soil columns"
                )
        return self

    @model_validator(mode="after")
    def clearcut_provides_new_growth_allometry_for_every_real_layer(self) -> Self:
        cutting_management = self.site_parameters.cutting_management
        if cutting_management is None:
            return self

        management_type = cutting_management.management_type
        if not isinstance(management_type, ClearCut):
            return self

        for layer, pre_cut_pointers in self.allometry_parameters.pointers.items():
            if pre_cut_pointers is None:
                continue

            post_cut_pointers = management_type.new_growth_allometry.pointers.get(layer)
            if not post_cut_pointers:
                raise ValueError(
                    f"A clear-cut is scheduled and the '{layer.value}' layer exists, "
                    f"but ClearCut.new_growth_allometry provides no post-clearcut "
                    f"allometry pointers for '{layer.value}'."
                )
        return self

    @model_validator(mode="after")
    def thinning_only_targets_layers_with_allometry(self) -> Self:
        """A canopy layer with pointers=None in allometry_parameters has no
        real stand growing in it -- Thinning.target_basal_area must not name
        a layer that doesn't exist."""
        cutting_management = self.site_parameters.cutting_management
        if cutting_management is None:
            return self

        management_type = cutting_management.management_type
        if not isinstance(management_type, Thinning):
            return self

        for layer in management_type.target_basal_area:
            if self.allometry_parameters.pointers.get(layer) is None:
                raise ValueError(
                    f"Thinning.target_basal_area targets the '{layer.value}' layer, "
                    f"but that layer does not exist (allometry_parameters.pointers"
                    f"['{layer.value}'] is None) -- there is no stand there to thin."
                )
        return self

    def _validate_layer_age(
        self,
        layer_name: str,
        initial_age: float,
        allometry_data: dict[AllometryRegistryNumber, pd.DataFrame],
        simulation_duration: float,
    ) -> None:
        """Helper method to validate age for a single canopy layer."""

        for registry_number, df in allometry_data.items():
            min_age_in_dataframe = df["Age"].min()
            max_age_in_dataframe = df["Age"].max()

            if initial_age < min_age_in_dataframe:
                raise ValueError(
                    f"Initial {layer_name} stand age ({initial_age}) is below "
                    f"minimum age ({min_age_in_dataframe}) in the allometry file corresponding to registry number {registry_number}."
                )

            if initial_age + simulation_duration > max_age_in_dataframe:
                raise ValueError(
                    f"Initial {layer_name} stand age ({initial_age}) plus simulation "
                    f"duration ({simulation_duration:.1f} years) exceeds maximum age "
                    f"({max_age_in_dataframe}) in allometry file corresponding to registry number {registry_number}."
                )

    def dump_json_to_file(self, filepath: Path) -> None:
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))
