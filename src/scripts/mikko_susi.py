import numpy as np
import pandas as pd
import rasterio
import datetime
import xmltodict

from netCDF4 import Dataset
from susi.core.susi_main import Susi
from susi.core.susi_utils import read_FMI_weather
from susi.core.allometric_road_map import Growth_and_Yield_Table
from susi.thinning_models import calculate_thinning_recommendation
from scipy.optimize import root_scalar
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask
from pyproj import Transformer

# %% functions


def lidar_ditch_depth(ditch_depth_raster, coords, buffer_m=10):
    # Build Shapely polygon
    polygon = Polygon(coords)

    # Add buffer (10 meters by default)
    polygon_buffered = polygon.buffer(buffer_m)
    geojson_polygon = [mapping(polygon_buffered)]

    # Open raster and mask
    with rasterio.open(ditch_depth_raster) as src:
        out_image, out_transform = mask(src, geojson_polygon, crop=True)
        data = out_image[0].astype(float).flatten()

        # Replace nodata with NaN
        nodata = src.nodata
        data[data == nodata] = np.nan

        # Drop NaNs
        data = data[~np.isnan(data)]

        mean_value = data.mean()

    return (-1) * np.round(mean_value, decimals=2)


# Predict ditch shallowing by Hökkä et al. 2020, Baltic Forestry 26(2), article id 453. https://doi.org/10.46490/BF453
def predict_ditch_depth(drainage_age, peat_thickness=0.61, ditch_bed_slope=0.62):
    base = (
        49.139
        + 23.981 * np.power(drainage_age, -0.2)
        - 0.343 * drainage_age
        - 0.404 * drainage_age * peat_thickness
        + 17.961 * peat_thickness
        + 3.224 * ditch_bed_slope
    )
    if drainage_age > 35:
        base += 0.00134 * np.power(drainage_age, 2)
    return base


def estimate_drainage_age(ditch_depth_cm, peat_thickness=0.61, ditch_bed_slope=0.62):
    # Ditch depth difference at a given age
    def ditch_depth_difference(age):
        return (
            predict_ditch_depth(age, peat_thickness, ditch_bed_slope) - ditch_depth_cm
        )

    # Try solving in a reasonable range of drainage ages
    solution = root_scalar(ditch_depth_difference, bracket=[1, 100], method="brentq")

    if solution.converged:
        return np.round(solution.root)
    else:
        return None


def get_ditch_shallowing(ditch_depth, time=20):
    if ditch_depth <= -0.27:
        est_drainage_age = estimate_drainage_age((-1) * 100 * ditch_depth)
        ditch_shallowing = predict_ditch_depth(est_drainage_age + time)
        return np.round((-1) * 0.01 * ditch_shallowing, 2)
    else:
        return round(ditch_depth * 0.8, 2)


def sampling_stand_thinning_rate(species_id, stem_count):
    target_N = 1800 if species_id == 2 else 2000
    return target_N / stem_count


# Initialize SUSI
def initialize_SUSI(spara, ditch_depth, scen):
    spara["drain_age"] = 30.0
    mass_mor = (
        1.616 * np.log(spara["drain_age"]) - 1.409
    )  # Pitkänen et al. 2012 Forest Ecology and Management 284 (2012) 100–106
    spara["h_mor"] = mass_mor / spara["rho_mor"]

    if np.median(sfc) > 4:
        spara["peat type"] = ["S", "S", "S", "S", "S", "S", "S", "S"]
        spara["peat type bottom"] = ["A"]
        spara["vonP top"] = [2, 5, 5, 5, 6, 6, 7, 7]
        spara["anisotropy"] = 10
        spara["rho_mor"] = 80.0
    else:
        spara["vonP top"] = [2, 5, 5, 5, 6, 6, 7, 7]
        spara["anisotropy"] = 10
        spara["rho_mor"] = 85.0

    spara["depoN"] = 3.5  # Lestijärvi
    spara["depoP"] = 1.0  # Lestijärvi
    spara["depoK"] = 0.6  # Lestijärvi

    spara["ditch depth west"] = [ditch_depth]
    spara["ditch depth east"] = [ditch_depth]
    spara["ditch depth 20y west"] = [get_ditch_shallowing(ditch_depth, 20)]
    spara["ditch depth 20y east"] = [get_ditch_shallowing(ditch_depth, 20)]

    spara["scenario name"] = [scen]
    spara["cutting_yr"] = 2200  # out of the simulation period

    # Fertilized at the start year (scen == fertilization) or out of the simulation period
    spara["fertilization"]["application year"] = (
        start_yr if scen == "_fertilized" else 2200
    )

    # Partial blocking
    if scen == "_partialblocking":
        spara["ditch depth east"] = [-0.10]
        spara["ditch depth 20y east"] = [-0.10]

    return spara


# Output NetCDF4 files:
def get_ncf_outputs(file):
    ncf = Dataset(file, mode="r")

    hdom = list(np.mean(ncf["stand"]["hdom"][:, :, 1:-1], axis=2)[0])
    ba = list(np.mean(ncf["stand"]["basalarea"][:, :, 1:-1], axis=2)[0])
    vol = list(np.mean(ncf["stand"]["volume"][:, :, 1:-1], axis=2)[0])
    log_vol = list(np.mean(ncf["stand"]["logvolume"][:, :, 1:-1], axis=2)[0])
    pulp_vol = list(np.mean(ncf["stand"]["pulpvolume"][:, :, 1:-1], axis=2)[0])

    dwtyr_latesummer = list(
        np.mean(ncf["strip"]["dwtyr_latesummer"][:, 1:, 1:-1], axis=2)[0]
    )

    stand_litter = list(np.mean(ncf["balance"]["C"]["stand_litter_in"], axis=2)[0])
    gv_litter = list(
        np.mean(ncf["balance"]["C"]["gv_litter_in"][:, 1:, 1:-1], axis=2)[0]
    )
    co2c_release = list(
        np.mean(ncf["balance"]["C"]["co2c_release"][:, 1:, 1:-1], axis=2)[0]
    )
    ch4c_release = list(
        np.mean(ncf["balance"]["C"]["ch4c_release"][:, 1:, 1:-1], axis=2)[0]
    )
    LMW_to_water = list(np.mean(ncf["balance"]["C"]["LMWdoc_to_water"], axis=2)[0])
    LMW_to_atm = list(
        np.mean(ncf["balance"]["C"]["LMWdoc_to_atm"][:, 1:, 1:-1], axis=2)[0]
    )
    HMW_to_water = list(
        np.mean(ncf["balance"]["C"]["HMW_to_water"][:, 1:, 1:-1], axis=2)[0]
    )
    HMW_to_atm = list(
        np.mean(ncf["balance"]["C"]["HMW_to_atm"][:, 1:, 1:-1], axis=2)[0]
    )
    soil_C = list(
        np.mean(ncf["balance"]["C"]["soil_c_balance_c"][:, 1:, 1:-1], axis=2)[0]
    )

    stand_change = list(np.mean(ncf["balance"]["C"]["stand_change"], axis=2)[0])
    gv_change = list(np.mean(ncf["balance"]["C"]["gv_change"][:, 1:, 1:-1], axis=2)[0])
    ecosystem_C = list(
        np.mean(ncf["balance"]["C"]["stand_c_balance_c"][:, 1:, 1:-1], axis=2)[0]
    )

    soil_CO2eq = list(
        np.mean(ncf["balance"]["C"]["soil_c_balance_co2eq"][:, 1:, 1:-1], axis=2)[0]
    )
    ecosystem_CO2eq = list(
        np.mean(ncf["balance"]["C"]["stand_c_balance_co2eq"][:, 1:, 1:-1], axis=2)[0]
    )

    N_to_water = list(np.mean(ncf["balance"]["N"]["to_water"][:, 1:, 1:-1], axis=2)[0])
    P_to_water = list(np.mean(ncf["balance"]["P"]["to_water"][:, 1:, 1:-1], axis=2)[0])

    ncf.close()

    return {
        "hdom": hdom,
        "ba": ba,
        "vol": vol,
        "log_vol": log_vol,
        "pulp_vol": pulp_vol,
        "dwtyr_latesummer": dwtyr_latesummer,
        "stand_litter": stand_litter,
        "gv_litter": gv_litter,
        "co2c_release": co2c_release,
        "ch4c_release": ch4c_release,
        "LMW_to_water": LMW_to_water,
        "LMW_to_atm": LMW_to_atm,
        "HMW_to_water": HMW_to_water,
        "HMW_to_atm": HMW_to_atm,
        "stand_change": stand_change,
        "gv_change": gv_change,
        "soil_C": soil_C,
        "ecosystem_C": ecosystem_C,
        "soil_CO2eq": soil_CO2eq,
        "ecosystem_CO2eq": ecosystem_CO2eq,
        "N_to_water": N_to_water,
        "P_to_water": P_to_water,
    }


# %%

""" File names and locations """

# Forest data
area_name = "Paroninkorpi"
base_folder = f"C:/Users/mikniemi/OneDrive - University of Helsinki/AEMES - Documents/General/susi_2024-master-20251125/{area_name}/"
stands_path = f"{base_folder}Forest_data/{area_name}.xml"

# Ditch depth raster
ditch_depth_raster = f"{base_folder}Ditches/ditch_depth_1m.tif"

# Weather data
wpath = f"{base_folder}Weather_data/"
wdata_location = "Janakkala"
weather_file = f"{wpath}Weather_observations_{wdata_location}_1980_2024.csv"

# Read forest stands from xml file
with open(stands_path, encoding="utf8") as fd:
    forestdata = xmltodict.parse(fd.read())

stands = forestdata["ForestPropertyData"]["st:Stands"]

# Allometry files
allometry_files = f"{base_folder}Stand_allometry/"

# Output folder
folderName = f"{base_folder}susi_outputs/"

# Coordinate transformer ETRS-TM35FIN (EPSG:3067) -> YKJ (EPSG:2393)
transformer = Transformer.from_crs("EPSG:3067", "EPSG:2393", always_xy=True)

# Empty dictionary for saving drainage attributes
ditch_attributes = {}

# %%

""" SUSI Simulations """

# Loop stands
for stand in stands["st:Stand"]:
    StandNumber = int(stand["@id"])

    ## For testing purposes, skip other stands than the selected one
    # if 9 <= StandNumber <= 13:
    #    continue

    ############################################################################################
    # ***** At first, we generate allometric road map based on the forest inventory data ***** #
    ############################################################################################

    # Save stand basic and tree stratum data to new variables:
    StandBasicData = stand["st:StandBasicData"]
    polygon_str = StandBasicData["gdt:PolygonGeometry"]["gml:polygonProperty"][
        "gml:Polygon"
    ]["gml:exterior"]["gml:LinearRing"]["gml:coordinates"]
    coords = []
    for pair in polygon_str.strip().split(" "):
        if pair.strip() == "":
            continue
        x, y = pair.split(",")
        coords.append((float(x), float(y)))

    TreeStandData = stand["ts:TreeStandData"]
    TreeStandSummary = TreeStandData["ts:TreeStandDataDate"]["tss:TreeStandSummary"]
    try:
        TreeStratum = TreeStandData["ts:TreeStandDataDate"]["tst:TreeStrata"][
            "tst:TreeStratum"
        ]
    except:
        continue

    # Get stand attributes:
    MainGroup = int(StandBasicData["st:MainGroup"])
    SubGroup = int(StandBasicData["st:SubGroup"])
    FertilityClass = int(StandBasicData["st:FertilityClass"])
    SoilType = int(StandBasicData["st:SoilType"])

    # Assume all stands are peatlands
    peat = 1

    print()
    print("--- PROCESSING NEXT STAND ---")
    print()
    print(f"Stand number:      {StandNumber}")
    print(f"Main group:        {MainGroup}")
    print(f"Sub group:         {SubGroup}")
    print(f"Fertility class:   {FertilityClass}")
    print(f"Soil type:         {SoilType}")
    print()

    # Initialize variables
    age_1 = G_1 = N_1 = Dg_1 = Hg_1 = 0
    age_2 = G_2 = N_2 = Dg_2 = Hg_2 = 0
    age_3 = G_3 = N_3 = Dg_3 = Hg_3 = 0

    # Extract stratum attributes
    for stratum in TreeStratum:
        species = int(stratum["tst:TreeSpecies"])
        if species == 1:
            age_1 = int(stratum["tst:Age"])
            G_1 = float(stratum["tst:BasalArea"])
            N_1 = int(stratum["tst:StemCount"])
            Dg_1 = float(stratum["tst:MeanDiameter"])
            Hg_1 = float(stratum["tst:MeanHeight"])
        elif species == 2:
            age_2 = int(stratum["tst:Age"])
            G_2 = float(stratum["tst:BasalArea"])
            N_2 = int(stratum["tst:StemCount"])
            Dg_2 = float(stratum["tst:MeanDiameter"])
            Hg_2 = float(stratum["tst:MeanHeight"])
        elif species >= 3:
            age_3 = int(stratum["tst:Age"])
            G_3 = float(stratum["tst:BasalArea"])
            N_3 = int(stratum["tst:StemCount"])
            Dg_3 = float(stratum["tst:MeanDiameter"])
            Hg_3 = float(stratum["tst:MeanHeight"])

    G_values = {1: G_1, 2: G_2, 4: G_3}
    main_sp = max(G_values, key=G_values.get) if G_values else None
    N_total = N_1 + N_2 + N_3

    print(f"Main tree species: {main_sp}")
    print(f"Mean age:          {int(TreeStandSummary['tss:MeanAge'])} years")
    print(f"Basal area:        {float(TreeStandSummary['tss:BasalArea'])} m2/ha")
    print(f"Stem count:        {round(N_total)} trees/ha")
    print(f"Mean diameter:     {float(TreeStandSummary['tss:MeanDiameter'])} cm")
    print(f"Mean height:       {float(TreeStandSummary['tss:MeanHeight'])} m")
    print(f"Total volume:      {float(TreeStandSummary['tss:Volume'])} m3/ha")
    print()

    # Check the need of sapling stand thinning
    Dg_total = float(TreeStandSummary["tss:MeanDiameter"])
    if ((main_sp == 2) & (N_total > 2200) & (Dg_total < 8)) | (
        (main_sp != 2) & (N_total > 2500) & (Dg_total < 8)
    ):
        thinning_rate = sampling_stand_thinning_rate(main_sp, N_total)

        G_1 *= thinning_rate
        G_2 *= thinning_rate
        G_3 *= thinning_rate
        N_1 *= thinning_rate
        N_2 *= thinning_rate
        N_3 *= thinning_rate

        print("Sampling stand thinning is necessary!")
        print(f"--- Stem count decreased to {round(thinning_rate * N_total)}")
        print(f"--- Basal area decreased to {round(thinning_rate * sum(G_values), 1)}")
        print()

    # Number of reference trees per stratum
    n_trees = 20

    # Temperature sum, degree days
    DDY = 1300

    # Location in YKJ coordinates, and input variables x & y to sawlog reduction model
    ykj_e, ykj_n = transformer.transform(coords[0][0], coords[0][1])
    y = round(ykj_n / 1000)
    x = round(ykj_e / 10000)

    # Altitude above the sea level
    altitude = 123

    # Generate stand allometry
    gy = Growth_and_Yield_Table(
        age_1,
        G_1,
        N_1,
        Dg_1,
        Hg_1,
        age_2,
        G_2,
        N_2,
        Dg_2,
        Hg_2,
        age_3,
        G_3,
        N_3,
        Dg_3,
        Hg_3,
        DDY,
        FertilityClass,
        peat,
        y,
        x,
        altitude,
        n_trees,
    )
    susi_input = gy.get_table()

    # Write to Excel with two sheets
    page2 = pd.DataFrame(
        {
            "StandID": [1],
            "Schedule": [1],
            "Year": [0],
            "HarvestType": ["no_loggings"],
            "Species_id": [main_sp],
        }
    )

    with pd.ExcelWriter(
        f"{allometry_files}susi_input_{StandNumber}.xlsx", engine="xlsxwriter"
    ) as writer:
        susi_input.to_excel(writer, sheet_name="StandData", index=False)
        page2.to_excel(writer, sheet_name="Loggings", index=False)

    print(f"Allometric road map successfully generated for stand {StandNumber}")
    print()

    ##################################################################################
    # ********** Stand allometry is ready - let's go to the world of SUSI ********** #
    ##################################################################################

    # Drainage attributes:
    ditch_depth = lidar_ditch_depth(
        ditch_depth_raster, coords, buffer_m=10
    )  # initial ditch depth, m
    sarkaSim = 40.0  # strip width, i.e. distance between ditches, m
    n = int(sarkaSim / 2)  # number of computation nodes in the strip

    # Save ditch attributes:
    ditch_attributes[StandNumber] = {}
    ditch_attributes[StandNumber]["ditch_depth"] = ditch_depth
    ditch_attributes[StandNumber]["strip_width"] = sarkaSim

    # Stand allometry
    mottifile = {
        "path": allometry_files,
        "dominant": {1: f"susi_input_{StandNumber}.xlsx"},
        "subdominant": {0: "susi_motti_input_lyr_1.xlsx"},
        "under": {0: "susi_motti_input_lyr_2.xlsx"},
    }

    # Simulation period:
    start_date = datetime.datetime(2005, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    start_yr = start_date.year
    end_yr = end_date.year
    yrs = (end_date - start_date).days / 365.25

    print(
        f"Simulation period {start_yr}-{end_yr}. Initial ditch depth {ditch_depth} m, and after {end_yr - start_yr + 1} years {get_ditch_shallowing(ditch_depth, 20)} m."
    )
    print()

    # Read weather input
    forc = read_FMI_weather(0, start_date, end_date, sourcefile=weather_file)

    # Age of the stand in each node
    ageSim = {
        "dominant": float(
            pd.read_excel(f"{mottifile['path']}{mottifile['dominant'][1]}")["Age"][0]
        )
        * np.ones(n),
        "subdominant": 0 * np.ones(n),
        "under": 0 * np.ones(n),
    }

    # Site fertility class in each node
    sfc = np.ones(n, dtype=int) * FertilityClass

    site = "develop_scens"

    ### SET BASE SCENARIOS
    if ditch_depth > -0.40:
        base_scenarios = ["", "_fertilized", "_partialblocking", "_DNM"]
    else:
        base_scenarios = ["", "_fertilized", "_partialblocking"]

    for scen in base_scenarios:
        if scen == "_DNM":
            ditch_depth = -0.60

        wpara, cpara, org_para, spara, outpara, photopara = get_susi_para(
            wlocation="undefined",
            peat=site,
            folderName=folderName,
            hdomSim=None,
            ageSim=ageSim,
            sarkaSim=sarkaSim,
            sfc=sfc,
            n=n,
        )
        spara = initialize_SUSI(spara, ditch_depth, scen)

        outpara["netcdf"] = f"{area_name}_StandNumber_{StandNumber}{scen}.nc"

        print("#######################")
        print("###### CALL SUSI ######")
        print("#######################")
        print()

        susi = Susi()

        susi.run_susi(
            forc,
            wpara,
            cpara,
            org_para,
            spara,
            outpara,
            photopara,
            start_yr,
            end_yr,
            wlocation="undefined",
            mottifile=mottifile,
            peat="other",
            photosite="All data",
            folderName=folderName,
            ageSim=ageSim,
            sarkaSim=sarkaSim,
            sfc=sfc,
        )

        predictions = get_ncf_outputs(f"{folderName}{outpara['netcdf']}")

        hdom = predictions["hdom"]
        ba = predictions["ba"]

        # STUDY THINNING ALTERNATIVES:

        print()
        print("Studying thinning alternatives:")
        print()

        region = "Southern_Finland"
        soil = "Organic_soil"
        species = "pine" if G_1 >= G_2 else "spruce"
        if FertilityClass <= 2:
            species = "spruce"
        if FertilityClass >= 4:
            species = "pine"

        for yr in range(0, 20, 5):
            thinningGuidelines = calculate_thinning_recommendation(
                region, soil, FertilityClass, species, hdom[yr]
            )
            if thinningGuidelines == (None, None):
                print(
                    f"No thinning at year {yr}, as dominant height ({hdom[yr]:.1f} m) outside the thinning model range."
                )
            elif ba[yr] < thinningGuidelines[1]:
                print(
                    f"No thinning at year {yr}, as basal area ({ba[yr]:.1f} m2/ha) below the thinning limit ({thinningGuidelines[1]:.1f} m2/ha)."
                )
            else:
                print(
                    f"Thinning possible at year {yr}, basal area from {ba[yr]:.1f} m2/ha to {thinningGuidelines[0]:.1f} m2/ha."
                )
                print("----- CALL SUSI! -----")
                print()

                wpara, cpara, org_para, spara, outpara, photopara = get_susi_para(
                    wlocation="undefined",
                    peat=site,
                    folderName=folderName,
                    hdomSim=None,
                    ageSim=ageSim,
                    sarkaSim=sarkaSim,
                    sfc=sfc,
                    n=n,
                )
                spara = initialize_SUSI(spara, ditch_depth, scen)

                spara["cutting_yr"] = int(start_yr + yr)
                spara["cutting_to_ba"] = thinningGuidelines[0]
                spara["scenario name"] = [f"{scen}_thinning_at_yr_{yr}"]

                outpara["netcdf"] = (
                    f"{area_name}_StandNumber_{StandNumber}{scen}_thinning_yr_{yr}.nc"
                )

                susi.run_susi(
                    forc,
                    wpara,
                    cpara,
                    org_para,
                    spara,
                    outpara,
                    photopara,
                    start_yr,
                    end_yr,
                    wlocation="undefined",
                    mottifile=mottifile,
                    peat="other",
                    photosite="All data",
                    folderName=folderName,
                    ageSim=ageSim,
                    sarkaSim=sarkaSim,
                    sfc=sfc,
                )

# %%

""" Combine results to Excel from the existing ncf-files """

from pathlib import Path

folder_path = Path(folderName)

results = []

# Loop stands:
for stand in stands["st:Stand"]:
    StandNumber = int(stand["@id"])

    # Save stand basic to a new variable:
    StandBasicData = stand["st:StandBasicData"]

    # Loop through files containing stand number in their name:
    for file in folder_path.iterdir():
        if file.is_file() and (
            f"StandNumber_{StandNumber}_" in file.name
            or f"StandNumber_{StandNumber}." in file.name
        ):
            fertilized = "Ash" if "fertilized" in file.name else ""
            if "partialblocking" in file.name:
                ditch_management = "partial_blocking"
            elif "DNM" in file.name:
                ditch_management = "DNM_60_cm"
            else:
                ditch_management = ""
            if "thinning" in file.name:
                logging_type = "Thinning"
                logging_yr = int(file.name.split("_")[-1].split(".")[0])
            else:
                logging_type = ""
                logging_yr = ""

            mottifile = pd.read_excel(f"{allometry_files}susi_input_{StandNumber}.xlsx")

            predictions = get_ncf_outputs(f"{folderName}{file.name}")

            sim_result = {
                "Area": area_name,
                "StandNumber": StandNumber,
                "Fertilization": fertilized,
                "Ditch_management": ditch_management,
                "Logging": logging_type,
                "Logging_yr": logging_yr,
                "Initial_ditch_depth": ditch_attributes[StandNumber]["ditch_depth"],
                "Strip_width": ditch_attributes[StandNumber]["strip_width"],
                "MainGroup": int(StandBasicData["st:MainGroup"]),
                "SubGroup": int(StandBasicData["st:SubGroup"]),
                "FertilityClass": int(StandBasicData["st:FertilityClass"]),
                "SoilType": int(StandBasicData["st:SoilType"]),
                "MainSp": pd.read_excel(
                    f"{allometry_files}susi_input_{StandNumber}.xlsx",
                    sheet_name="Loggings",
                )["Species_id"][0],
                "MeanAge": mottifile["Age"][0],
                "BasalArea": np.round(mottifile["BA"][0], 1),
                "StemCount": mottifile["N"][0],
                "MeanDiameter": mottifile["Dg"][0],
                "MeanHeight": mottifile["Hg"][0],
                "Volume": np.round(mottifile["Volume"][0]),
                "Annual_vol_gr": np.round(
                    (predictions["vol"][20] - predictions["vol"][0]) / 20, 1
                ),  ## ADD HARVEST VOLUME
                "dwtyr_latesummer": np.round(
                    np.mean(predictions["dwtyr_latesummer"][1:]), 2
                ),
                "stand_litter": np.round(np.mean(predictions["stand_litter"][1:])),
                "gv_litter": np.round(np.mean(predictions["gv_litter"][1:])),
                "co2c_release": (-1)
                * np.round(np.mean(predictions["co2c_release"][1:])),
                "ch4c_release": (-1)
                * np.round(np.mean(predictions["ch4c_release"][1:]), 1),
                "LMW_to_water": (-1)
                * np.round(np.mean(predictions["LMW_to_water"][1:]), 1),
                "LMW_to_atm": (-1)
                * np.round(np.mean(predictions["LMW_to_atm"][1:]), 1),
                "HMW_to_water": (-1)
                * np.round(np.mean(predictions["HMW_to_water"][1:]), 1),
                "HMW_to_atm": (-1)
                * np.round(np.mean(predictions["HMW_to_atm"][1:]), 1),
                "soil_C_balance": np.round(np.mean(predictions["soil_C"][1:])),
                "stand_change": np.round(np.mean(predictions["stand_change"][1:])),
                "gv_change": np.round(np.mean(predictions["gv_change"][1:])),
                "ecos_C_balance": np.round(np.mean(predictions["ecosystem_C"][1:])),
                "soil_CO2eq": np.round(np.mean(predictions["soil_CO2eq"][1:])),
                "ecos_CO2eq": np.round(np.mean(predictions["ecosystem_CO2eq"][1:])),
                "N_to_water": np.round(np.mean(predictions["N_to_water"][1:]), 2),
                "P_to_water": np.round(np.mean(predictions["P_to_water"][1:]), 2),
            }
            results.append(sim_result)

# Convert list of dicts to DataFrame
df_results = pd.DataFrame(results)

df_results.to_excel(f"{base_folder}{area_name}_simulation_results.xlsx", index=False)
print()
print(f"Simulation results saved to {base_folder}{area_name}_simulation_results.xlsx")
