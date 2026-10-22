"""Dead code moved out of `src/susi` for review (#346).

Nothing imports this file and it doesn't have to run. It's a review queue: every item below was cut,
unchanged, from the `src/susi` file named in its header, because no entry point reaches it. Methods
became plain functions that keep their `self` parameter.

"Dead" means not reachable from `susi.run()`, the `[project.scripts]` tools, `src/analysis/`,
`example_projects/` or `test_projects/`. Tests don't keep code alive. Code that only other dead code
calls is dead too.

How to review an item: fill in its `Review:` line, then either
- restore it to its original file (and delete it here), or
- delete it here and add a line under `Removed` below naming the commit where it last lived, so git
  can bring it back.

The review is tracked in #352. When this file is empty, delete it and close #352 and #346.

Header fields:
- `From`: the original location, at the commit the item was moved from.
- `Dead because`: no callers, or only called by other dead code.
- `Reads SusiParams fields`: the parameter-object attributes the body reads. "(not a field any more)"
  means `SusiParams` (and its nested models) has no field of that name, so the code would fail if
  called.
"""

# Removed
# -------
# One line per item deleted after review: `<qualname> (<original file>) - last lived at <sha>`.
# (none yet)

# The imports below only let the moved code resolve its names for ruff; nothing runs this module.
import configparser

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d

from susi.core.canopygrid import e_sat

# ============================================================================
# src/susi/core/canopygrid.py
# ============================================================================


# --- CanopyGrid._degreeDays --------------------------------------------------
# From: src/susi/core/canopygrid.py:210-221 at ec9be39
# Dead because: no callers
# Review:
def _degreeDays(self, T, doy):
    """
    Calculates and updates degree-day sum from the current mean Tair.
    INPUT:
        T - daily mean temperature (degC)
        doy - day of year 1...366 (integer)
    """
    To = 5.0  # threshold temperature
    if doy == 1:  # reset in the beginning of the year
        self.DDsum = 0.0
    else:
        self.DDsum += np.maximum(0.0, T - To)


# --- CanopyGrid._lai_dynamics ------------------------------------------------
# From: src/susi/core/canopygrid.py:237-273 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: phenopara.ddo (not a field any more), phenopara.ddur (not a field any more), phenopara.lai_decid_min (not a field any more), phenopara.sdur (not a field any more), phenopara.sso (not a field any more)
# Review:
def _lai_dynamics(self, doy, phenopara):
    """
    Seasonal cycle of deciduous leaf area

    Args:
        self - object
        doy - day of year

    Returns:
        none, updates state variables self.LAIdecid, self._growth_stage,
        self._senec_stage
    """
    lai_min = phenopara.lai_decid_min
    ddo = phenopara.ddo
    ddur = phenopara.ddur
    sso = phenopara.sso
    sdur = phenopara.sdur

    # growth phase
    if self.DDsum <= ddo:
        f = lai_min
        self._growth_stage = 0.0
        self._senesc_stage = 0.0
    elif self.DDsum > ddo:
        self._growth_stage += 1.0 / ddur
        f = np.minimum(1.0, lai_min + (1.0 - lai_min) * self._growth_stage)

    # senescence phase
    if doy > sso:
        self._growth_stage = 0.0
        self._senesc_stage += 1.0 / sdur
        f = 1.0 - (1.0 - lai_min) * np.minimum(1.0, self._senesc_stage)

    # update self.LAIdecid and total LAI
    self._LAIdecid = self._LAIdecid_max * f
    self.LAI = self._LAIconif + self._LAIdecid
    return f


# --- degreeDays --------------------------------------------------------------
# From: src/susi/core/canopygrid.py:626-640 at ec9be39
# Dead because: no callers
# Review:
# @staticmethod
def degreeDays(dd0, T, Tbase, doy):
    """
    Calculates degree-day sum from the current mean Tair.
    INPUT:
        dd0 - previous degree-day sum (degC)
        T - daily mean temperature (degC)
        Tbase - base temperature at which accumulation starts (degC)
        doy - day of year 1...366 (integer)
    OUTPUT:
        x - degree-day sum (degC)
    """
    if doy == 1:  # reset in the beginning of the year
        dd0 = 0.0
    return dd0 + max(0, T - Tbase)


# --- eq_evap -----------------------------------------------------------------
# From: src/susi/core/canopygrid.py:643-671 at ec9be39
# Dead because: no callers
# Review:
# @staticmethod
def eq_evap(AE, T, P=101300.0, units="W"):
    """
    Calculates the equilibrium evaporation according to McNaughton & Spriggs,\
    1986.
    INPUT:
        AE - Available energy (Wm-2)
        T - air temperature (degC)
        P - pressure (Pa)
        units - W (Wm-2), mm (mms-1=kg m-2 s-1), mol (mol m-2 s-1)
    OUTPUT:
        equilibrium evaporation rate (Wm-2)
    """
    Mw = 18e-3  # kg mol-1
    # latent heat of vaporization of water [J/kg]
    L = 1e3 * (2500.8 - 2.36 * T + 1.6e-3 * T**2 - 6e-5 * T**3)
    # latent heat of sublimation [J/kg]
    if T < 0:
        L = 1e3 * (2834.1 - 0.29 * T - 0.004 * T**2)

    _, s, g = e_sat(T, P)

    x = np.divide((AE * s), (s + g))  # Wm-2 = Js-1m-2
    if units == "mm":
        x = x / L  # kg m-2 s-1 = mm s-1
    elif units == "mol":
        x = x / L / Mw  # mol m-2 s-1
    x = np.maximum(x, 0.0)
    return x


# --- wind_profile ------------------------------------------------------------
# From: src/susi/core/canopygrid.py:831-871 at ec9be39
# Dead because: no callers
# Review:
def wind_profile(LAI, hc, Uo, z, zm=2.0, zg=0.2):
    """
    Computes wind speed at ground height assuming logarithmic profile above and
    hyperbolic cosine profile within canopy
    INPUT:
        LAI - one-sided leaf-area /plant area index (m2m-2)
        hc - canopy height (m)
        Uo - mean wind speed at height zm (ms-1)
        zm - wind speed measurement height above canopy (m)
        zg - height above ground where U is computed
    OUTPUT:
        Uh - wind speed at hc (ms-1)
        Ug - wind speed at zg (ms-1)
    SOURCE:
       Cammalleri et al. 2010 Hydrol. Earth Syst. Sci
       Massman 1987, BLM 40, 179 - 197.
    """

    k = 0.4  # von Karman const
    Cd = 0.2  # drag coeff
    alpha = 1.5  # (-)

    zm = zm + hc
    d = 0.66 * hc
    zom = 0.123 * hc
    beta = 4.0 * Cd * LAI / (k**2.0 * alpha**2.0)
    # solve ustar and U(hc) from log-profile above canopy
    ustar = Uo * k / np.log((zm - d) / zom)  # m/s

    U = np.ones(len(z)) * np.NaN

    # above canopy top wind profile is logarithmic
    U[z >= hc] = ustar / k * np.log((z[z >= hc] - d) / zom)

    # at canopy top, match log and exponential profiles
    Uh = ustar / k * np.log((hc - d) / zom)  # m/s

    # within canopy hyperbolic cosine profile
    U[z <= hc] = Uh * (np.cosh(beta * z[z <= hc] / hc) / np.cosh(beta)) ** 0.5

    return U, ustar, Uh


# --- daylength ---------------------------------------------------------------
# From: src/susi/core/canopygrid.py:874-905 at ec9be39
# Dead because: no callers
# Review:
def daylength(LAT, LON, DOY):
    """
    Computes daylength from location and day of year.

    Args:
        LAT, LON - in deg, float or arrays of floats
        doy - day of year, float or arrays of floats

    Returns:
        dl - daylength (hours), float or arrays of floats
    """
    CF = np.pi / 180.0  # conversion deg -->rad

    LAT = LAT * CF
    LON = LON * CF

    # ---> compute declination angle
    xx = 278.97 + 0.9856 * DOY + 1.9165 * np.sin((356.6 + 0.9856 * DOY) * CF)
    DECL = np.arcsin(0.39785 * np.sin(xx * CF))
    del xx

    # --- compute day length, the period when sun is above horizon
    # i.e. neglects civil twilight conditions
    cosZEN = 0.0
    dl = (
        2.0
        * np.arccos(cosZEN - np.sin(LAT) * np.sin(DECL) / (np.cos(LAT) * np.cos(DECL)))
        / CF
        / 15.0
    )  # hours

    return dl


# --- read_ini ----------------------------------------------------------------
# From: src/susi/core/canopygrid.py:908-930 at ec9be39
# Dead because: no callers
# Review:
def read_ini(inifile):
    """read_ini(inifile): reads canopygrid.ini parameter file into pp dict"""

    cfg = configparser.ConfigParser()
    cfg.read(inifile)

    pp = {}
    for s in cfg.sections():
        section = s.encode("ascii", "ignore")
        pp[section] = {}
        for k, v in cfg.items(section):
            key = k.encode("ascii", "ignore")
            val = v.encode("ascii", "ignore")
            if section == "General":  # 'general' section
                pp[section][key] = val
            else:
                pp[section][key] = float(val)

    pp["General"]["dt"] = float(pp["General"]["dt"])

    pgen = pp["General"]
    cpara = pp["CanopyGrid"]
    return pgen, cpara


# ============================================================================
# src/susi/core/strip.py
# ============================================================================


# --- StripHydrology.boundNoFlow ----------------------------------------------
# From: src/susi/core/strip.py:214-224 at ec9be39
# Dead because: no callers
# Review:
def boundNoFlow(A, n, implic, Trwest, Treast, alfa):
    """
    Diriclet (constant head boundary conditions)
    """

    A[0, 0] = 2.0 * implic * (Treast[0]) + alfa[0]  # Diagonal element
    A[0, 1] = -2 * implic * Trwest[0]  # East element
    A[n - 1, n - 1] = 2.0 * implic * (Trwest[n - 1]) + alfa[0]
    A[n - 1, n - 2] = -2 * implic * Treast[n - 1]  # West element

    return A


# ============================================================================
# src/susi/core/susi_utils.py
# ============================================================================


# --- z_distrib_decomposition -------------------------------------------------
# From: src/susi/core/susi_utils.py:282-292 at ec9be39
# Dead because: no callers
# Review:
def z_distrib_decomposition(
    gwl, Qdistrib_beta=0.93
):  # Alkuperäinen 0.88, Koukkujoki 0.87
    """
     Distribution Gale & Grigal Canadian Journal of Forest Research, 1987, 17(8): 829-834, 10.1139/x87-131  \n
     Input:
         gwl m
    Output:
         share of actual decomposition from the potential
    """
    return Qdistrib_beta**0.0 - Qdistrib_beta ** (np.abs(gwl * 100.0))


# --- potential_peat_heterotrophic_respiration --------------------------------
# From: src/susi/core/susi_utils.py:295-347 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: spara['sfc'] (not a field any more)
# Review:
def potential_peat_heterotrophic_respiration(T, sfc, V):
    """
    Returns potential soil oxygen consumption through soil respiration (CO2) flux through the soil surface according to Ojanen et al 2010 ForEco, eq1
    Boundary conditions: max soil temperature 16 deg C. No minimum value. \n
    Input:
        Tsoil soil temperature in 5 cm depth, deg C \n
        spara site and soil parameters
    Output:
        Potential O2 consumption flux to soil as kg m-2 s-1, refers to level demanded by total respiration \n
        Total respiration: Potential CO2 efflux from soil as kg m-2 s-1 \n
        Heterotrophic respiration, CO2 efflux caused by decomposition: Potential CO2 efflux from soil as kg m-2 s-1 \n
    """
    # sfc=spara['sfc']

    # Ojanen et al. 2010 Forest Ecology and Management 260:411-421
    T = min(T, 16.0)
    # Computes momentary Total CO2 flux as a function of soil temperature and peat bulk density
    B = 350.0
    T5ref = 10.0
    T50 = -46.02
    #   Huom Ojanen 2010 B=350. Table 2
    # Adjusted with bulk density
    wtm = 80.0  # wtm mean water table in cm May-Oct, maximum 80 in the Ojanen data, max used becase later cut with the current wt
    pt = 99.0
    # -----properties according to soil fertility class sfc:                                                                  #peat depth cm
    bd = {
        2: 0.14,
        3: 0.11,
        4: 0.10,
        5: 0.08,
    }  # Mese study: bulk densities in different fertility classes                                                                 # peat layer thickness, cm

    wtm = 50.0  # Huikari & Paarlahti 1968 Comm Inst For Fenn 64.1, Max growth at DWT 50 cm
    Rref = (
        0.28 + V * 5.9 * 10**-4 + bd[sfc] * 1000.0 * 9.1 * 10**-4 + wtm * 1.5 * 10**-3
    )  # Parameters: Table 3 RTot
    TotCO2 = Rref * np.exp(B * (1.0 / (T5ref - T50) - 1.0 / (T - T50)))  # g m-2 h-1
    TotCO2 = TotCO2 / 1000.0 / 3600.0  # Conversion to kg m-2 s-1
    TotCO2 / 1.375  # Conversion from CO2 to O2 with mole mass ratio

    # Computes momentary Heterotrophic CO2 flux as a function of soil temperature and peat bulk density
    Rref = (
        0.0695 + V * 3.7 * 10**-4 + bd[sfc] * 1000.0 * 5.4 * 10**-4 + wtm * 1.2 * 10**-3
    )  # Parameters: Table 3 RHet
    TASmr = 10.4  # mean temperature may-october, deg C
    pt = 99.0  # peat layer thickness, cm
    B = (
        156.032 + 16.5 * TASmr - 0.196 * pt + 0.354 * bd[sfc]
    )  # Ojanen 2010 et al. Table 4
    HetCO2 = Rref * np.exp(B * (1.0 / (T5ref - T50) - 1.0 / (T - T50)))  # g m-2 h-1
    HetCO2 = HetCO2 / 1000.0 / 3600.0  # Conversion to kg m-2 s-1
    # return: final converstaion into kg/ha/day
    return HetCO2 * 86400.0 * 10000.0


# --- nutrient_release --------------------------------------------------------
# From: src/susi/core/susi_utils.py:456-507 at ec9be39
# Dead because: no callers
# Review:
def nutrient_release(sfc, sfc_specification, co2release, N=None, P=None, K=None):
    # sfc                                                                         # soil fertility class
    # sfc_specification    1 (Myrtillus I and Vaccinium I) 2 (Myrtillus II, Vaccinium II)
    Nd = {
        2: {1: 1.9, 2: 1.9},
        3: {1: 1.6, 2: 1.6},
        4: {1: 1.4, 2: 1.4},
        5: {1: 1.2, 2: 1.2},
    }  # Mese study: N cont in OM % dm
    Pd = {
        2: {1: 0.1, 2: 0.1},
        3: {1: 0.08, 2: 0.08},
        4: {1: 0.06, 2: 0.06},
        5: {1: 0.05, 2: 0.05},
    }  # Mese study: P cont in OM % dm
    Kd = {
        2: {1: 0.045, 2: 0.045},
        3: {1: 0.04, 2: 0.038},
        4: {1: 0.037, 2: 0.034},
        5: {1: 0.03, 2: 0.03},
    }  # Mese study: P cont in OM % dm (alkuperäinen)

    # Kd =  {2: {1:0.045, 2:0.041}, 3: {1:0.04, 2:0.38}, 4: {1:0.035, 2:0.035}, 5: {1:0.03, 2:0.03}}                                 # Mese study: P cont in OM % dm (alkuperäinen)

    # Ndeposition = 5.                            #2-6 kg/ha/yr    Palviainen & Finer 2012
    # Pdeposition = 0.12                          #0.04-0.25
    # Kdeposition = 0.5                           #0.3-0.6 kg/ha/yr

    if N is None:
        N = Nd[sfc][sfc_specification]
    if P is None:
        P = Pd[sfc][sfc_specification]
    if K is None:
        K = Kd[sfc][sfc_specification]

    C_in_OM = 0.55  # C content in OM kg kg-1
    CO2_to_C = 12.0 / 44.0
    Nmicrob = 0.8  # microbial immobilisation
    Pmicrob = 0.7
    Kmicrob = 0.0  # microbial immobilisation

    Nt = (
        co2release * CO2_to_C / C_in_OM * N / 100.0 * (1.0 - Nmicrob)
    )  # + Ndeposition   # Nrelease kg ha-1 day-1
    Pt = (
        co2release * CO2_to_C / C_in_OM * P / 100.0 * (1.0 - Pmicrob)
    )  # +  Pdeposition # Prelease kg ha-1 day-1
    Kt = (
        co2release * CO2_to_C / C_in_OM * K / 100.0 * (1.0 - Kmicrob)
    )  # + Kdeposition  # Prelease kg ha-1 day-1

    return Nt, Pt, Kt


# --- diff_nutrient_release ---------------------------------------------------
# From: src/susi/core/susi_utils.py:510-704 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: spara.scenario_name, spara.sfc (not a field any more), spara.sfc_specification
# Review:
def diff_nutrient_release(
    co2release,
    vol,
    spara,
    wpara,
    outpara,
    dates,
    wlocation,
    summer_dwt,
    co2_respi,
    ditch_depth,
    rounds,
    yi,
    bm,
    npps,
    bmToVol,
    NtoVol,
    PtoVol,
    KtoVol,
):
    import susi_io

    """
    Computes nutrient release in each drain depth scenario, calculates the difference between the scenarios, and extends \n
    the difference into growth responses.
    Input:
        - co2release: daily mean value (accross the nodes in strip) kg/ha/day
        - vol: stand volume m3/ha daily basis
        - spara soil parameters
        - scen scenario name
    """
    gr_response = []
    sfc = spara.sfc
    sfc_specification = spara.sfc_specification
    N = {
        2: {1: 1.9, 2: 1.9},
        3: {1: 1.6, 2: 1.6},
        4: {1: 1.4, 2: 1.4},
        5: {1: 1.2, 2: 1.2},
    }  # Mese study: N cont in OM % dm
    P = {
        2: {1: 0.1, 2: 0.1},
        3: {1: 0.08, 2: 0.08},
        4: {1: 0.06, 2: 0.06},
        5: {1: 0.05, 2: 0.05},
    }  # Mese study: P cont in OM % dm
    K = {
        2: {1: 0.045, 2: 0.045},
        3: {1: 0.04, 2: 0.038},
        4: {1: 0.037, 2: 0.034},
        5: {1: 0.03, 2: 0.03},
    }  # Mese study: P cont in OM % dm (alkuperäinen)
    C_in_OM = 0.55  # C content in OM kg kg-1
    CO2_to_C = 12.0 / 44.0
    Nmicrob = 0.8  # microbial immobilisation
    Pmicrob = 0.7
    Kmicrob = 0.0  # microbial immobilisation

    print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
    for c in range(np.shape(co2release)[0]):
        co2 = co2release[c, :]
        Nt = (
            co2
            * CO2_to_C
            / C_in_OM
            * N[sfc][sfc_specification]
            / 100.0
            * (1.0 - Nmicrob)
        )  # Nrelease kg ha-1 day-1
        Pt = (
            co2
            * CO2_to_C
            / C_in_OM
            * P[sfc][sfc_specification]
            / 100.0
            * (1.0 - Pmicrob)
        )  # Prelease kg ha-1 day-1
        Kt = (
            co2
            * CO2_to_C
            / C_in_OM
            * K[sfc][sfc_specification]
            / 100.0
            * (1.0 - Kmicrob)
        )  # Prelease kg ha-1 day-1
        print(
            "Released nutrients N",
            np.round(sum(Nt), 2),
            "P",
            np.round(sum(Pt), 2),
            "K",
            np.round(sum(Kt), 2),
        )
        gN = NtoVol(sum(Nt))
        gP = PtoVol(sum(Pt))
        gK = KtoVol(sum(Kt))
        print("Growths", np.round(gN, 2), np.round(gP, 2), np.round(gK, 2))

    print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")

    print("***************************************")
    print("Comparing scenarios:")
    comparisons = np.shape(co2release)[0]
    vol[0] + (yi[-1] - yi[0])
    for c in range(comparisons)[1:]:
        diff = co2release[c] - co2release[0]
        Nrel = (
            diff * CO2_to_C / C_in_OM * N[sfc] / 100.0 * (1.0 - Nmicrob)
        )  # Nrelease kg ha-1 day-1
        Prel = (
            diff * CO2_to_C / C_in_OM * P[sfc] / 100.0 * (1.0 - Pmicrob)
        )  # Prelease kg ha-1 day-1
        Krel = (
            diff * CO2_to_C / C_in_OM * K[sfc] / 100.0 * (1.0 - Kmicrob)
        )  # Prelease kg ha-1 day-1

        yrs = (dates[-1] - dates[0]).days / 365.0

        tg = vol[-1] - vol[0]
        ag = bmToVol(bm[0] + sum(npps[c, :])) - bmToVol(bm[0])
        print("  +Control vs", spara.scenario_name[c])
        print(
            "    -Nutrient release:",
            "N",
            np.round(sum(Nrel), 3),
            "P",
            np.round(sum(Prel), 3),
            "K",
            np.round(sum(Krel), 3),
            "kg/ha",
        )
        print(
            "    -Volume control",
            np.round(vol[-1], 1),
            "increment: min (",
            np.round(gN, 2),
            np.round(gP, 2),
            np.round(gK, 2),
            ")",
        )
        print(
            "    -Increment per yr: min(",
            np.round(gN / yrs, 1),
            np.round(gP / yrs, 1),
            np.round(gK / yrs, 1),
            ")",
        )
        print("    -Years in simulation", np.round(yrs))
        print("    -Table growth", np.round(tg, 2), "m3/ha")
        print("    -Assimilation growth", np.round(ag, 2), "m3/ha")
        print("    -Difference", np.round(ag - tg, 2), "m3/ha")
        print("    -Table max growth", np.round(tg * 0.15), "m3/ha")

        limiting = min(gN, gP, gK)
        gr_response.append(limiting)

        if c == range(comparisons)[-1]:
            gr_response = np.ravel(gr_response)
            summer_dwt = np.array(summer_dwt)
            co2_respi = np.array(co2_respi)
            gr_response = np.array(gr_response)
            ditch_depth = np.array(ditch_depth)
            gr_response = np.insert(gr_response, 0, 0.0)
            fwt = interp1d(
                summer_dwt, ditch_depth, bounds_error=False, fill_value="extrapolate"
            )
            fgr = interp1d(
                ditch_depth, gr_response, bounds_error=False, fill_value="extrapolate"
            )
            cr_depth = fwt(0.35)
            gr_crd = fgr(fwt(0.35))
        else:
            cr_depth = 0
            gr_crd = 0
        susi_io.write_gr_excel(
            wlocation, wpara, spara, outpara, gN, gP, gK, c, cr_depth, gr_crd
        )

        po = False
        if po:
            fig = plt.figure(
                num="Control vs" + spara.scenario_name[c],
                facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
                edgecolor="k",
                figsize=(18.0, 12.0),
            )  # Figsize(w,h), tuple inches
            fig.add_axes([0.05, 0.5, 0.55, 0.46])  # left, bottom, width, height
            plt.ylabel("stand volume")
            plt.plot(dates, vol, "k-")
            plt.plot(dates, vol + np.cumsum(diff) / sum(diff) * limiting, "r-")
            plt.fill_between(
                dates, vol, vol + np.cumsum(Krel * gK), color="red", alpha=0.5
            )

    return gr_response


# --- critical_ditch_depth ----------------------------------------------------
# From: src/susi/core/susi_utils.py:707-714 at ec9be39
# Dead because: no callers
# Review:
def critical_ditch_depth(summer_dwt, ditch_depth, growth_response):
    fwt = interp1d(
        summer_dwt, ditch_depth, bounds_error=False, fill_value="extrapolate"
    )
    fgr = interp1d(
        ditch_depth, growth_response, bounds_error=False, fill_value="extrapolate"
    )
    return fwt(-0.35), fgr(fwt(-0.35))


# --- get_mese_input ----------------------------------------------------------
# From: src/susi/core/susi_utils.py:717-753 at ec9be39
# Dead because: no callers
# Review:
def get_mese_input(ifile):
    df = pd.read_excel(ifile, sheet_name=0, usecols=range(28), skiprows=1, header=None)
    cnames = [
        "nro",
        "koe",
        "kaista",
        "ddist",
        "age_ini",
        "stripw",
        "dd_west",
        "dd_east",
        "sfc",
        "bd",
        "n_mg/g",
        "p_mg/g",
        "k_mg/g",
        "V_tot5",
        "wfile",
        "Vtot_0",
        "Ntot_0",
        "H_0	",
        "iv_tot5",
        "wtmax",
        "wtmed",
        "wtmin",
        "Ndem",
        "Pdem",
        "Kdem",
        "depoN",
        "depoP",
        "depoK",
    ]
    df = pd.read_excel(ifile, sheet_name=0, usecols=range(28), skiprows=1, header=None)
    df.columns = cnames
    df = pd.DataFrame(df).set_index("nro")

    return df


# --- get_mese_out ------------------------------------------------------------
# From: src/susi/core/susi_utils.py:756-775 at ec9be39
# Dead because: no callers
# Review:
def get_mese_out(ifile):
    df = pd.read_excel(ifile, sheet_name=0, usecols=range(24), skiprows=1, header=None)
    cnames = [
        "nro",
        "v_ini",
        "v",
        "iv5",
        "Nsup",
        "Psup",
        "Ksup",
        "Crel",
        "dwt_loc",
        "cb",
        "cbt",
        "sfc",
    ]
    df = pd.read_excel(ifile, sheet_name=0, usecols=range(10), skiprows=1, header=None)
    df.columns = cnames
    df = pd.DataFrame(df).set_index("nro")
    return df


# --- get_mese_scen -----------------------------------------------------------
# From: src/susi/core/susi_utils.py:778-789 at ec9be39
# Dead because: no callers
# Review:
def get_mese_scen(ifile):
    # df = pd.read_excel(ifile, sheetname=0, parse_cols=range(26), skiprows=1, header=None )
    # cnames = ['nro', 'v_ini', 'v_end_30', 'v_end_50','v_end_70', 'v_end_90', 'dg_30', 'dg_50', 'dg_70', 'dg_90', 'cb_30',
    #          'cb_50', 'cb_70', 'cb_90', 'dcb_30', 'dcb_50', 'dcb_70', 'dcb_90', 'wt_30', 'wt_50', 'wt_70', 'wt_90',
    #          'dwt_30', 'dwt_50', 'dwt_70', 'dwt_90', 'sfc', 'koe', 'dead_w','cbn']
    # df = pd.read_excel(ifile, sheetname=0, parse_cols=range(30), skiprows=1, header=None )
    # df.columns=cnames

    df = pd.read_excel(ifile, sheet_name=0)

    df = pd.DataFrame(df).set_index("nro")
    return df


# --- nut_to_vol --------------------------------------------------------------
# From: src/susi/core/susi_utils.py:792-898 at ec9be39
# Dead because: no callers
# Review:
def nut_to_vol(
    vol_ini,
    Nrel,
    Prel,
    Krel,
    litter_mass,
    gvNup,
    gvPup,
    gvKup,
    leafmass,
    gv_leafmass,
    species="Pine",
):
    """
    Input:
        vol_ini -  initial volume m3 ha-1
        Nrel, Prel, Krel - available mass of N, P, K (release from decomp + deposition) kg ha-1
        litter_mass - litter mass within the time step. kg ha-1
        gvNup, gvPup, gvKup - uptake of N P and K by ground vegetation kg ha-1 in time step
        leafmass, gv_leafmass  - tree leafmass, ground vegetation leafmass kg ha-1
        species - main tree species: pine, spruce, birch (other deciduous)
    Output
        volN, volP, volK - new volumes allowed by nutrient release
    """
    # gvshare = np.mean(gv_leafmass / (leafmass + gv_leafmass))     # maximum share of ground vegetation from the nutrient uptake -> shared in proportion of green biomass
    gvshare = (
        gv_leafmass / (leafmass + gv_leafmass)
    )  # maximum share of ground vegetation from the nutrient uptake -> shared in proportion of green biomass

    littershare = (
        1.0 - gvshare
    ) / 2.0  # maximum share of nutrient supply allocated for litter

    def MarjoNut(vol, lna, b, k):
        return np.exp(lna + b * np.log(vol) + k)

    def MarjoBck(nut, lna, b, k):
        return np.exp((np.log(nut) - lna - k) / b)

    #'N':{'pine':[1.856,0.631,0.050]},
    #'P':{'pine':[-2.387,0.754,0.158]},
    #'K':{'pine':[0.839,0.650,0.070]}

    par = {  # Palviainen & Finer, 2012 Eur J For Res 131:945-964, eq 2, Table 7
        "N": {
            "Pine": [1.856, 0.631, 0.050],
            "Spruce": [2.864, 0.557, 0.051],
            "Birch": [1.590, 0.788, 0.044],
        },
        "P": {
            "Pine": [-0.202, 0.566, 0.115],
            "Spruce": [1.109, 0.461, 0.051],
            "Birch": [-1.034, 0.838, 0.162],
        },
        "K": {
            "Pine": [0.839, 0.650, 0.078],
            "Spruce": [2.487, 0.478, 0.061],
            "Birch": [-0.811, 1.087, 0.055],
        },
    }

    # litter_nut={'N':12.0, 'P':1.0, 'K':3.5}      #mg/g Laiho 1997 page 49
    litter_nut = {
        "Pine": {"N": 6.08, "P": 0.63, "K": 2.29},  # mg/g Palviainen & Finer 2012
        "Spruce": {"N": 6.9, "P": 0.81, "K": 2.81},
        "Birch": {"N": 9.7, "P": 0.86, "K": 3.08},
    }

    # retrans = {'N': 0.69, 'P': 0.73, 'K':0.8}     #Nieminen Helmisaari 1996 Tree Phys
    retrans = {"N": 0.69, "P": 0.73, "K": 0.84}

    litterN = litter_mass * (1.0 - retrans["N"]) * litter_nut[species]["N"] / 1000.0
    litterP = litter_mass * (1.0 - retrans["P"]) * litter_nut[species]["P"] / 1000.0
    litterK = litter_mass * (1.0 - retrans["K"]) * litter_nut[species]["K"] / 1000.0

    gvNup = np.minimum(gvNup, gvshare * (Nrel))
    gvPup = np.minimum(gvPup, gvshare * (Prel))
    gvKup = np.minimum(gvKup, gvshare * (Krel))

    litterN = np.minimum(litterN, littershare * (Nrel))
    litterP = np.minimum(litterP, littershare * (Prel))
    litterK = np.minimum(litterK, littershare * (Krel))

    # print ('nut bal ***************************')
    # print (np.mean(Krel), np.mean(litterK), np.mean(gvKup))
    # print ('****************************')

    # ******** Nutrients in the stand **************
    lna, b, k = par["N"][species]
    Ns = MarjoNut(vol_ini, lna, b, k)
    lna, b, k = par["P"][species]
    Ps = MarjoNut(vol_ini, lna, b, k)
    lna, b, k = par["K"][species]
    Ks = MarjoNut(vol_ini, lna, b, k)

    lna, b, k = par["N"][species]
    volN = MarjoBck(Ns + Nrel - litterN - gvNup, lna, b, k)
    lna, b, k = par["P"][species]
    volP = MarjoBck(Ps + Prel - litterP - gvPup, lna, b, k)
    lna, b, k = par["K"][species]
    volK = MarjoBck(Ks + Krel - litterK - gvKup, lna, b, k)

    volN = np.maximum(volN, vol_ini)
    volP = np.maximum(volP, vol_ini)
    volK = np.maximum(volK, vol_ini)

    return volN, volP, volK


# --- nutrient_demand ---------------------------------------------------------
# From: src/susi/core/susi_utils.py:901-942 at ec9be39
# Dead because: no callers
# Review:
def nutrient_demand(vol_ini, vol, litter_mass):
    # vol as list: start, end, leaf_mass kg/ha at the end, yrs is length of the study period
    # this is not in use

    def MarjoNut(vol, lna, b, k):
        return np.exp(lna + b * np.log(vol) + k)

    from_abovegr_to_total = 1.2

    litter_nut = {"N": 12.0, "P": 1.0, "K": 3.5}  # mg/g Laiho 1997 page 49
    # litter_nut={'N':6.9, 'P':0.81, 'K':2.81}      #mg/g Palviainen & Finer 2012
    retrans = {"N": 0.69, "P": 0.73, "K": 0.8}  # Nieminen Helmisaari 1996 Tree Phys
    # longevityLeaves = {'pine':4., 'spruce':5.} #yrs, Lamppu & Huttunen 2001 CJFR life span of leaves and fine roots

    par = {  # Palviainen & Finer, 2012 Eur J For Res 131:945-964, eq 2, Table 7
        "N": {"pine": [1.856, 0.631, 0.050]},
        "P": {"pine": [-2.387, 0.754, 0.158]},
        "K": {"pine": [0.839, 0.650, 0.070]},
    }

    lna, b, k = par["N"]["pine"]  # pine
    N = (
        MarjoNut(vol, lna, b, k) - MarjoNut(vol_ini, lna, b, k)
    ) * from_abovegr_to_total + litter_mass * (1.0 - retrans["N"]) * litter_nut[
        "N"
    ] / 1000.0

    lna, b, k = par["P"]["pine"]  # pine
    P = (
        MarjoNut(vol, lna, b, k) - MarjoNut(vol_ini, lna, b, k)
    ) * from_abovegr_to_total + litter_mass * (1.0 - retrans["P"]) * litter_nut[
        "P"
    ] / 1000.0

    lna, b, k = par["K"]["pine"]  # pine
    K = (
        MarjoNut(vol, lna, b, k) - MarjoNut(vol_ini, lna, b, k)
    ) * from_abovegr_to_total + litter_mass * (1.0 - retrans["K"]) * litter_nut[
        "P"
    ] / 1000.0

    return N, P, K


# --- rewFloor ----------------------------------------------------------------
# From: src/susi/core/susi_utils.py:957-970 at ec9be39
# Dead because: no callers
# Review:
def rewFloor(dwt, LAI):  # Leenan testi
    # Restriction for efloor: similar to rew for transpiration but the restriction increases with decreasing LAI (3...1.5)
    LAIarray = np.array([0, 1.5, 3, 5])
    rajKerroin = np.array([1, 1, 1, 1])

    fLAI = interp1d(LAIarray, rajKerroin, fill_value="extrapolate")

    wt = np.array([-150.0, -1.0, -0.5, -0.3, -0.1, 0.0])  # water table in m

    re = fLAI(LAI) * np.array([1, 1, 1, 1, 1, 1])

    frew = interp1d(wt, re, fill_value="extrapolate")

    return frew(dwt)


# --- assimilation ------------------------------------------------------------
# From: src/susi/core/susi_utils.py:984-1050 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: photopara.Smax, photopara.X0, photopara.alfa, photopara.beta, photopara.gamma, photopara.kappa, photopara.nu, photopara.tau
# Review:
def assimilation(photopara, rg, vpd, Ta_minus1, Ta, rew, LAI, Xk, Ns, Ps, Ks, hdom):
    """
    Computes photosynthesis and respiration of the stand in daily time step
    Mäkelä et al. 2008. Empirical model of stand GPP LUE approach. Global Change Biology 14: 92-108
    IN:
        (develpment of all sided leaf area index (LAI) m2 m-2)
        meteorological input data:
        rg in W/m2
        vpd in kPa
        Ta previous day tamperature
        Ta  in deg C
        Xk acclimatisation temperature, Ta in the first time step
        Ns,Ps,Ks nutrient release in time step kg/ha/timestep
        Nconc, Pconc, Kconc nutrient concentration in stand
        hdom dominant height of the stand (used in computation of respiration)
    OUT:
        Net primary production NPP in kg/ha/yr
        Xa
    """
    par = (
        rg * 4.6 * 0.5 / 1000000.0 * 86400.0
    )  # Unit conversion to mol/m2/day, 0.5 is the share of par from rg

    """ Eq 2 """
    fL = 1.0 / (photopara.gamma * par + 1.0)

    """Eq. 3a and 3b"""
    Xk = Xk + (Ta - Xk) / photopara.tau
    Sk = np.maximum(Xk - photopara.X0, 0.0)

    """Eq 4"""
    fs = np.minimum(Sk / photopara.Smax, 1.0)  # Final temperature adustment

    """Eq 5"""  # Vapor pressure deficit function: limits photos when stomata are closed
    fd = np.exp(photopara.kappa * vpd)  # Eq 5

    """Eq 6"""  # Soil water content modifyer
    fREW = 1.0 / (1.0 + ((1.0 - rew) / photopara.alfa) ** photopara.nu)

    """ Beer-Lambert function: LAI function"""  # Fabrika 2013
    kext = 0.2

    """Eq 1"""
    # LAI = LAI*11./6.4     #to all sided LAI Mäkelä et al. 2001 (LAI 11)
    Pk = (
        photopara.beta * (1.0 - np.exp(-kext * LAI)) * par * fL * fs * fd * fREW
    )  # canopy GPP gC m-2 day-1
    """Respi"""
    # Conversion micro mol/m2/s ->g/m2/day
    # Rtrees= np.maximum((r0*q10**((Ta-10.)/10.)-cr)*rlai, np.zeros(sim_len))
    # Rtrees = 0.47*Pk  #Waring et al. 1998, Tree Physiology 18, 129-132
    fr = interp1d(
        np.array(
            [
                0.0,
                10.0,
                20.0,
                30.0,
            ]
        ),
        np.array([0.6, 0.50, 0.36, 0.20]),
        fill_value="extrapolate",
    )  # Mäkelä & Valentine 2001 Tree Physiology 21, 1015-1030, Härkönen et al. 2010 ForEco 259, 524-533

    NPP = Pk * fr(hdom) * 2.0  # Change from gC m-2 to g organic matter

    return NPP * 10.0  # kg/ha organic matter


# --- heterotrophic_respiration_yr_bck ----------------------------------------
# From: src/susi/core/susi_utils.py:1229-1275 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: spara.bd_top, spara.sfc (not a field any more)
# Review:
def heterotrophic_respiration_yr_bck(t5, yr, dfwt, dfair_r, v, spara):
    """
    Output:
        mean time series kg CO2 ha-1 day-1 and annual sum for each computation  node
    """
    sfc = spara.sfc
    # peat bulk density: change from g/cm3 to kg m-3 -> multiply by 1000
    bd_d = {
        2: 0.14,
        3: 0.11,
        4: 0.10,
        5: 0.08,
    }  # Mese study: bulk densities in different fertility classes                                                                 # peat layer thickness, cm

    if spara.bd_top is None:
        bd = bd_d[sfc] * 1000.0
    else:
        bd = np.mean(spara.bd_top) * 1000.0  # Unit conversion from g/cm3 -> kg/m3

    air_ratio = dfair_r.loc[str(yr)]
    wt = (
        dfwt[str(yr) + "-05-01" : str(yr) + "-10-31"].mean().values * -100.0
    )  # .values[:-1])

    T5 = t5.loc[str(yr)][0]

    B = 350.0
    T5zero = -46.02
    T5ref = 10.0
    T5 = np.minimum(T5, 16.0)
    Rref = (
        0.0695 + 3.7 * 10 ** (-4) * v + 5.4 * 10 ** (-4) * bd + 1.2 * 10 ** (-3) * wt
    ) * 24.0  # unit g/m2/h CO2-> g/m2/day
    Rhet = [
        rref * np.exp(B * ((1.0 / (T5ref - T5zero)) - (1.0 / (T5 - T5zero))))
        for rref in Rref
    ]
    Rhet = np.array(Rhet).T
    Rhet_root = Rhet * air_ratio
    _n, days = np.shape(Rhet)

    return (
        days,
        np.mean(Rhet, axis=1) * 10.0,
        np.sum(Rhet, axis=0) * 10.0,
        np.sum(Rhet_root, axis=0) * 10.0,
    )


# --- heterotrophic_respiration_yr_bck2 ---------------------------------------
# From: src/susi/core/susi_utils.py:1278-1322 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: spara.bd_top, spara.sfc (not a field any more)
# Review:
def heterotrophic_respiration_yr_bck2(forc, yr, dfwt, dfair_r, v, spara):
    """
    Output:
        mean time series kg CO2 ha-1 day-1 and annual sum for each computation  node
    """
    sfc = spara.sfc
    bd_d = {
        2: 0.14,
        3: 0.11,
        4: 0.10,
        5: 0.08,
    }  # Mese study: bulk densities in different fertility classes                                                                 # peat layer thickness, cm

    if spara.bd_top is None:
        bd = bd_d[sfc]
    else:
        bd = np.mean(spara.bd_top)
    air_ratio = dfair_r.loc[str(yr)]
    wt = (
        dfwt.loc[str(yr) + "-05-01" : str(yr) + "-10-31"].mean().values * -100.0
    )  # .values[:-1])

    T5 = forc.loc[str(yr)]["T"]

    B = 350.0
    T5zero = -46.02
    T5ref = 10.0
    T5 = np.minimum(T5, 16.0)
    Rref = (
        0.0695 + 3.7 * 10 ** (-4) * v + 5.4 * 10 ** (-4) * bd + 1.2 * 10 ** (-3) * wt
    ) * 24.0  # unit g/m2/h CO2-> g/m2/day
    Rhet = [
        rref * np.exp(B * ((1.0 / (T5ref - T5zero)) - (1.0 / (T5 - T5zero))))
        for rref in Rref
    ]
    Rhet = np.array(Rhet).T
    Rhet_root = Rhet * air_ratio
    _n, days = np.shape(Rhet)

    return (
        days,
        np.mean(Rhet, axis=1) * 10.0,
        np.sum(Rhet, axis=0) * 10.0,
        np.sum(Rhet_root, axis=0) * 10.0,
    )


# --- CH4_flux_yr -------------------------------------------------------------
# From: src/susi/core/susi_utils.py:1325-1352 at ec9be39
# Dead because: no callers
# Review:
def CH4_flux_yr(yr, dfwt):
    """
    Leena Stenberg 2021
    Input:
        yr: year
        dfwt: dataframe of water table depths (m, negative down)
    Output:
        CH4 flux, node-wise kg CH4 ha-1 year-1
        CH4 flux, mean kg CH4 ha-1 year-1 over computation nodes
        CH4 flux, mean kg CO2-eq. ha-1 year-1 over computation nodes
    """

    wt = (
        dfwt[str(yr) + "-05-01" : str(yr) + "-10-31"].mean().values * -100.0
    )  # convert to cm positive down
    # wt = wt[1:-1] # exclude ditch nodes

    CH4 = (
        -0.378 + 12.3 * np.exp(-0.121 * wt)
    ) * 10.0  # Ojanen et al. 2010, Fig. 6, convert to kg CH4 /ha/year

    CH4mean = np.mean(CH4, axis=0)

    return (
        CH4,
        CH4mean,
        CH4mean * 54.0,
    )  # convert to kg CO2-eq./ha/year, SGWP100 coefficient = 54 (Ojanen et al. 2021)


# --- understory_uptake -------------------------------------------------------
# From: src/susi/core/susi_utils.py:1355-1936 at ec9be39
# Dead because: no callers
# Review:
def understory_uptake(
    n,
    lat,
    lon,
    barea0,
    barea1,
    stems0,
    stems1,
    yi0,
    yi1,
    sp,
    ts,
    simtime,
    sfc,
    standAge,
):
    """
    INPUT:
        n - number of computation nodes between ditches
        lat - latitude in [units]
        lon - longitude in [units]
        barea 0,1 - stand basal area [m2], start, end
        stems 0,1 - number of stems per ha, start, end
        yi 0,1 - stand volume as time series, start, end
        sp - dominant tree species - 1 pine, 2 spruce, 3 birch
        ts - temperature sum [dd]
        simtime  - simulation time in yrs
        sfc - site fertility class,
        standAge in yrs
    SOURCE:
        Muukkonen & Makipaa, 2006. Bor.Env.Res. 11, 355-369.\n
    AUTHOR:
        Samuli Launiainen 18.06.2014, Modified for array operations by Ari Laurén 13.4.2020 \n
    NOTE:
         Multi-regression models not yet tested!
         In model equations independent variables named differently to M&M (2006): here x[0] = z1, x[1]=z2, ... x[7]=z8 and x[8]=z10\n
         \n

         Site nutrient fertility class (sfc) at mires:\n
             1: herb-rich hw-spruce swamps, pine mires, fens,
             2: V.myrtillus / tall sedge spruce swamps, tall sedge pine fens, tall sedge fens,
             3: Carex clobularis / V.vitis-idaea swamps, Carex globularis pine swamps, low sedge (oligotrophic) fens,
             4: Low sedge, dwarf-shrub & cottongrass pine bogs, ombo-oligotrophic bogs,
             5: S.fuscum pine bogs, ombotrophic and S.fuscum low sedge bogs.
         Drainage status x[8] at mires (Paavilainen & Paivanen, 1995):
             1: undrained
             2: Recently draines, slight effect on understory veg., no effect on stand
             3: Transforming drained mires, clear effect on understory veg and stand
             4: Transforming drained mires, veget. resembles upland forest site type, tree-stand forest-like.

    """
    # ATTN! convert barea, stems, yi, standAge from time series to list containing start and end state (adjustment to annual call)

    if sp == 2:
        smc = np.ones(n) * 2  # spruce
    else:
        smc = np.ones(n) * 3  # pine and others

    # n, lat, lon, barea, stems, yi, sp, ts, simtime, sfc
    age = np.ones(n) * standAge
    sfc = np.ones(n) * sfc  ## x2 surface elevation m asl
    dem = np.ones(n) * 80.0  ## x2 surface elevation m asl

    vol = yi0  # np.ones(n) * yi[0]  # x5 stand volume m3 ha-1
    ba = barea0  # np.ones(n) * barea[0]  # x7 basal area m2 ha-1
    Nstems = stems0  # np.ones(n)*stems[0]   # x6 number of stems -ha, default 900

    # expected_yield = np.ones(n)*(yi[-1]- yi[0])

    # ------------- classify and map pixels--------------------------------------------------------
    ix_spruce_mire = np.where(np.equal(smc, 2))
    ix_pine_bog = np.where(np.equal(smc, 3))
    ix_open_peat = np.where(np.equal(smc, 4))

    # ---------------------------------------
    inProj = CRS("epsg:3067")
    outProj = CRS("epsg:4326")
    transformer = Transformer.from_crs(inProj, outProj)
    latitude, longitude = transformer.transform(lon, lat)

    drain_s = 4  # x8 drainage status, default value 4

    # ---------------------------------------

    def gv_biomass_and_nutrients(
        n,
        ix_spruce_mire,
        ix_pine_bog,
        ix_open_peat,
        latitude,
        longitude,
        dem,
        ts,
        sfc,
        vol,
        Nstems,
        ba,
        drain_s,
        age,
    ):
        # --------------- nutrient contents in vegetation-----------------------
        """
        Computed:
           - total biomass and bottom layer; field layer is gained as a difference of tot and bottom layer (more cohrent results)
           - N and P storage in the each pixel
           - annual use of N and P due to litterfall
        Muukkonen Mäkipää 2005 upland sites: field layer contains dwarf shrubs and (herbs + grasses), see Fig 1
            share     dwarf shrubs     herbs
            - Pine       91%            9%
            - Spruce     71%            29%
            - broad l    38%            62%
        Peatland sites (assumption):
            share      dwarf shrubs    herbs
            - Pine bogs    90%          10%
            - Spruce mires 50%          50%
        Palviainen et al. 2005 Ecol Res (2005) 20: 652–660, Table 2
        Nutrient concentrations for
                                N              P           K
            - Dwarf shrubs      1.2%         1.0 mg/g     4.7 mg/g
            - herbs & grasses   1.8%         2.0 mg/g    15.1 mg/g
            - upland mosses     1.25%        1.4 mg/g     4.3 mg/g
        Nutrient concentrations for sphagna (FIND):
                                N              P     for N :(Bragazza et al Global Change Biology (2005) 11, 106–114, doi: 10.1111/j.1365-2486.2004.00886.x)
            - sphagnum          0.6%           1.4 mg/g     (Palviainen et al 2005)
        Annual litterfall proportions from above-ground biomass (Mälkönen 1974, Tamm 1953):
            - Dwarf shrubs          0.2
            - herbs & grasses        1
            - mosses                0.3
            Tamm, C.O. 1953. Growth, yield and nutrition in carpets of a forest moss (Hylocomium splendens). Meddelanden från Statens Skogsforsknings Institute 43 (1): 1-140.

        We assume retranslocation of N and P away from senescing tissues before litterfall:
                                N           P
            - Dwarf shrubs     0.5         0.5
            - Herbs & grasses  0.5         0.5
            - mossess          0.0         0.0

        Turnover of total biomass including the belowground biomass is assumed to be 1.2 x above-ground biomass turnover

        """

        fl_share = {
            "description": "share of dwarf shrubs (ds) and herbas & grasses (h) from field layer biomass, kg kg-1",
            "pine_upland": {"ds": 0.91, "h": 0.09},
            "spruce_upland": {"ds": 0.71, "h": 0.29},
            "broadleaved_upland": {"ds": 0.38, "h": 0.62},
            "spruce_mire": {"ds": 0.50, "h": 0.50},
            "pine_bog": {"ds": 0.90, "h": 0.10},
        }
        nut_con = {
            "description": "nutrient concentration of dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit mg/g",
            "ds": {"N": 12.0, "P": 1.0, "K": 4.7},
            "h": {"N": 18.0, "P": 2.0, "K": 15.1},
            "um": {"N": 12.5, "P": 1.4, "K": 4.3},
            "s": {"N": 6.0, "P": 1.4, "K": 4.3},
        }
        lit_share = {
            "description": "share of living biomass that is lost as litter annually for dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit: kg kg-1",
            "ds": 0.2,
            "h": 0.5,
            "um": 0.3,
            "s": 0.3,
        }
        green_share = {
            "description": "share of green mass from the total biomass of dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit kg/kg",
            "ds": 0.2,
            "h": 0.5,
            "um": 0.3,
            "s": 0.3,
        }
        retrans = {
            "description": "share of nutrients retranslocated before litterfallfor dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit: kg kg-1",
            "ds": {"N": 0.69, "P": 0.73, "K": 0.87},
            "h": {"N": 0.69, "P": 0.73, "K": 0.87},
            "um": {"N": 0.0, "P": 0.0, "K": 0.0},
            "s": {"N": 0.0, "P": 0.0, "K": 0.0},
        }
        # retrans for pine {'N': 0.69, 'P': 0.73, 'K':0.87}
        # ATTN: changes 4.1.2021 fl_share pine_bog vs spruce_mire (vice versa)
        # check, and change back restranslocation for herbs

        fl_to_total_turnover = 1.2  # converts the turnover of above-ground bionmass to total including root turnover
        fl_above_to_total = 1.7  # converts aboveground biomass to total biomass

        # --------- create output arrays -----------------------------
        gv_tot = np.zeros(n)  # Ground vegetation mass kg ha-1
        gv_field = np.zeros(n)  # Field layer vegetation mass
        gv_bot = np.zeros(n)  # Bottom layer vegetation mass
        gv_leafmass = np.zeros(n)  # Leaf mass in ground vegetation kg ha-1
        ds_litterfall = np.zeros(n)  # dwarf shrub litterfall kg ha-1 yr-1
        h_litterfall = np.zeros(n)  # herbs and grasses litterfall kg ha-1 yr-1
        s_litterfall = np.zeros(n)  # sphagnum mosses litterfall kg ha-1 yr-1
        nup_litter = np.zeros(n)  # N uptake due to litterfall kg ha-1 yr-1
        pup_litter = np.zeros(n)  # P uptake due to litterfall kg ha-1 yr-1
        kup_litter = np.zeros(n)  # K uptake due to litterfall kg ha-1 yr-1
        n_gv = np.zeros(n)  # N in ground vegetation kg ha-1
        p_gv = np.zeros(n)  # P in ground vegetation kg ha-1
        k_gv = np.zeros(n)  # K in ground vegetation kg ha-1

        """------ Ground vegetation models from Muukkonen & Mäkipää 2006 BER vol 11, Tables 6,7,8"""

        #     ***************** Spruce mire ***************************************
        ix = ix_spruce_mire
        gv_tot[ix] = (
            np.square(
                35.52
                + 0.001 * longitude * dem[ix]
                - 1.1 * drain_s**2
                - 2e-5 * vol[ix] * Nstems[ix]
                + 4e-5 * Nstems[ix] * age[ix]
                + 0.139 * longitude * drain_s
            )
            - 0.5
            + 116.54
        )  # Total, Eq.39, Table 9
        gv_bot[ix] = (
            np.square(
                -3.182
                + 0.022 * latitude * longitude
                + 2e-4 * dem[ix] * age[ix]
                - 0.077 * sfc[ix] * longitude
                - 0.003 * longitude * vol[ix]
                + 2e-4 * np.square(vol[ix])
            )
            - 0.5
            + 98.10
        )  # Bottom layer total, Eq. 35, Table 9
        gv_field[ix] = (
            np.square(
                23.24
                - 1.163 * drain_s**2
                + 1.515 * sfc[ix] * drain_s
                - 2e-5 * vol[ix] * Nstems[ix]
                + 8e-5 * ts * age[ix]
                + 1e-5 * Nstems[ix] * dem[ix]
            )
            - 0.5
            + 162.58
        )  # Field layer total, Eq. 37, Table 9

        # removing inconsistent values
        gv_field[ix] = np.minimum(gv_tot[ix], gv_field[ix])
        gv_bot[ix] = np.minimum(gv_tot[ix], gv_bot[ix])
        gv_field[ix] = np.maximum(gv_field[ix], gv_tot[ix] - gv_bot[ix])

        # annual litterfall rates
        # ATTN! vaihda nämä suoraan field layeriksi, poista tot ja bommomin kautta menevä yhteys
        ds_litterfall[ix] = (
            fl_share["spruce_mire"]["ds"]
            * (gv_tot[ix] - gv_bot[ix])
            * lit_share["ds"]
            * fl_to_total_turnover
        )
        h_litterfall[ix] = (
            fl_share["spruce_mire"]["h"]
            * (gv_tot[ix] - gv_bot[ix])
            * lit_share["h"]
            * fl_to_total_turnover
        )
        s_litterfall[ix] = gv_bot[ix] * lit_share["s"]

        # ATTN! tarkista tämä, onko järkevä? Tee oma dictionary lehtimassalle
        gv_leafmass[ix] = (
            fl_share["spruce_mire"]["ds"]
            * (gv_tot[ix] - gv_bot[ix])
            * green_share["ds"]
            + fl_share["spruce_mire"]["h"]
            * (gv_tot[ix] - gv_bot[ix])
            * green_share["h"]
            + gv_bot[ix] * green_share["s"]
        )

        n_gv[ix] = (
            gv_field[ix]
            * fl_share["spruce_mire"]["ds"]
            * nut_con["ds"]["N"]
            * 1e-3
            * fl_above_to_total
            + gv_field[ix]
            * fl_share["spruce_mire"]["h"]
            * nut_con["h"]["N"]
            * 1e-3
            * fl_above_to_total
            + gv_bot[ix] * nut_con["s"]["N"] * 1e-3
        )
        p_gv[ix] = (
            gv_field[ix]
            * fl_share["spruce_mire"]["ds"]
            * nut_con["ds"]["P"]
            * 1e-3
            * fl_above_to_total
            + gv_field[ix]
            * fl_share["spruce_mire"]["h"]
            * nut_con["h"]["P"]
            * 1e-3
            * fl_above_to_total
            + gv_bot[ix] * nut_con["s"]["P"] * 1e-3
        )
        k_gv[ix] = (
            gv_field[ix]
            * fl_share["spruce_mire"]["ds"]
            * nut_con["ds"]["K"]
            * 1e-3
            * fl_above_to_total
            + gv_field[ix]
            * fl_share["spruce_mire"]["h"]
            * nut_con["h"]["K"]
            * 1e-3
            * fl_above_to_total
            + gv_bot[ix] * nut_con["s"]["K"] * 1e-3
        )

        nup_litter[ix] = (
            ds_litterfall[ix] * nut_con["ds"]["N"] * 1e-3 * (1.0 - retrans["ds"]["N"])
            + h_litterfall[ix] * nut_con["h"]["N"] * 1e-3 * (1.0 - retrans["h"]["N"])
            + s_litterfall[ix] * nut_con["s"]["N"] * 1e-3 * (1.0 - retrans["s"]["N"])
        )

        pup_litter[ix] = (
            ds_litterfall[ix] * nut_con["ds"]["P"] * 1e-3 * (1.0 - retrans["ds"]["P"])
            + h_litterfall[ix] * nut_con["h"]["P"] * 1e-3 * (1.0 - retrans["h"]["P"])
            + s_litterfall[ix] * nut_con["s"]["P"] * 1e-3 * (1.0 - retrans["s"]["P"])
        )

        kup_litter[ix] = (
            ds_litterfall[ix] * nut_con["ds"]["K"] * 1e-3 * (1.0 - retrans["ds"]["K"])
            + h_litterfall[ix] * nut_con["h"]["K"] * 1e-3 * (1.0 - retrans["h"]["K"])
            + s_litterfall[ix] * nut_con["s"]["K"] * 1e-3 * (1.0 - retrans["s"]["K"])
        )

        # ***************** Pine bogs ***************************************

        ix = ix_pine_bog
        gv_tot[ix] = (
            np.square(
                50.098
                + 0.005 * longitude * dem[ix]
                - 1e-5 * vol[ix] * Nstems[ix]
                + 0.026 * sfc[ix] * age[ix]
                - 1e-4 * dem[ix] * ts
                - 0.014 * vol[ix] * drain_s
            )
            - 0.5
            + 167.40
        )  # Total, Eq 45, Table 9
        gv_bot[ix] = (
            np.square(
                31.809
                + 0.008 * longitude * dem[ix]
                - 3e-4 * Nstems[ix] * ba[ix]
                + 6e-5 * Nstems[ix] * age[ix]
                - 0.188 * dem[ix]
            )
            - 0.5
            + 222.22
        )  # Bottom layer total, Eq 41, Table 9
        gv_field[ix] = (
            np.square(
                48.12
                - 1e-5 * ts**2
                + 0.013 * sfc[ix] * age[ix]
                - 0.04 * vol[ix] * drain_s
                + 0.026 * sfc[ix] * vol[ix]
            )
            - 0.5
            + 133.26
        )  # Field layer total, Eq. 43, Table 9

        # removing inconsistent values
        gv_field[ix] = np.minimum(gv_tot[ix], gv_field[ix])
        gv_bot[ix] = np.minimum(gv_tot[ix], gv_bot[ix])
        gv_field[ix] = np.maximum(gv_field[ix], gv_tot[ix] - gv_bot[ix])

        # annual litterfall rates

        ds_litterfall[ix] = (
            fl_share["pine_bog"]["ds"]
            * (gv_tot[ix] - gv_bot[ix])
            * lit_share["ds"]
            * fl_to_total_turnover
        )
        h_litterfall[ix] = (
            fl_share["pine_bog"]["h"]
            * (gv_tot[ix] - gv_bot[ix])
            * lit_share["h"]
            * fl_to_total_turnover
        )
        s_litterfall[ix] = gv_bot[ix] * lit_share["s"]
        gv_leafmass[ix] = (
            fl_share["pine_bog"]["ds"] * (gv_tot[ix] - gv_bot[ix]) * green_share["ds"]
            + fl_share["pine_bog"]["h"] * (gv_tot[ix] - gv_bot[ix]) * green_share["h"]
            + gv_bot[ix] * green_share["s"]
        )

        n_gv[ix] = (
            gv_field[ix]
            * fl_share["pine_bog"]["ds"]
            * nut_con["ds"]["N"]
            * 1e-3
            * fl_above_to_total
            + gv_field[ix]
            * fl_share["pine_bog"]["h"]
            * nut_con["h"]["N"]
            * 1e-3
            * fl_above_to_total
            + gv_bot[ix] * nut_con["s"]["N"] * 1e-3
        )

        p_gv[ix] = (
            gv_field[ix]
            * fl_share["pine_bog"]["ds"]
            * nut_con["ds"]["P"]
            * 1e-3
            * fl_above_to_total
            + gv_field[ix]
            * fl_share["pine_bog"]["h"]
            * nut_con["h"]["P"]
            * 1e-3
            * fl_above_to_total
            + gv_bot[ix] * nut_con["s"]["P"] * 1e-3
        )

        k_gv[ix] = (
            gv_field[ix]
            * fl_share["pine_bog"]["ds"]
            * nut_con["ds"]["K"]
            * 1e-3
            * fl_above_to_total
            + gv_field[ix]
            * fl_share["pine_bog"]["h"]
            * nut_con["h"]["K"]
            * 1e-3
            * fl_above_to_total
            + gv_bot[ix] * nut_con["s"]["K"] * 1e-3
        )

        nup_litter[ix] = (
            ds_litterfall[ix] * nut_con["ds"]["N"] * 1e-3 * (1.0 - retrans["ds"]["N"])
            + h_litterfall[ix] * nut_con["h"]["N"] * 1e-3 * (1.0 - retrans["h"]["N"])
            + s_litterfall[ix] * nut_con["s"]["N"] * 1e-3 * (1.0 - retrans["s"]["N"])
        )

        pup_litter[ix] = (
            ds_litterfall[ix] * nut_con["ds"]["P"] * 1e-3 * (1.0 - retrans["ds"]["P"])
            + h_litterfall[ix] * nut_con["h"]["P"] * 1e-3 * (1.0 - retrans["h"]["P"])
            + s_litterfall[ix] * nut_con["s"]["P"] * 1e-3 * (1.0 - retrans["s"]["P"])
        )

        kup_litter[ix] = (
            ds_litterfall[ix] * nut_con["ds"]["K"] * 1e-3 * (1.0 - retrans["ds"]["K"])
            + h_litterfall[ix] * nut_con["h"]["K"] * 1e-3 * (1.0 - retrans["h"]["K"])
            + s_litterfall[ix] * nut_con["s"]["K"] * 1e-3 * (1.0 - retrans["s"]["K"])
        )

        # ------------Change clear-cut areas: reduce to 1/3 of modelled ---------------------------------------------------
        to_cc = 0.33
        # ix_cc = np.where(np.logical_and(gisdata['age']<5.0, gisdata['smc']!=4))  #small stands excluding open peatlands
        ix_cc = np.where(age < 5.0)
        n_gv[ix_cc] = n_gv[ix_cc] * to_cc
        p_gv[ix_cc] = p_gv[ix_cc] * to_cc
        k_gv[ix_cc] = k_gv[ix_cc] * to_cc
        nup_litter[ix_cc] = nup_litter[ix_cc] * to_cc
        pup_litter[ix_cc] = pup_litter[ix_cc] * to_cc
        kup_litter[ix_cc] = kup_litter[ix_cc] * to_cc
        gv_tot[ix_cc] = gv_tot[ix_cc] * to_cc

        litterfall_gv = ds_litterfall + h_litterfall + s_litterfall
        return (
            n_gv,
            p_gv,
            k_gv,
            nup_litter,
            pup_litter,
            kup_litter,
            gv_tot,
            litterfall_gv,
            gv_leafmass,
        )

    # initial N and P mass, kg ha-1
    (
        n_gv,
        p_gv,
        k_gv,
        nup_litter,
        pup_litter,
        kup_litter,
        _gv_tot,
        litterfall_gv,
        gv_leafmass,
    ) = gv_biomass_and_nutrients(
        n,
        ix_spruce_mire,
        ix_pine_bog,
        ix_open_peat,
        latitude,
        longitude,
        dem,
        ts,
        sfc,
        vol,
        Nstems,
        ba,
        drain_s,
        age,
    )

    # ground vegetation mass at the end of simulation, kg ha-1
    vol = yi1  # vol + expected_yield
    Nstems = stems1  # np.ones(n)*stems[-1]
    ba = barea1  # np.ones(n)*barea[-1]

    age = np.ones(n) * (standAge + simtime)
    (
        n_gv_end,
        p_gv_end,
        k_gv_end,
        nup_litter_end,
        pup_litter_end,
        kup_litter_end,
        _gv_tot_end,
        litterfall_gv_end,
        gv_leafmass_end,
    ) = gv_biomass_and_nutrients(
        n,
        ix_spruce_mire,
        ix_pine_bog,
        ix_open_peat,
        latitude,
        longitude,
        dem,
        ts,
        sfc,
        vol,
        Nstems,
        ba,
        drain_s,
        age,
    )

    # nutrient uptake due to net change in gv biomass, only positive values accepted, negative do not associate to nutrient uptake
    nup_net = np.where(n_gv_end - n_gv > 0.0, n_gv_end - n_gv, 0.0)
    pup_net = np.where(p_gv_end - p_gv > 0.0, p_gv_end - p_gv, 0.0)
    kup_net = np.where(k_gv_end - k_gv > 0.0, k_gv_end - k_gv, 0.0)

    nup_litter_mean = np.mean([nup_litter, nup_litter_end], axis=0)
    pup_litter_mean = np.mean([pup_litter, pup_litter_end], axis=0)
    kup_litter_mean = np.mean([kup_litter, kup_litter_end], axis=0)

    nup = (
        nup_net + nup_litter_mean * simtime
    )  # total N uptake kg ha-1 simulation time (in yrs) -1
    pup = (
        pup_net + pup_litter_mean * simtime
    )  # total P uptake kg ha-1 simulation time (in yrs) -1
    kup = (
        kup_net + kup_litter_mean * simtime
    )  # total P uptake kg ha-1 simulation time (in yrs) -1

    litterfall_tot = (
        np.mean([litterfall_gv, litterfall_gv_end], axis=0) * simtime
    )  # total gv litterfall in the simulation time kg ha-1
    np.mean([gv_leafmass, gv_leafmass_end], axis=0)

    """
    print ('    + Ground vegetation nutrient demand')
    fill = ('      -')
    print (fill, 'gv biomass', np.round(np.mean(gv_tot),2),'...' ,np.round(np.mean(gv_tot_end),2))
    print (fill, 'N demand', np.round(np.mean(nup),2))
    print (fill, 'P demand', np.round(np.mean(pup),2))
    print (fill, 'K demand', np.round(np.mean(kup),2))
    print (fill, 'gv litterfall', np.round(np.mean(litterfall_tot),2))
    """
    return (
        nup,
        pup,
        kup,
        nup_litter_mean,
        pup_litter_mean,
        kup_litter_mean,
        litterfall_tot,
        gv_leafmass,
    )


# ============================================================================
# src/susi/io/outputs.py
# ============================================================================


# --- Outputs.initialize_doc --------------------------------------------------
# From: src/susi/io/outputs.py:1914-1944 at ec9be39
# Dead because: no callers
# Review:
def initialize_doc(self):
    DOC = self.ncf.createVariable(
        "/doc/DOC",
        "f4",
        (
            "nscens",
            "nyrs",
            "ncols",
        ),
    )
    DOC.units = "total DOC release [kg/ha/yr]"
    LMW = self.ncf.createVariable(
        "/doc/LMW",
        "f4",
        (
            "nscens",
            "nyrs",
            "ncols",
        ),
    )
    LMW.units = "light molecular weight DOC release [kg/ha/yr]"
    HMW = self.ncf.createVariable(
        "/doc/HMW",
        "f4",
        (
            "nscens",
            "nyrs",
            "ncols",
        ),
    )
    HMW.units = "high molecular weight DOC release [kg/ha/yr]"


# --- Outputs.write_doc -------------------------------------------------------
# From: src/susi/io/outputs.py:2625-2628 at ec9be39
# Dead because: no callers
# Review:
def write_doc(self, scen, year, DOC, HMW):
    self.ncf["doc"]["DOC"][scen, year, :] = DOC
    self.ncf["doc"]["HMW"][scen, year, :] = HMW
    self.ncf["doc"]["LMW"][scen, year, :] = DOC - HMW


# ============================================================================
# src/susi/io/susi_io.py
# ============================================================================


# --- dwt_to_excel ------------------------------------------------------------
# From: src/susi/io/susi_io.py:15-19 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['gwl_file'] (not a field any more), outpara['outfolder'] (not a field any more)
# Review:
def dwt_to_excel(gwl, outpara, scen):
    df = pd.DataFrame(gwl)
    f = outpara["outfolder"] + outpara["gwl_file"] + "_" + scen + ".xlsx"
    df.to_excel(f)
    print("DWT to ", f)


# --- runoff_to_excel ---------------------------------------------------------
# From: src/susi/io/susi_io.py:22-26 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['outfolder'] (not a field any more), outpara['runfile'] (not a field any more)
# Review:
def runoff_to_excel(runoff, swe, outpara, scen):
    datadic = {"runoff": runoff, "swe": swe}
    df = pd.DataFrame(data=datadic)
    f = outpara["outfolder"] + outpara["runfile"] + "_" + scen + ".xlsx"
    df.to_excel(f)


# --- c_and_nut_to_excel ------------------------------------------------------
# From: src/susi/io/susi_io.py:29-87 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['c_and_nut_file'] (not a field any more), outpara['outfolder'] (not a field any more)
# Review:
def c_and_nut_to_excel(
    inivol,
    vols,
    phys_restrictions,
    c_bals,
    c_bals_trees,
    ch4_yr,
    n_export_yr,
    p_export_yr,
    k_export_yr,
    krels,
    outpara,
    scen,
):
    data = {
        "vols": vols,
        "phys_restr": phys_restrictions,
        "c_bals": c_bals,
        "c_bals_trees": c_bals_trees,
        "ch4_yr": ch4_yr,
        "n_export_yr": n_export_yr,
        "p_export_yr": p_export_yr,
        "k_export_yr": k_export_yr,
        "krelease": krels,
    }
    f = outpara["outfolder"] + scen + "_" + outpara["c_and_nut_file"]

    writer = pd.ExcelWriter(f, engine="xlsxwriter")
    dfcbals = pd.DataFrame(data["c_bals"])
    dfcbalstrees = pd.DataFrame(data["c_bals_trees"])
    dfn = pd.DataFrame(data["n_export_yr"])
    dfp = pd.DataFrame(data["p_export_yr"])
    dfk = pd.DataFrame(data["k_export_yr"])
    dfch4 = pd.DataFrame(data["ch4_yr"])
    dfkrels = pd.DataFrame(data["krelease"])
    dfphys = pd.DataFrame(data["phys_restr"])

    dfrow = pd.DataFrame([inivol])
    dfrow.columns = dfcbals.columns
    dfv = pd.DataFrame(data["vols"])
    dfv = pd.concat([dfrow, dfv], axis=0)

    gr = np.gradient(dfv.to_numpy(), axis=0)
    cols = np.shape(gr)[1]
    colnames = list(range(cols))
    dfgr = pd.DataFrame(data=gr, columns=colnames)

    dfv.to_excel(writer, sheet_name="Stand volume")
    dfgr.to_excel(writer, sheet_name="Growth")
    dfcbals.to_excel(writer, sheet_name="Peat C balance")
    dfcbalstrees.to_excel(writer, sheet_name="Stand C balance")
    dfn.to_excel(writer, sheet_name="N export")
    dfp.to_excel(writer, sheet_name="P export")
    dfk.to_excel(writer, sheet_name="K export")
    dfch4.to_excel(writer, sheet_name="ch4")
    dfkrels.to_excel(writer, sheet_name="K release")
    dfphys.to_excel(writer, sheet_name="Physical restriction")

    writer.save()


# --- output_dwt_growing_season -----------------------------------------------
# From: src/susi/io/susi_io.py:90-124 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['endday'], outpara['endmonth'], outpara['outfolder'] (not a field any more), outpara['startday'], outpara['startmonth'], outpara['to_file'] (not a field any more), outpara['tsfile'] (not a field any more), wpara['end_yr'] (not a field any more), wpara['start_yr'] (not a field any more)
# Review:
def output_dwt_growing_season(
    dwt, length, start_yr, end_yr, start_date, outpara, wpara, scen
):
    """
    Output for growing season water tables
    """
    import datetime

    _days, n = np.shape(dwt)
    #    dfOut = pd.DataFrame(data={'dwt': dwt},
    #                   index=pd.date_range(start_date,periods=length))  #len(deltas)
    dfOut = pd.DataFrame(
        dwt, columns=range(n), index=pd.date_range(start_date, periods=length)
    )  # len(deltas)

    dfOut["doy"] = dfOut.index.dayofyear
    if outpara["to_file"]:
        dfOut.to_csv(outpara["outfolder"] + outpara["tsfile"] + "_" + scen + ".csv")

    y = wpara["start_yr"]
    m = outpara["startmonth"]
    d = outpara["startday"]
    ey = wpara["end_yr"]
    start = datetime.datetime(y, m, d).timetuple().tm_yday
    m = outpara["endmonth"]
    d = outpara["endday"]
    end = datetime.datetime(ey, m, d).timetuple().tm_yday

    summer = dfOut.groupby(dfOut["doy"]).mean()
    summer = summer[start:end]
    # summer_mean_dwt = np.round(summer['dwt'].mean(),3)
    summer_mean_dwt = summer.mean(axis=0)

    # print '  +Summer mean dwt', summer_mean_dwt.values
    return summer_mean_dwt.values, summer


# --- write_mese --------------------------------------------------------------
# From: src/susi/io/susi_io.py:127-157 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['ofile'] (not a field any more), outpara['outfolder'] (not a field any more)
# Review:
def write_mese(fout, nro, v_ini, v, iv5, Nrel, Prel, Krel, Crel, dwt_loc, cb, cbt, sfc):
    # from xlutils.copy import copy
    import xlrd
    from xlutils import copy

    # fout = outpara['outfolder'] + outpara['ofile']
    rb = xlrd.open_workbook(fout)  # ,formatting_info=True)
    rows = rb.sheet_by_name("Summary").nrows
    for i in range(rb.nsheets):
        sheet = rb.sheet_by_index(i)
        if sheet.name == "Summary":
            indSummary = i

    wb = copy.copy(rb)
    outSummary = wb.get_sheet(indSummary)
    # sarakkeita = outSummary.ncols
    # print sarakkeita
    outSummary.write(rows, 0, nro)
    outSummary.write(rows, 1, v_ini)
    outSummary.write(rows, 2, v)
    outSummary.write(rows, 3, iv5)
    outSummary.write(rows, 4, Nrel)
    outSummary.write(rows, 5, Prel)
    outSummary.write(rows, 6, Krel)
    outSummary.write(rows, 7, Crel)
    outSummary.write(rows, 8, dwt_loc)
    outSummary.write(rows, 9, cb)
    outSummary.write(rows, 10, cbt)
    outSummary.write(rows, 11, sfc)

    wb.save(fout)


# --- write_mese_scen ---------------------------------------------------------
# From: src/susi/io/susi_io.py:160-198 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['ofile'] (not a field any more), outpara['outfolder'] (not a field any more)
# Review:
def write_mese_scen(fout, nro, v_ini, v_end, gr, w, dw):
    # from xlutils.copy import copy
    import xlrd
    from xlutils import copy

    r = len(v_end)  # rounds
    ix0 = range(2, 2 + r)
    ix1 = range(2 + r, 2 + 2 * r)
    ix2 = range(2 + 2 * r, 2 + 3 * r)
    ix3 = range(2 + 3 * r, 2 + 4 * r)
    range(2 + 4 * r, 2 + 5 * r)
    range(2 + 5 * r, 2 + 6 * r)
    range(2 + 6 * r, 2 + 7 * r)
    range(2 + 7 * r, 2 + 8 * r)
    2 + 8 * r
    # fout = outpara['outfolder'] + outpara['ofile']
    rb = xlrd.open_workbook(fout)  # ,formatting_info=True)
    rows = rb.sheet_by_name("Summary").nrows
    for i in range(rb.nsheets):
        sheet = rb.sheet_by_index(i)
        if sheet.name == "Summary":
            indSummary = i

    wb = copy.copy(rb)
    outSummary = wb.get_sheet(indSummary)
    # sarakkeita = outSummary.ncols
    # print sarakkeita
    outSummary.write(rows, 0, nro)
    outSummary.write(rows, 1, v_ini)
    for ix, vv in zip(ix0, v_end):
        outSummary.write(rows, ix, vv)
    for ix, g in zip(ix1, gr):
        outSummary.write(rows, ix, g)
    for ix, ww in zip(ix2, w):
        outSummary.write(rows, ix, ww)
    for ix, dww in zip(ix3, dw):
        outSummary.write(rows, ix, dww)

    wb.save(fout)


# --- write_jaali_scen --------------------------------------------------------
# From: src/susi/io/susi_io.py:201-244 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['ofile'] (not a field any more), outpara['outfolder'] (not a field any more)
# Review:
def write_jaali_scen(fout, nro, ID, v_ini, v_end, gr, cb, dcb, w, dw, runo, druno):
    # from xlutils.copy import copy
    import xlrd
    from xlutils import copy

    r = len(v_end)  # rounds
    ix0 = range(2, 2 + r)
    ix1 = range(2 + r, 2 + 2 * r)
    ix2 = range(2 + 2 * r, 2 + 3 * r)
    ix3 = range(2 + 3 * r, 2 + 4 * r)
    ix4 = range(2 + 4 * r, 2 + 5 * r)
    ix5 = range(2 + 5 * r, 2 + 6 * r)
    ix6 = 2 + 6 * r  # range(2+6*r, 2+7*r)

    # fout = outpara['outfolder'] + outpara['ofile']
    rb = xlrd.open_workbook(fout)  # ,formatting_info=True)
    rows = rb.sheet_by_name("Summary").nrows
    for i in range(rb.nsheets):
        sheet = rb.sheet_by_index(i)
        if sheet.name == "Summary":
            indSummary = i

    wb = copy.copy(rb)
    outSummary = wb.get_sheet(indSummary)
    # sarakkeita = outSummary.ncols
    # print sarakkeita
    outSummary.write(rows, 0, nro)
    outSummary.write(rows, 1, v_ini)
    for ix, vv in zip(ix0, v_end):  # vol at the end
        outSummary.write(rows, ix, vv)
    for ix, g in zip(ix1, gr):  # growth
        outSummary.write(rows, ix, g)
    for ix, ww in zip(ix2, w):
        outSummary.write(rows, ix, ww)  # wt
    for ix, dww in zip(ix3, dw):
        outSummary.write(rows, ix, dww)  # dwt
    for ix, r in zip(ix4, runo):
        outSummary.write(rows, ix, r)  # runoff
    for ix, dr in zip(ix5, druno):
        outSummary.write(rows, ix, dr)  # drunoff
    print(rows, ix6, float(ID))
    outSummary.write(rows, ix6, float(ID))

    wb.save(fout)


# --- write_demand ------------------------------------------------------------
# From: src/susi/io/susi_io.py:247-269 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['ofile'] (not a field any more), outpara['outfolder'] (not a field any more)
# Review:
def write_demand(fout, nro, Ndem, Pdem, Kdem):
    # from xlutils.copy import copy
    import xlrd
    from xlutils import copy

    # fout = outpara['outfolder'] + outpara['ofile']
    rb = xlrd.open_workbook(fout)  # ,formatting_info=True)
    rows = rb.sheet_by_name("Summary").nrows
    for i in range(rb.nsheets):
        sheet = rb.sheet_by_index(i)
        if sheet.name == "Summary":
            indSummary = i

    wb = copy.copy(rb)
    outSummary = wb.get_sheet(indSummary)
    # sarakkeita = outSummary.ncols
    # print sarakkeita
    outSummary.write(rows, 0, nro)
    outSummary.write(rows, 1, Ndem)
    outSummary.write(rows, 2, Pdem)
    outSummary.write(rows, 3, Kdem)

    wb.save(fout)


# --- write_excel -------------------------------------------------------------
# From: src/susi/io/susi_io.py:272-308 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: outpara['ofile'] (not a field any more), outpara['outfolder'] (not a field any more), spara.L, spara.peat_type, spara.slope, wpara['end_yr'] (not a field any more), wpara['start_yr'] (not a field any more)
# Review:
def write_excel(
    wlocation, wpara, spara, outpara, LAI, hdom, h0_west, h0_east, summer, summermed
):
    # from xlutils.copy import copy
    import xlrd
    from xlutils import copy

    fout = outpara["outfolder"] + outpara["ofile"]
    rb = xlrd.open_workbook(fout)  # ,formatting_info=True)
    rows = rb.sheet_by_name("Summary").nrows
    for i in range(rb.nsheets):
        sheet = rb.sheet_by_index(i)
        if sheet.name == "Summary":
            indSummary = i

    wb = copy.copy(rb)
    outSummary = wb.get_sheet(indSummary)
    # sarakkeita = outSummary.ncols
    # print sarakkeita
    outSummary.write(rows, 0, rows)
    outSummary.write(rows, 1, h0_west)
    outSummary.write(rows, 2, h0_east)
    outSummary.write(rows, 3, spara.L)
    outSummary.write(rows, 4, spara.slope)
    outSummary.write(rows, 5, wlocation)
    outSummary.write(rows, 6, spara.peat_type)
    LAI if LAI != "iterable" else max(LAI)
    outSummary.write(rows, 7, max(LAI))
    hdom if hdom != "iterable" else max(hdom)
    outSummary.write(rows, 8, max(hdom))
    outSummary.write(rows, 9, str(wpara["start_yr"]) + " " + str(wpara["end_yr"]))
    outSummary.write(rows, 10, summer["dwt"].mean())
    outSummary.write(rows, 11, summer["dwt"].std())
    outSummary.write(rows, 12, summermed["dwt"].mean())
    outSummary.write(rows, 13, summermed["dwt"].std())

    wb.save(fout)


# --- write_gr_excel ----------------------------------------------------------
# From: src/susi/io/susi_io.py:311-355 at ec9be39
# Dead because: only called by `diff_nutrient_release` (also dead)
# Reads SusiParams fields: outpara['gr_file'] (not a field any more), outpara['outfolder'] (not a field any more), spara.L, spara.ditch_depth (not a field any more), spara.ditch_depth_20y (not a field any more), spara.hdom, spara.peat_type, spara.peat_type_bottom, spara.scenario_name, spara.sfc (not a field any more), spara.slope, spara.stand_params, spara.vol, spara.vonP_bottom, spara.vonP_top, wpara['end_yr'] (not a field any more), wpara['start_yr'] (not a field any more)
# Review:
def write_gr_excel(wlocation, wpara, spara, outpara, gN, gP, gK, c, cr_depth, gr_crd):
    print("now printing gr-excel")
    # from xlutils.copy import copy
    import xlrd
    from xlutils import copy

    title = "Control vs " + spara.scenario_name[c]
    fout = outpara["outfolder"] + outpara["gr_file"]
    rb = xlrd.open_workbook(fout)  # ,formatting_info=True)
    rows = rb.sheet_by_name("Summary").nrows
    print(rows)
    for i in range(rb.nsheets):
        sheet = rb.sheet_by_index(i)
        if sheet.name == "Summary":
            indSummary = i
    wb = copy.copy(rb)
    outSummary = wb.get_sheet(indSummary)
    # sarakkeita = outSummary.ncols
    # print sarakkeita
    outSummary.write(rows, 0, rows)
    outSummary.write(rows, 1, title)
    outSummary.write(rows, 2, spara.ditch_depth[0])
    outSummary.write(rows, 3, spara.ditch_depth[c])
    outSummary.write(rows, 4, spara.ditch_depth_20y[0])
    outSummary.write(rows, 5, spara.ditch_depth_20y[c])
    outSummary.write(rows, 6, spara.L)
    outSummary.write(rows, 7, spara.slope)
    outSummary.write(rows, 8, wlocation)
    outSummary.write(rows, 9, spara.peat_type)
    outSummary.write(rows, 10, str(spara.vonP_top))
    outSummary.write(rows, 11, spara.peat_type_bottom)
    outSummary.write(rows, 12, spara.vonP_bottom)
    outSummary.write(rows, 13, spara.vol)
    outSummary.write(rows, 14, spara.hdom)
    outSummary.write(
        rows, 15, str(spara.stand_params.canopy_layer_allometry.zones_species_id)
    )
    outSummary.write(rows, 16, spara.sfc)
    outSummary.write(rows, 17, str(wpara["start_yr"]) + " " + str(wpara["end_yr"]))
    outSummary.write(rows, 18, float(gN))
    outSummary.write(rows, 19, float(gP))
    outSummary.write(rows, 20, float(gK))
    outSummary.write(rows, 21, float(cr_depth))
    outSummary.write(rows, 22, float(gr_crd))
    wb.save(fout)


# --- outfig ------------------------------------------------------------------
# From: src/susi/io/susi_io.py:358-391 at ec9be39
# Dead because: no callers
# Review:
def outfig(
    summer_dwt, co2_respi, growth_response, ditch_depth, relative_response, rounds
):
    plt.figure(
        num="Susi drainage",
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(18.0, 10.0),
    )  # Figsize(w,h), tuple inches
    growth_response = np.ravel(growth_response)
    summer_dwt = np.array(summer_dwt)
    co2_respi = np.array(co2_respi)
    growth_response = np.array(growth_response)
    ditch_depth = np.array(ditch_depth)
    # growth_response= np.insert(growth_response, 0,0.0)
    plt.subplot(221)
    plt.plot(ditch_depth, -1 * summer_dwt, "bo-")
    plt.xlabel("ditch depth m")
    plt.ylabel("summer dwt m")
    plt.subplot(222)
    plt.plot(ditch_depth, co2_respi, "ko-")
    plt.xlabel("ditch depth m")
    plt.ylabel("heterotrophic respiration kg/ha CO2")
    plt.subplot(223)
    plt.plot(ditch_depth, growth_response, "go-")
    plt.xlabel("ditch depth m")
    plt.ylabel("growth response m3 ha-1")
    plt.subplot(224)
    plt.plot(ditch_depth, relative_response, "go-")
    plt.xlabel("ditch depth m")
    plt.ylabel("relative growth response %")

    # plt.plot()
    plt.show()


# --- fig_stand_growth --------------------------------------------------------
# From: src/susi/io/susi_io.py:429-469 at ec9be39
# Dead because: no callers
# Review:
def fig_stand_growth(
    r, rounds, ageSim, start_yr, end_yr, ageToVol, agearray, nppvol, nutvol, name
):
    sns.set()

    plt.figure(
        num="Growth and production",
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(20.0, 10.0),
    )
    gr_limit = 0.15  # allowed difference from table growth
    if rounds < 3:
        layout = "12"
    elif rounds < 5:
        layout = "22"
    else:
        layout = "33"

    fg_no = layout + str(r + 1)
    plt.subplot(fg_no)
    length = end_yr - start_yr + 1.0
    start = max(0.0, ageSim - 3.0)
    x = np.arange(start, ageSim + length, 0.1)
    y = ageToVol(x)
    gr_age = np.arange(ageSim, ageSim + length, 0.1)
    gr = np.gradient(ageToVol(gr_age))
    gr_up = ageToVol(gr_age) + np.cumsum(gr) * (1.0 + gr_limit)
    gr_low = ageToVol(gr_age) - np.cumsum(gr) * (1.0 - gr_limit)
    plt.plot(x, y)
    plt.plot(ageSim, ageToVol(ageSim), "ro")
    plt.plot(gr_age, gr_up, "g-")
    plt.plot(gr_age, gr_low, "g-")
    plt.fill_between(gr_age, gr_low, gr_up, color="gray", alpha=0.3)
    plt.plot(agearray, nppvol, "m-")
    plt.plot(agearray, nutvol, "r-")
    plt.title(name)
    plt.ylabel("Stand volume $m^3 ha^{-1}$")
    endvol = np.round(min(nppvol[-1], nutvol[-1]), 3)
    plt.text(ageSim, y[-1], str(endvol))
    plt.show()


# --- fig_stand_growth_node_bck -----------------------------------------------
# From: src/susi/io/susi_io.py:473-516 at ec9be39
# Dead because: no callers
# Review:
def fig_stand_growth_node_bck(
    r, rounds, ageSim, start_yr, end_yr, ageToVol, agearray, nppvol, nutvol, name
):
    sns.set()

    plt.figure(
        num="Growth and production",
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(20.0, 10.0),
    )
    gr_limit = 0.15  # allowed difference from table growth
    if rounds < 3:
        layout = "12"
    elif rounds < 5:
        layout = "22"
    else:
        layout = "33"

    fg_no = layout + str(r + 1)
    plt.subplot(fg_no)
    length = end_yr - start_yr + 1.0
    start = max(0.0, ageSim - 3.0)
    x = np.arange(start, ageSim + length, 0.1)
    y = ageToVol(x)
    gr_age = np.arange(ageSim, ageSim + length, 0.1)
    gr = np.gradient(ageToVol(gr_age))
    gr_up = ageToVol(gr_age) + np.cumsum(gr) * (1.0 + gr_limit)
    gr_low = ageToVol(gr_age) - np.cumsum(gr) * (1.0 - gr_limit)
    plt.plot(x, y)
    plt.plot(ageSim, ageToVol(ageSim), "ro")
    plt.plot(gr_age, gr_up, "g-")
    plt.plot(gr_age, gr_low, "g-")
    plt.fill_between(gr_age, gr_low, gr_up, color="gray", alpha=0.3)
    # nodes = np.shape(nppvol)[1]
    for n in nppvol.T:
        plt.plot(agearray, n, "m-")
    for n in nutvol.T:
        plt.plot(agearray, n, "r-")
    plt.title(name)
    plt.ylabel("Stand volume $m^3 ha^{-1}$")
    endvol = np.round(min(np.mean(nppvol[-1, :]), np.mean(nutvol[-1, :])), 3)
    plt.text(ageSim, y[-1], str(endvol))
    plt.show()


# --- fig_stand_growth_node ---------------------------------------------------
# From: src/susi/io/susi_io.py:519-574 at ec9be39
# Dead because: no callers
# Review:
def fig_stand_growth_node(
    rounds, ageSim, start_yr, end_yr, ageToVol, agearray, vols, name, dwts
):
    sns.set()
    # yrs, cols = np.shape(agearray)

    plt.figure(
        num="Susi-tulokset",
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(18.0, 9.0),
    )
    gr_limit = 0.15  # allowed difference from table growth

    plt.subplot(211)  # growth figure
    length = end_yr - start_yr + 1.0

    for column, agerange in enumerate(agearray.T):
        start = max(0.0, ageSim[column] - 3.0)
        x = np.arange(start, ageSim[column] + length, 0.1)
        y = ageToVol(x)
        gr_age = np.arange(ageSim[column], ageSim[column] + length + 0.1, 0.1)
        gr = np.gradient(ageToVol(gr_age))
        gr_up = ageToVol(gr_age) + np.cumsum(gr) * (1.0 + gr_limit)
        gr_low = ageToVol(gr_age) - np.cumsum(gr) * (1.0 - gr_limit)
        plt.plot(x, y, "k-")
        plt.plot(ageSim[column], ageToVol(ageSim[column]), "ro")
        plt.plot(gr_age, gr_up, "g-")
        plt.plot(gr_age, gr_low, "g-")
        plt.fill_between(gr_age, gr_low, gr_up, color="gray", alpha=0.3)

        colors = ["blue", "red", "green", "yellow", "cyan", "magenta"]
        rnds, _yrs, _nodes = np.shape(vols)
        for r in range(rnds):
            vtmp = vols[r, :, column]
            vtmp = np.insert(vtmp, 0, ageToVol(ageSim[column]))
            plt.plot(agerange, vtmp, color=colors[r], linewidth=3)
    # ---------------------------------------------------------

    plt.subplot(212)
    rnds, days, _nodes = np.shape(dwts)
    agedays = np.array(range(days))
    for r in range(rnds):
        dwttmp = dwts[r, :, 1:-1]
        up = np.max(dwttmp.T, axis=0)
        down = np.min(dwttmp.T, axis=0)
        mean = np.mean(dwttmp.T, axis=0)
        # plt.plot(age, up , color= colors[r], alpha=0.25)
        # plt.plot(age, down, color= colors[r], alpha=0.25)
        plt.plot(agedays, mean, color=colors[r], linewidth=1)
        plt.fill_between(agedays, up, down, color=colors[r], alpha=0.2, label=name[r])
    plt.legend(loc="lower left")
    plt.xlabel("Time, days", fontsize=18)
    plt.ylabel("Water level,m", fontsize=18)

    plt.show()


# --- fig_hydro ---------------------------------------------------------------
# From: src/susi/io/susi_io.py:577-798 at ec9be39
# Dead because: no callers
# Reads SusiParams fields: spara.L, spara.n, spara.peat_type, spara.slope, wpara['description'] (not a field any more), wpara['end_yr'] (not a field any more), wpara['start_yr'] (not a field any more)
# Review:
def fig_hydro(
    ele,
    hts,
    spara,
    wpara,
    wlocation,
    ets,
    Prec,
    T,
    het,
    lai,
    h0_west,
    h0_east,
    runoff,
    scen,
):
    from matplotlib.lines import Line2D

    het = np.ravel(het)
    n = spara.n
    L = spara.L
    sim_yrs = len(het) / 365.0
    aa, _bb = np.shape(hts)
    x = np.linspace(0, L, n)
    float(L / n)
    fig = plt.figure(
        num="Striphy" + scen,
        facecolor=(232 / 255.0, 243 / 255.0, 245.0 / 255),
        edgecolor="k",
        figsize=(20.0, 11.0),
    )  # Figsize(w,h), tuple inches
    ax = fig.add_axes([0.05, 0.5, 0.55, 0.46])  # left, bottom, width, height
    low = min([ele[0] + h0_west, ele[n - 1] + h0_east]) * 0.4
    high = max(ele) * 1.2
    ax.set_ylim([low, high])
    (_line2,) = ax.plot(
        x[1 : n - 1], ele[1 : n - 1], "k-", linewidth=2, label="Surface elevation"
    )
    line1 = Line2D(
        [],
        [],
        color="blue",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        label="Water table",
    )
    limit = ele - 0.5
    limitGr = ele - 0.35
    plt.plot(x, ele, "k-", x, limit, "k--")
    mid = int(n / 2)
    west = int(n / 10)
    east = n - int(n / 10)
    plt.fill_between(x, limit, ele, color="green", alpha=0.3)
    plt.fill_between(x, limitGr, ele, color="red", alpha=0.3)
    plt.plot(x[mid], ele[mid], "ro", markersize=20)
    plt.plot(x[west], ele[west], "go", markersize=20)
    plt.plot(x[east], ele[east], "mo", markersize=20)
    plt.xlabel("Distance, m", fontsize=16)
    plt.ylabel("Elevation, m", fontsize=16)

    y = np.mean(hts, axis=0)
    yu = y + np.std(hts, axis=0)
    yl = y - np.std(hts, axis=0)
    yuu = np.max(hts, axis=0)
    yll = np.min(hts, axis=0)
    line1.set_data(x, y)
    plt.plot(x, yu, "b--")
    plt.plot(x, yl, "b--")
    plt.fill_between(x, yl, yu, color="blue", alpha=0.1)
    plt.plot(x, yuu, linestyle="dotted")
    plt.plot(x, yll, linestyle="dotted")
    plt.fill_between(x, yll, yuu, color="blue", alpha=0.1)

    t1 = "Ditch depth " + str(h0_west) + str(h0_east) + " m"
    ax.text(0.5, high * 0.95, str(t1), fontsize=16, color="0.25")
    t2 = "Slope " + str(spara.slope) + " %"
    ax.text(0.5, high * 0.9, str(t2), fontsize=16, color="0.25")
    t3 = "Peat type: " + spara.peat_type[0]
    ax.text(0.5, high * 0.85, t3, fontsize=16, color="0.25")
    if type(lai) is float:
        lai = [lai]
    t4 = "LAI " + str(lai)  # str(lai[0]) + '...' + str(lai[-1])
    ax.text(0.5, high * 0.8, t4, fontsize=16, color="0.25")

    ax.add_line(line1)
    ax.legend(loc=1)

    ax2 = fig.add_axes([0.05, 0.1, 0.55, 0.3])  # left, bottom, width, height)
    ax2.set_ylim([h0_west * 2.0, 0.2])
    surf = np.zeros(aa)
    limit2 = -0.5 * np.ones(aa)
    limit3 = -0.35 * np.ones(aa)
    plt.plot(range(aa), limit2, "k--", range(aa), surf, "k-")
    plt.fill_between(range(aa), limit2, surf, color="green", alpha=0.3)
    plt.fill_between(range(aa), limit3, surf, color="red", alpha=0.3)
    range(np.shape(hts)[0])
    yy = hts[:, mid] - ele[mid]
    yywest = hts[:, west] - ele[west]
    yyeast = hts[:, east] - ele[east]
    line3 = Line2D(
        [],
        [],
        color="red",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        linewidth=1,
        label="Mid field wt",
    )
    ax2.add_line(line3)
    line4 = Line2D(
        [],
        [],
        color="green",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        linewidth=0.5,
        label="West wt",
    )
    ax2.add_line(line4)
    line5 = Line2D(
        [],
        [],
        color="m",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        linewidth=0.5,
        label="East wt",
    )
    ax2.add_line(line5)
    ax2.set_xlim([0, aa])
    plt.xlabel("Time, days", fontsize=16)
    plt.ylabel("wt depth, m", fontsize=16)
    ax2.legend(loc=1)
    line3.set_data(range(len(yy)), yy)
    line4.set_data(range(len(yywest)), yywest)
    line5.set_data(range(len(yyeast)), yyeast)

    ax22 = ax2.twinx()
    plt.plot(range(len(runoff)), runoff * 1000.0, color="blue")
    ax22.set_ylim([0.0, max(runoff) * 3.0 * 1000.0])
    plt.fill_between(range(len(runoff)), 0.0, runoff * 1000.0, color="blue", alpha=0.3)
    plt.ylabel("Runoff, mm $day^{-1}$")

    ax3 = fig.add_axes([0.68, 0.75, 0.3, 0.21])  # left, bottom, width, height
    ax3.set_ylim([0, sum(Prec) * 1.1])
    ax3.set_xlim([0, aa])
    x = range(len(runoff))
    z = np.zeros(len(runoff))
    y = np.cumsum(runoff) * 1000.0
    plt.plot(x, y, color="blue")
    plt.fill_between(x, z, y, color="blue", alpha=0.3, label="Cumul R")
    yy = y + np.cumsum(ets)
    plt.plot(x, yy, color="red")
    plt.fill_between(x, y, yy, color="red", alpha=0.3, label="Cumul ET")
    plt.plot(range(len(Prec)), np.cumsum(Prec), color="black", label="Cumul P")
    plt.xlabel("Time, days", fontsize=16)
    plt.ylabel("mm", fontsize=16)
    ax3.legend(loc=1)

    t1 = "Weather from " + wpara["description"]
    ax3.text(10, sum(Prec) * 1.0, str(t1), fontsize=14, color="0.25")
    t2 = "From " + str(wpara["start_yr"]) + " to " + str(wpara["end_yr"])
    ax3.text(10, sum(Prec) * 0.85, t2, fontsize=14, color="0.25")
    t3 = (
        "Rain "
        + str(np.round(sum(Prec) / sim_yrs))
        + ", ET "
        + str(np.round(sum(ets) / sim_yrs))
        + " mm yr-1"
    )
    ax3.text(10, sum(Prec) * 0.7, t3, fontsize=14, color="0.25")
    t4 = "Runoff " + str(np.round(sum(runoff) * 1000.0 / sim_yrs)) + " mm yr-1"
    ax3.text(10, sum(Prec) * 0.55, t4, fontsize=14, color="0.25")

    ax4 = fig.add_axes([0.68, 0.45, 0.3, 0.21])  # left, bottom, width, height
    ax4.set_ylim([-10, 35])
    ax4.set_xlim([0, aa])
    line8 = Line2D(
        [],
        [],
        color="b",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        label="Air temperature, deg C",
    )
    ax4.add_line(line8)
    nolla = np.zeros(aa)
    lo = np.ones(aa) * -10
    plt.fill_between(range(aa), lo, nolla, color="yellow", alpha=0.2)
    line8.set_data(range(len(T)), T)
    plt.xlabel("Time, days", fontsize=16)
    plt.ylabel("deg C", fontsize=16)
    ax4.legend(loc=1)
    t1 = "Mean temperature " + str(np.round(np.mean(T)))
    ax4.text(10, 30, t1, fontsize=14, color="0.25")

    ax5 = fig.add_axes([0.68, 0.1, 0.3, 0.21])  # left, bottom, width, height
    line9 = Line2D(
        [],
        [],
        color="b",
        marker="o",
        markeredgecolor="b",
        markersize=1,
        label="CO2 efflux",
    )
    ax5.set_ylim([-10, 100])
    ax5.set_xlim([0, aa])
    ax5.add_line(line9)
    ax5.legend(loc=1)
    plt.fill_between(range(aa), lo, nolla, color="yellow", alpha=0.2)
    line9.set_data(range(len(het)), het)
    plt.xlabel("Time, days", fontsize=16)
    plt.ylabel("kg/ha", fontsize=16)
    t1 = "Annual CO2 efflux " + str(np.round(np.sum(het) / sim_yrs))
    ax5.text(10, 85, t1, fontsize=14, color="0.25")
    plt.show()


# --- weather_fig -------------------------------------------------------------
# From: src/susi/io/susi_io.py:801-862 at ec9be39
# Dead because: no callers
# Review:
def weather_fig(df):
    sns.set()
    # import string
    # printable = set(string.printable)
    fs = 12
    fig = plt.figure(
        num="Susi - weather data", figsize=[15.0, 8.0], facecolor="#C1ECEC"
    )  # see hex color codes from https://www.rapidtables.com/web/color/html-color-codes.html
    df["Kunta"][0]
    # fig.suptitle('Weather data, '+ filter(lambda x: x in printable, municipality), fontsize=18)
    ax1 = fig.add_axes([0.05, 0.55, 0.6, 0.35])  # left, bottom, width, height
    ax1.plot(df.index, df["Prec"].values, "b-", label="Rainfall")
    ax1.set_xlabel("Time", fontsize=fs)
    ax1.set_ylabel("Rainfall, mm", fontsize=12)
    ax1.legend(loc="upper left")
    ax11 = ax1.twinx()
    ax11.plot(
        df.index,
        np.cumsum(df["Prec"].values),
        "m-",
        linewidth=2.0,
        label="Cumulative rainfall",
    )
    ax11.set_ylabel("Cumulative rainfall [mm]", fontsize=fs)
    ax11.legend(loc="upper right")

    annual_prec = df["Prec"].resample("A").sum()
    ax2 = fig.add_axes([0.73, 0.55, 0.25, 0.35])

    t1 = "Mean annual rainfall " + str(np.round(np.mean(annual_prec.values))) + " mm"
    ax2.set_title(t1, fontsize=14)
    y_pos = np.arange(len(annual_prec))
    plt.bar(y_pos, annual_prec.values, align="center", alpha=0.5)
    plt.xticks(y_pos, annual_prec.index.year, rotation=45)
    ax2.set_ylabel("mm")

    zeroline = np.zeros(len(df.index))
    ax3 = fig.add_axes([0.05, 0.08, 0.6, 0.35])
    ax3.plot(df.index, df["T"], "g", linewidth=0.5)
    ax3.plot(df.index, zeroline, "b-")
    ax3.fill_between(
        df.index, df["T"], 0, where=df["T"] < 0.0, facecolor="b", alpha=0.3
    )
    ax3.fill_between(
        df.index, df["T"], 0, where=df["T"] >= 0.0, facecolor="r", alpha=0.3
    )
    ax3.set_ylabel("Air temperature,  C", fontsize=fs)

    annual_temp = df["T"].resample("A").mean()
    t2 = (
        "Mean annual temperature "
        + str(np.round(np.mean(annual_temp.values), 2))
        + " C"
    )

    ax4 = fig.add_axes([0.73, 0.08, 0.25, 0.35])
    ax4.set_title(t2, fontsize=14)
    y_pos = np.arange(len(annual_temp))
    plt.bar(y_pos, annual_temp.values, align="center", alpha=0.5)
    plt.xticks(y_pos, annual_temp.index.year, rotation=45)
    ax4.set_ylabel(" C", fontsize=fs)
    plt.show()


# --- motti_fig ---------------------------------------------------------------
# From: src/susi/io/susi_io.py:865-909 at ec9be39
# Dead because: no callers
# Review:
def motti_fig(df, ini_age, sim_time):
    sns.set()

    fig = plt.figure(
        num="Susi, stand data", figsize=[15.0, 8.0], facecolor="#C1ECEC"
    )  # see hex color codes from https://www.rapidtables.com/web/color/html-color-codes.html
    fig.suptitle("Stand development (Motti simulation)", fontsize=18)

    ax = plt.subplot(231)
    ax.plot(df["age"], df["hdom"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Dominant height [m]")
    # plt.xlabel('Age, yrs')

    ax = plt.subplot(232)
    ax.plot(df["age"], df["N"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Stem count [$ha^{-1}$]")
    # plt.xlabel('Age, yrs')

    ax = plt.subplot(233)
    ax.plot(df["age"], df["BA"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Basal area [$m^{2}$ $ha^{-1}$]")
    # plt.xlabel('Age, yrs')

    ax = plt.subplot(234)
    ax.plot(df["age"], df["vol"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Tilavuus [$m^{3}$ $ha^{-1}$]")
    # plt.xlabel('Age, yrs')

    ax = plt.subplot(235)
    ax.plot(df["age"], df["Dg"], color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Mean diameter [cm]")
    plt.xlabel("Age, years")

    sla = 6.8
    ax = plt.subplot(236)
    ax.plot(df["age"], df["leaves"] / 10.0 * sla, color="green")
    ax.axvspan(ini_age, ini_age + sim_time, color="red", alpha=0.3)
    plt.ylabel("Leaf area index [m2 $m^{-2}$]")
    plt.xlabel("Age, years")
    plt.show()


# --- print_scenario ----------------------------------------------------------
# From: src/susi/io/susi_io.py:912-925 at ec9be39
# Dead because: no callers
# Review:
def print_scenario(r, co2release, deltas, h0ts, dwts, bmToYi, npps, bm, yi, ets):
    print(
        "  +Heterotrophic respiration", np.round(sum(co2release[r, :]), 2), "kg CO2/ha"
    )
    gwlev = [np.mean((dwts[r, d, :])[1:-1]) for d in range(len(deltas))]
    print("Scenario summary, ditch depth:", h0ts[0], "m")
    print("  +Mean dwt", np.round(np.mean(gwlev), 3), "m")
    print("  +Table growth", yi[-1] - yi[0])
    print(
        "  +Assimilation growth",
        np.round(bmToYi(bm[0] + sum(npps[r, :])) - bmToYi(bm[0]), 3),
    )
    print("  +Evapotranspiration", np.round(sum(ets) * 1000.0))
    print("  +Deltas ", np.round(sum(deltas) * 1000.0, 2))


# --- print_scenario_nodes ----------------------------------------------------
# From: src/susi/io/susi_io.py:928-945 at ec9be39
# Dead because: no callers
# Review:
def print_scenario_nodes(
    r, c_bals, deltas, ets, h0ts_west, h0ts_east, dwts, bmToYi, g_nuts, end_vols
):
    print(
        "Scenario summary, ditch depth:", h0ts_west[0], h0ts_east[0], "m", "round ", r
    )
    # print ('  +C balance',  'kg CO2/ha?',np.round(np.sum(c_bals[r,:]), 2))
    gwlev = [np.mean((dwts[r, d, :])[1:-1]) for d in range(len(deltas))]
    print("  +Mean dwt", np.round(np.mean(gwlev), 3), "m")
    print("  +Evapotranspiration", np.round(sum(ets) * 1000.0), "mm")
    print("  +Deltas ", np.round(sum(deltas) * 1000.0, 2), "mm")
    # print ('  +Table growth', np.round(yi[-1]-yi[0],2), 'm3')
    print(
        "  +Nutrient limited volume",
        np.round(np.mean(g_nuts[r, -1], axis=0), 3),
        np.round(np.std(g_nuts[r, -1], axis=0), 3),
        "(sd)",
    )
