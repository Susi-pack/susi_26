"""
Created on Tue Aug 08 10:38:45 2017

@author: lauren
"""

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as interS
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def peat_hydrol_properties(x, unit="g/cm3", var="bd", ptype="A"):
    """
    Peat water retention and saturated hydraulic conductivity as a function of bulk density
    Päivänen 1973. Hydraulic conductivity and water retention in peat soils. Acta forestalia fennica 129.
    see bulk density: page 48, fig 19; degree of humification: page 51 fig 21
    Hydraulic conductivity (cm/s) as a function of bulk density(g/cm3), page 18, as a function of degree of humification see page 51
    input:
        - x peat inputvariable in: db, bulk density or dgree of humification (von Post)  as array \n
        - bulk density unit 'g/cm3' or 'kg/m3' \n
        - var 'db' if input variable is as bulk density, 'H' if as degree of humification (von Post) \n
        - ptype peat type: 'A': all, 'S': sphagnum, 'C': Carex, 'L': wood, list with length of x
    output: (ThetaS and ThetaR in m3 m-3)
        van Genuchten water retention parameters as array [ThetaS, ThetaR, alpha, n] \n
        hydraulic conductivity (m/s)
    """
    # paras is dict variable, parameter estimates are stored in tuples, the model is water content = a0 + a1x + a2x2, where x is
    para = {}  #'bd':bulk density in g/ cm3; 'H': von Post degree of humification
    para["bd"] = {
        "pF0": (97.95, -79.72, 0.0),
        "pF1.5": (20.83, 759.69, -2484.3),
        "pF2": (3.81, 705.13, -2036.2),
        "pF3": (9.37, 241.69, -364.6),
        "pF4": (-0.06, 249.8, -519.9),
        "pF4.2": (0.0, 174.48, -348.9),
    }
    para["H"] = {
        "pF0": (95.17, -1.26, 0.0),
        "pF1.5": (46.20, 8.32, -0.54),
        "pF2": (27.03, 8.14, -0.43),
        "pF3": (17.59, 3.22, -0.07),
        "pF4": (8.81, 3.03, -0.10),
        "pF4.2": (5.8, 2.27, -0.08),
    }

    intp_pF1 = {}  # interpolation functions for pF1
    intp_pF1["bd"] = interp1d(
        [0.04, 0.08, 0.1, 0.2], [63.0, 84.0, 86.0, 80.0], fill_value="extrapolate"
    )
    intp_pF1["H"] = interp1d(
        [1.0, 4.0, 6.0, 10.0], [75.0, 84.0, 86.0, 80.0], fill_value="extrapolate"
    )

    # Saturatated hydraulic conductivity parameters
    Kpara = {
        "bd": {
            "A": (-2.271, -9.80),
            "S": (-2.321, -13.22),
            "C": (-1.921, -10.702),
            "L": (-1.921, -10.702),
        },
        "H": {
            "A": (-2.261, -0.205),
            "S": (-2.471, -0.253),
            "C": (-1.850, -0.278),
            "L": (-2.399, -0.124),
        },
    }

    vg_ini = (
        0.88,
        0.09,
        0.03,
        1.3,
    )  # initial van Genuchten parameters (porosity, residual water content, alfa, n)

    x = np.array(x)
    prs = para[var]
    pF1 = intp_pF1[var]
    if unit == "kg/m3" and var == "db":
        x = x / 1000.0
    if np.shape(x)[0] > 1 and len(ptype) == 1:
        ptype = np.repeat(ptype, np.shape(x)[0])
    vgen = np.zeros((np.size(x), 4))
    Ksat = np.zeros(np.size(x))

    # wcont = lambda x, (a0, a1, a2): a0 + a1*x + a2*x**2.
    def wcont(x, *a):
        return a[0] + a[1] * x + a[2] * x**2.0

    def van_g(pot, *p):
        return p[1] + (p[0] - p[1]) / (1.0 + (p[2] * pot) ** p[3]) ** (1.0 - 1.0 / p[3])

    # K = lambda x, (a0, a1): 10.**(a0 + a1*x) / 100.   # to m/s
    def K(x, *a):
        return 10.0 ** (a[0] + a[1] * x) / 100.0  # to m/s

    potentials = np.array([0.01, 10.0, 32.0, 100.0, 1000.0, 10000.0, 15000.0])

    wc = (
        np.array(
            [
                wcont(x, *prs["pF0"]),
                pF1(x),
                wcont(x, *prs["pF1.5"]),
                wcont(x, *prs["pF2"]),
                wcont(x, *prs["pF3"]),
                wcont(x, *prs["pF4"]),
                wcont(x, *prs["pF4.2"]),
            ]
        )
    ) / 100.0

    for i, s in enumerate(np.transpose(wc)):
        try:
            vgen[i], _ = curve_fit(van_g, potentials, s, p0=vg_ini)
        except RuntimeError:  # curve_fit's "optimal parameters not found"
            print(
                "water retention parameters did not converge, Replaced with generic velues"
            )
            vgen[i] = np.array([0.88, 0.09, 0.03, 1.3])  # van Genuchten parameters

    for i, a, pt in zip(range(len(x)), x, ptype):
        Ksat[i] = K(a, *Kpara[var][pt])  # hydraulic conductivity (cm/s -> m/s)

    return vgen, Ksat


def CWTr(nLyrs, z, dz, pF, Ksat, direction="positive"):
    """
    Returns interpolation functions
        sto=f(gwl)  profile water storage as a function ofground water level
        gwl=f(sto)  ground water level
        tra=f(gwl)  transissivity
    Input:
        nLyrs number of soil layers
        d depth of layer midpoint
        dz layer thickness
        pF van Genuchten water retention parameters: ThetaS, ThetaR, alfa, n
        Ksat saturated hydraulic conductivity in m s-1
        direction: positive or negative downwards
    """
    # -------Parameters ---------------------
    z = np.array(z)
    dz = np.array(dz)
    nroot = 8  # 8 good number of layers in rooting zone
    nroot2 = 2  # 10 cm root layers for air-filled porosity

    # --------- Connection between gwl and water storage------------
    d = 6 if direction == "positive" else -6
    gwl = np.linspace(0, d, 150)
    if direction == "positive":
        sto = [
            sum(wrc(pF, x=np.minimum(z - g, 0.0)) * dz) for g in gwl
        ]  # equilibrium head m
        storoot = [
            np.sum(wrc(pF, x=np.minimum(z - g, 0.0))[0:nroot] * dz[0:nroot])
            for g in gwl
        ]
        storoot2 = [
            np.sum(wrc(pF, x=np.minimum(z - g, 0.0))[0:nroot2] * dz[0:nroot2])
            for g in gwl
        ]
    else:
        sto = [
            sum(wrc(pF, x=np.minimum(z + g, 0.0)) * dz) for g in gwl
        ]  # equilibrium head m
        storoot = [
            np.sum(wrc(pF, x=np.minimum(z + g, 0.0))[0:nroot] * dz[0:nroot])
            for g in gwl
        ]
        storoot2 = [
            np.sum(wrc(pF, x=np.minimum(z + g, 0.0))[0:nroot2] * dz[0:nroot2])
            for g in gwl
        ]

    gwlToSto = interp1d(np.array(gwl), np.array(sto), fill_value="extrapolate")
    airtot = sto[0] - sto  # m air in the profile
    airroot = storoot[0] - storoot  # m air in the rooting zone
    afproot = (storoot2[0] - storoot2) / (
        sum(dz[:nroot2])
    )  # air-filled porosity in root layer
    ratio = (
        airroot[1:] / airtot[1:]
    )  # share of air-filled porosity in rooting zone to total air volume
    sto = list(sto)
    gwl = list(gwl)
    ratio = list(ratio)
    afproot = list(afproot)
    sto.reverse()
    gwl.reverse()
    ratio.reverse()
    afproot.reverse()
    stoToGwl = interp1d(np.array(sto), np.array(gwl), fill_value="extrapolate")
    gwlToRatio = interp1d(np.array(gwl[1:]), np.array(ratio), fill_value="extrapolate")
    gwlToAfp = interp1d(np.array(gwl), np.array(afproot), fill_value="extrapolate")
    C = interp1d(
        np.array(gwl),
        np.array(np.gradient(gwlToSto(gwl)) / np.gradient(gwl)),
        fill_value="extrapolate",
    )  # storage coefficient function

    del gwl, sto, ratio, afproot

    # ----------Transmissivity-------------------
    K = np.array(Ksat * 86400.0)  # from m/s to m/day
    tr = [sum(K[t:] * dz[t:]) for t in range(nLyrs)]
    if direction == "positive":
        gwlToTra = interS(z, np.array(tr))
    else:
        z = list(z)
        z.reverse()
        tr.reverse()
        gwlToTra = interS(-np.array(z), np.array(tr))
    del tr
    return gwlToSto, stoToGwl, gwlToTra, C, gwlToRatio, gwlToAfp


def wrc(pF, x=None, var=None):
    """
    vanGenuchten-Mualem soil water retention curve\n
    IN:
        pF - dict['ThetaS': ,'ThetaR': ,'alpha':, 'n':,] OR
           - list [ThetaS, ThetaR, alpha, n]
        x  - soil water tension [m H2O = 0.1 kPa]
           - volumetric water content [vol/vol]
        var-'Th' is x=vol. wat. cont.
    OUT:
        res - Theta(Psii) or Psii(Theta)
    NOTE:\n
        sole input 'pF' draws water retention curve and returns 'None'. For drawing give only one pF-parameter set.
        if several pF-curves are given, x can be scalar or len(x)=len(pF). In former case var is pF(x), in latter var[i]=pf[i,x[i]]

    Samuli Launiainen, Luke 2/2016
    """
    if type(pF) is dict:  # dict input
        Ts = np.array(pF["ThetaS"].values())
        Tr = np.array(pF["ThetaR"].values())
        alfa = np.array(pF["alpha"].values())
        n = np.array(pF["n"].values())
        m = 1.0 - np.divide(1.0, n)
    elif type(pF) is list:  # list input
        pF = np.array(
            pF, ndmin=1
        )  # ndmin=1 needed for indexing to work for 0-dim arrays
        Ts = pF[0]
        Tr = pF[1]
        alfa = pF[2]
        n = pF[3]
        m = 1.0 - np.divide(1.0, n)
    elif type(pF) is np.ndarray:
        Ts, Tr, alfa, n = pF.T[0], pF.T[1], pF.T[2], pF.T[3]
        m = 1.0 - np.divide(1.0, n)
    else:
        print("Unknown type in pF")

    def theta_psi(x):  #'Theta-->Psi'
        x = np.minimum(x, Ts)
        x = np.maximum(x, Tr)  # checks limits
        s = (Ts - Tr) / (x - Tr)  # **(1/m)
        Psi = -1e-2 / alfa * (s ** (1 / m) - 1) ** (1 / n)  # in m
        return Psi

    def psi_theta(x):  # 'Psi-->Theta'
        x = 100 * np.minimum(x, 0)  # cm
        Th = Tr + (Ts - Tr) / (1 + abs(alfa * x) ** n) ** m
        return Th

    if var == "Th":
        y = theta_psi(x)  #'Theta-->Psi'
    else:
        y = psi_theta(x)  # 'Psi-->Theta'
    return y


def read_FMI_weather(ID, start_date, end_date, sourcefile=None) -> pd.DataFrame:
    """
    reads FMI interpolated daily weather data from file
    IN:
        ID - sve catchment ID. set ID=0 if all data wanted
        start_date - 'yyyy-mm-dd'
        end_date - 'yyyy-mm-dd'
    OUT:
        fmi - pd.dataframe with datetimeindex
            fmi columns:['ID','Kunta','aika','lon','lat','T','Tmax','Tmin','Prec','Rg','h2o','dds','Prec_a','Par','RH','esa','VPD','doy']
            units: T, Tmin, Tmax, dds[degC], VPD, h2o,esa[kPa], Prec, Prec_a[mm], Rg,Par[Wm-2],lon,lat[deg]
    """

    # OmaTunniste;OmaItä;OmaPohjoinen;Kunta;siteid;vuosi;kk;paiva;longitude;latitude;t_mean;t_max;t_min;
    # rainfall;radiation;hpa;lamposumma_v;rainfall_v;lamposumma;lamposumma_cum
    # -site number
    # -date (yyyy mm dd)
    # -latitude (in KKJ coordinates, metres)
    # -longitude (in KKJ coordinates, metres)
    # -T_mean (degrees celcius)
    # -T_max (degrees celcius)
    # -T_min (degrees celcius)
    # -rainfall (mm)
    # -global radiation (per day in kJ/m2)
    # -H2O partial pressure (hPa)

    ID = 0
    print(" + Reading meteorological input file from ")
    print("    -", sourcefile)
    ID = 0
    # import forcing data

    fmi = pd.read_csv(
        sourcefile,
        sep=";",
        header="infer",
        usecols=[
            "OmaTunniste",
            "Kunta",
            "aika",
            "longitude",
            "latitude",
            "t_mean",
            "t_max",
            "t_min",
            "rainfall",
            "radiation",
            "hpa",
        ],
        encoding="ISO-8859-1",
    )

    # print pd.to_datetime(fmi['aika'][0], format="%Y%m%d")
    # print pd.tseries.tools.to_datetime(fmi['aika'][0], format="%Y%m%d")
    time = pd.to_datetime(fmi["aika"], format="%Y%m%d")

    fmi.index = time
    fmi = fmi.rename(
        columns={
            "OmaTunniste": "ID",
            "longitude": "lon",
            "latitude": "lat",
            "t_mean": "T",
            "t_max": "Tmax",
            "t_min": "Tmin",
            "rainfall": "Prec",
            "radiation": "Rg",
            "hpa": "h2o",
        }
    )

    fmi["h2o"] = 1e-1 * fmi["h2o"]  # hPa-->kPa
    fmi["Rg"] = 1e3 / 86400.0 * fmi["Rg"]  # kJ/m2/d-1 to Wm-2
    fmi["Par"] = 0.5 * fmi["Rg"]

    # saturated vapor pressure
    esa = 0.6112 * np.exp((17.67 * fmi["T"]) / (fmi["T"] + 273.16 - 29.66))  # kPa
    vpd = esa - fmi["h2o"]  # kPa
    vpd[vpd < 0] = 1e-5
    rh = 100.0 * fmi["h2o"] / esa
    rh[rh < 0] = 1e-6
    rh[rh > 100] = 100.0

    fmi["RH"] = rh
    fmi["esa"] = esa
    fmi["vpd"] = vpd
    fmi["doy"] = fmi.index.dayofyear
    fmi = fmi.drop(columns=["aika"])

    # replace nan's in prec with 0.0
    fmi["Prec"].fillna(value=0.0)

    # get desired period
    fmi = fmi[(fmi.index >= start_date) & (fmi.index <= end_date)]
    if ID > 0:
        fmi = fmi[fmi["ID"] == ID]

    return fmi


def rew(dwt):
    """Alla hyväksi havaitut"""
    wt = np.array([-150.0, -1.0, -0.5, -0.3, -0.1, 0.0])  # water table in m
    re = np.array([0.0, 0.1, 0.4, 1.0, 1.0, 0.7])  # aml relative water uptake

    frew = interp1d(wt, re, fill_value="extrapolate")
    return frew(dwt)


def rew_drylimit(dwt):
    # Koivusalo et al. 2008 HESS without wet side limit
    wt = np.array([-150.0, -1.0, -0.5, -0.3, -0.1, 0.0])  # aml water table in m
    re = np.array([0.0, 0.1, 0.4, 1.0, 1.0, 0.7])  # aml relative water uptake

    frew = interp1d(wt, re, fill_value="extrapolate")
    return frew(dwt)


def assimilation_yr(photopara, dfforc, wt, afp, LAI, LAI_above, NPP_share):
    """
    Computes photosynthesis and respiration of the stand in daily time step
    Mäkelä et al. 2008. Empirical model of stand GPP LUE approach. Global Change Biology 14: 92-108
    IN:
        (develpment of all sided leaf area index (LAI) m2 m-2)
        LAI, double sided LAI m2 m-2
        meteorological input data:
        rg in W/m2
        vpd in kPa
        Ta previous day tamperature
        Ta  in deg C
        Xk acclimatisation temperature, Ta in the first time step
        Ns,Ps,Ks nutrient release in time step kg/ha/timestep
        Nconc, Pconc, Kconc nutrient concentration in stand
        hdom dominant height of the stand (used in computation of respiration)
        wt water table [m] negative down, as pandas dataframe shape (days, nodes)
        NPP_share divides the calculated GPP into NPP and respiration.
        The higher the trees, the smaller NPP share
    OUT:
        Net primary production NPP in kg/ha/yr
        Xa
    """

    attenuation = np.exp(-0.2 * LAI_above)
    days, ncols = np.shape(wt)
    par = np.zeros((ncols, 366))
    rg, vpd, Ta = dfforc["Rg"], dfforc["vpd"], dfforc["T"]

    for m in range(ncols):
        par[m, 0:days] = (
            rg * 4.6 * 0.5 / 1000000.0 * 86400.0 * attenuation[m]
        )  # Unit conversion to mol/m2/day, 0.5 is the share of par from rg

    """ Eq 2 """
    fL = np.zeros((ncols, 366))
    for m in range(ncols):
        fL[m:days] = 1.0 / (photopara.gamma * par[m] + 1.0)  # Light modifier

    """Eq. 3a and 3b"""
    sim_len = len(Ta)
    Xk = np.zeros(sim_len)  # Acclimatiosation: Temerature adjustment with delay tau
    Xk[0] = Ta.iloc[0]
    for n, T in enumerate(Ta[0:-1]):
        Xk[n + 1] = Xk[n] + (T - Xk[n]) / photopara.tau
    Sk = np.maximum(Xk - photopara.X0, np.zeros(sim_len))

    """Eq 4"""
    fs = np.zeros(366)
    fs[: len(Sk)] = np.minimum(Sk / photopara.Smax, 1.0)  # Final temperature adustment

    """Eq 5"""
    fd = np.ones(
        366
    )  # Vapor pressure deficit function: limits photos when stomata are closed
    fd[: len(vpd)] = np.exp(photopara.kappa * vpd)  # Eq 5

    # ************* From here on: different in different columns********************************
    """Eq 6"""  # Soil water content modifyer
    rew = rew_drylimit(wt)
    days, ncols = np.shape(rew)
    rewarr = np.zeros((366, ncols))
    rewarr[:days, :] = rew

    """air filled porosity function"""

    afptmp = np.array([1.0, 0.1, 0.0])  # afp in root zone
    afp_response = np.array([1.0, 1.0, 0.0])  # relative water uptake
    fafp = interp1d(afptmp, afp_response, fill_value="extrapolate")
    f_fafp = fafp(afp)
    fafparr = np.zeros((366, ncols))
    fafparr[:days, :] = f_fafp

    npp_arr = np.zeros(ncols)
    npp_arr_pot = np.zeros(ncols)

    for column in range(ncols):
        fREW = (1.0 + ((1.0 - rewarr[:, column]) / photopara.alfa) ** photopara.nu) ** (
            -1.0
        )
        fAFP = fafparr[:, column]
        """ Beer-Lambert function: LAI function"""  # Fabrika 2013
        kext = 0.2

        """Eq 1"""
        Pk = (
            photopara.beta
            * (1.0 - np.exp(-kext * LAI[column]))
            * par[column]
            * fL[column]
            * fs
            * fd
            * fREW
            * fAFP
        )  # canopy GPP gC m-2 day-1
        Pk_pot = (
            photopara.beta
            * (1.0 - np.exp(-kext * LAI[column]))
            * par[column]
            * fL[column]
            * fs
            * fd
            * 1.0
            * 1.0
        )  # growth that is not restricted by edaphic factors gC m-2 day-1

        """Respi"""
        NPP = Pk * NPP_share[column] * 2.0
        NPP_pot = (
            Pk_pot * NPP_share[column] * 2.0
        )  # Change from gC m-2 to g organic matter
        npp_arr[column] = sum(NPP)
        npp_arr_pot[column] = sum(NPP_pot)

    return npp_arr * 10.0, npp_arr_pot * 10.0  # kg/ha organic matter


def heterotrophic_respiration_yr(t5, yr, dfwt, v, spara, sfc):
    """
    Output:
        mean time series kg CO2 ha-1 day-1 and annual sum for each computation  node
    """
    sfc = np.median(sfc)
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
    _n, days = np.shape(Rhet)

    return (
        days,
        np.mean(Rhet, axis=1) * 10.0,
        np.sum(Rhet, axis=0) * 10.0,
    )  # , np.sum(Rhet_root, axis=0)*10.


def ojanen_2019(spara, yr, dfwt, sfc):
    wts = (
        dfwt[str(yr) + "-05-01" : str(yr) + "-10-31"].mean().values * -100.0
    )  # .values[:-1])
    sfc = np.median(sfc)
    if sfc < 3:
        soil_co2_balance = (
            (-115 + 12 * wts) * 10 * -1
        )  # conversion from g/m2/yr to kg co2/ha/ yr and sign conversion: source negative; sink positive
    else:
        soil_co2_balance = (-259 + 6 * wts) * 10 * -1
    return soil_co2_balance


def get_temp_sum(forc):
    """
    Cumulative temperature sum degree days
    """
    base = 5.0
    dd = forc["T"] - base
    dd[dd < 0.0] = 0.0
    dds = dd.sum(axis=0)  # cumulative temperature sum degree days
    yrs = np.shape(forc)[0] / 365.25
    ts = dds / yrs
    return ts
