"""
Created on Fri Mar 24 11:01:50 2017

@author: slauniai

******************************************************************************
CanopyGrid:

Gridded canopy and snow hydrology model for SpaFHy -integration
Based on simple schemes for computing water flows and storages within vegetation
canopy and snowpack at daily or sub-daily timesteps.

(C) Samuli Launiainen, 2016-
last edit: Oct 2018 / Samuli
******************************************************************************

"""

import copy

import numpy as np

eps = np.finfo(float).eps


class CanopyGrid:
    def __init__(self, cpara, state, outputs=False):
        """
        initializes CanopyGrid -object

        Args:
            cpara - parameter dict:
            state - dict of initial state
            outputs - True saves output grids to list at each timestep

        Returns:
            self - object

        NOTE:
            Currently the initialization assumes simulation start 1st Jan,
            and sets self._LAI_decid and self.X equal to minimum values.
            Also leaf-growth & senescence parameters are intialized to zero.
        """
        epsi = 0.01

        # physiology: transpi + floor evap
        self.amax = copy.deepcopy(cpara.physpara.amax_init)

        # canopy parameters and state
        self.hc = state.hc + epsi
        self.cf = state.cf + epsi

        self._LAIconif = state.lai_conif + epsi  # m2m-2
        self._LAIdecid = state.lai_decid_max
        self.LAI = self._LAIconif + self._LAIdecid

        self._LAIdecid_max = state.lai_decid_max + epsi  # m2m-2

        # --- state variables
        self.W = np.minimum(state.w, cpara.interception.wmax * self.LAI)
        self.SWE = state.swe
        self.SWEi = self.SWE
        self.SWEl = np.zeros(np.shape(self.SWE))

        # NOTE: this assumes simulations start 1st Jan each year !!!
        self.X = 0.0

        # create dictionary of empty lists for saving results
        if outputs:
            self.results = {
                "PotInf": [],
                "Trfall": [],
                "Interc": [],
                "Evap": [],
                "ET": [],
                "Transpi": [],
                "Efloor": [],
                "SWE": [],
                "LAI": [],
                "Mbe": [],
                "LAIdecid": [],
                "erate": [],
                "Unload": [],
                "fact": [],
            }

    def update_amax(self, nutstat):
        self.amax = self.amax * 1.0 + (nutstat - 1)  # 1.035

    def run_timestep(
        self,
        cpara,
        doy,
        dt,
        Ta,
        Prec,
        Rg,
        Par,
        VPD,
        hc,
        LAIconif,
        U=2.0,
        CO2=380.0,
        Rew=1.0,
        beta=1.0,
        P=101300.0,
    ):
        """
        Runs CanopyGrid instance for one timestep
        IN:
            doy - day of year
            dt - timestep [s]
            Ta - air temperature  [degC], scalar or (n x m) -matrix
            prec - precipitatation rate [mm/s]
            Rg - global radiation [Wm-2], scalar or matrix
            Par - photos. act. radiation [Wm-2], scalar or matrix
            VPD - vapor pressure deficit [kPa], scalar or matrix
            U - mean wind speed at ref. height above canopy top [ms-1], scalar or matrix
            CO2 - atm. CO2 mixing ratio [ppm]
            Rew - relative extractable water [-], scalar or matrix
            beta - term for soil evaporation resistance (Wliq/FC) [-]
            P - pressure [Pa], scalar or matrix
        OUT:
            updated CanopyGrid instance state variables
            flux grids PotInf, Trfall, Interc, Evap, ET, MBE [m]
        """
        # SL 18.9.2018: tässä oli kaiken pahan alku ja juuri: pitää päivittää myös tilanmuuttujat self._LAIconif ja self._LAIdecid!
        # näitä käytetään kun lasketaan gsref
        # muutin kutsua niin että siinä annetaan LAIconif ja päivitetään LAI sen kautta. Oletus että
        self._LAIconif = LAIconif  # oletus on että _LAIdecid = vakio == 0.01
        self.LAI = self._LAIconif + self._LAIdecid
        self.hc = hc

        Rn = (
            np.maximum(2.57 * self.LAI / (2.57 * self.LAI + 0.57) - 0.2, 0.55) * Rg
        )  # Launiainen et al. 2016 GCB, fit to Fig 2a

        """ --- update phenology: self.ddsum & self.X ---"""
        fPheno = self._photoacclim(Ta, phenopara=cpara.phenology)

        """ --- aerodynamic conductances --- """
        Ra, _Rb, Ras, _ustar, _Uh, _Ug = aerodynamics(
            self.LAI,
            self.hc,
            U,
            w=0.01,
            zm=cpara.flow.zmeas,
            zg=cpara.flow.zground,
            zos=cpara.flow.zo_ground,
        )

        """ --- interception, evaporation and snowpack --- """
        PotInf, Trfall, Evap, Interc, MBE, erate, unload, fact = self.canopy_water_snow(
            cpara, dt, Ta, Prec, Rn, VPD, Ra=Ra
        )

        """--- dry-canopy evapotranspiration [mm s-1] --- """
        Transpi, Efloor, _Gc = self.dry_canopy_et(
            cpara=cpara,
            D=VPD,
            Qp=Par,
            AE=Rn,
            Ta=Ta,
            Ra=Ra,
            Ras=Ras,
            CO2=CO2,
            Rew=Rew,
            beta=beta,
            fPheno=fPheno,
        )

        Transpi = Transpi * dt
        Efloor = Efloor * dt
        ET = Transpi + Efloor

        # append results to lists; use only for testing small grids!
        if hasattr(self, "results"):
            self.results["PotInf"].append(PotInf)
            self.results["Trfall"].append(Trfall)
            self.results["Interc"].append(Interc)
            self.results["Evap"].append(Evap)
            self.results["ET"].append(ET)
            self.results["Transpi"].append(Transpi)
            self.results["Efloor"].append(Efloor)
            self.results["SWE"].append(self.SWE)
            self.results["LAI"].append(self.LAI)
            self.results["Mbe"].append(np.nanmax(MBE))
            self.results["LAIdecid"].append(self._LAIdecid)
            self.results["erate"].append(erate)
            self.results["Unload"].append(unload)
            self.results["fact"].append(fact)

        return (
            PotInf * 1e-3,
            Trfall * 1e-3,
            Interc * 1e-3,
            Evap * 1e-3,
            ET * 1e-3,
            Transpi * 1e-3,
            Efloor * 1e-3,
            MBE * 1e-3,
            self.SWE,
        )  # fQ,fD,fRew

    def _photoacclim(self, T, phenopara):
        """
        computes new stage of temperature acclimation and phenology modifier.
        Peltoniemi et al. 2015 Bor.Env.Res.
        IN: object, T = daily mean air temperature,
            phenopara parameters
        OUT: fPheno - phenology modifier [0...1], updates object state
        """

        self.X = self.X + 1.0 / phenopara.tau * (T - self.X)  # degC
        S = np.maximum(self.X - phenopara.xo, 0.0)
        fPheno = np.maximum(phenopara.fmin, np.minimum(S / phenopara.smax, 1.0))
        return fPheno

    def dry_canopy_et(
        self,
        cpara,
        D,
        Qp,
        AE,
        Ta,
        Ra=25.0,
        Ras=250.0,
        CO2=380.0,
        Rew=1.0,
        beta=1.0,
        fPheno=1.0,
    ):
        """
        Computes ET from 2-layer canopy in absense of intercepted precipitiation,
        i.e. in dry-canopy conditions
        IN:
           self - object
           cpara - canopy parameters
           D - vpd in kPa
           Qp - PAR in Wm-2
           AE - available energy in Wm-2
           Ta - air temperature degC
           Ra - aerodynamic resistance (s/m)
           Ras - soil aerodynamic resistance (s/m)
           CO2 - atm. CO2 mixing ratio (ppm)
           Rew - relative extractable water [-]
           beta - relative soil conductance for evaporation [-]
           fPheno - phenology modifier [-]
        Args:
           Tr - transpiration rate (mm s-1)
           Efloor - forest floor evaporation rate (mm s-1)
           Gc - canopy conductance (integrated stomatal conductance)  (m s-1)
        SOURCES:
        Launiainen et al. (2016). Do the energy fluxes and surface conductance
        of boreal coniferous forests in Europe scale with leaf area?
        Global Change Biol.
        Modified from: Leuning et al. 2008. A Simple surface conductance model
        to estimate regional evaporation using MODIS leaf area index and the
        Penman-Montheith equation. Water. Resources. Res., 44, W10419
        Original idea Kelliher et al. (1995). Maximum conductances for
        evaporation from global vegetation types. Agric. For. Met 85, 135-147

        Samuli Launiainen, Luke
        Last edit: 13.6.2018: TESTING UPSCALING
        """

        # ---Amax and g1 as LAI -weighted average of conifers and decid.

        rhoa = 101300.0 / (8.31 * (Ta + 273.15))  # mol m-3

        Amax = (
            1.0 / self.LAI * (self._LAIconif * self.amax + self._LAIdecid * self.amax)
        )  # umolm-2s-1

        g1 = (
            1.0
            / self.LAI
            * (
                self._LAIconif * cpara.physpara.g1_conif
                + self._LAIdecid * cpara.physpara.g1_decid
            )
        )

        kp = cpara.physpara.kp  # (-) attenuation coefficient for PAR
        q50 = cpara.physpara.q50  # Wm-2, half-sat. of leaf light response

        tau = np.exp(-kp * self.LAI)  # fraction of Qp at ground relative to canopy top

        """--- canopy conductance Gc (integrated stomatal conductance)----- """

        # fQ: Saugier & Katerji, 1991 Agric. For. Met., eq. 4. Leaf light response = Qp / (Qp + q50)
        fQ = (
            1.0
            / kp
            * np.log((kp * Qp + q50) / (kp * Qp * np.exp(-kp * self.LAI) + q50 + eps))
        )
        # Leuning et al., 2008 WRR use a different fQ for daily Gc; they refer to
        # Kelliher et al. 1995 AFM but their equation is not the exact integral of K95.

        # Soil moisture response (Lagergren & Lindroth, xxxx): the REW limitation of transpiration
        # (Physpara.rw, rwmin) is switched off, so fRew = Rew. See #307 A2.
        fRew = Rew

        # CO2 -response of canopy conductance, derived from APES-simulations
        # (Launiainen et al. 2016, Global Change Biology). relative to 380 ppm
        fCO2 = 1.0 - 0.387 * np.log(CO2 / 380.0)

        # leaf level light-saturated gs (m/s)
        gs = 1.6 * (1.0 + g1 / np.sqrt(D)) * Amax / CO2 / rhoa

        # canopy conductance
        Gc = gs * fQ * fRew * fCO2 * fPheno
        Gc[np.isnan(Gc)] = eps

        """ --- transpiration rate --- """
        Tr = penman_monteith((1.0 - tau) * AE, 1e3 * D, Ta, Gc, 1.0 / Ra, units="mm")
        Tr[Tr < 0] = 0.0

        """--- forest floor evaporation rate--- """
        # beta = Wliq / FC; Best et al., 2011 Geosci. Model. Dev. JULES
        Gcs = cpara.physpara.gsoil

        Efloor = beta * penman_monteith(
            tau * AE, 1e3 * D, Ta, Gcs, 1.0 / Ras, units="mm"
        )
        Efloor[self.SWE > 0] = (
            0.0  # no evaporation from floor if snow on ground or beta == 0
        )

        return Tr, Efloor, Gc

    def canopy_water_snow(self, cpara, dt, T, Prec, AE, D, Ra=25.0, U=2.0):
        """
        Calculates canopy water interception and SWE during timestep dt
        Args:
            self - object
            cpara - canopy parameters
            dt - timestep [s]
            T - air temperature (degC)
            Prec - precipitation rate during (mm d-1)
            AE - available energy (~net radiation) (Wm-2)
            D - vapor pressure deficit (kPa)
            Ra - canopy aerodynamic resistance (s m-1)
        Returns:
            self - updated state W, Wf, SWE, SWEi, SWEl
            Infil - potential infiltration to soil profile (mm)
            Evap - evaporation / sublimation from canopy store (mm)
            MBE - mass balance error (mm)
        Samuli Launiainen & Ari Laurén 2014 - 2017
        Last edit 12 / 2017
        """

        # quality of precipitation
        Tmin = 0.0  # 'C, below all is snow
        Tmax = 1.0  # 'C, above all is water
        Tmelt = 0.0  # 'C, T when melting starts

        # storage capacities mm
        Wmax = cpara.interception.wmax * self.LAI
        Wmaxsnow = cpara.interception.wmaxsnow * self.LAI

        # melting/freezing coefficients mm/s
        Kmelt = cpara.snow.kmelt - 1.64 * self.cf / dt  # Kuusisto E, 'Lumi Suomessa'
        Kfreeze = cpara.snow.kfreeze

        kp = cpara.physpara.kp
        tau = np.exp(-kp * self.LAI)  # fraction of Rn at ground

        # inputs to arrays, needed for indexing later in the code
        gridshape = np.shape(self.LAI)  # rows, cols

        if np.shape(T) != gridshape:
            T = np.ones(gridshape) * T
            Prec = np.ones(gridshape) * Prec
            AE = np.ones(gridshape) * AE
            D = np.ones(gridshape) * D
            Ra = np.ones(gridshape) * Ra

        Prec = Prec * dt  # mm
        # latent heat of vaporization (Lv) and sublimation (Ls) J kg-1
        Lv = 1e3 * (3147.5 - 2.37 * (T + 273.15))
        Ls = Lv + 3.3e5

        # compute 'potential' evaporation / sublimation rates for each grid cell
        erate = np.zeros(gridshape)
        ixs = np.where((Prec == 0) & (T <= Tmin))
        ixr = np.where((Prec == 0) & (T > Tmin))
        Ga = 1.0 / Ra  # aerodynamic conductance

        # resistance for snow sublimation adopted from:
        # Pomeroy et al. 1998 Hydrol proc; Essery et al. 2003 J. Climate;
        # Best et al. 2011 Geosci. Mod. Dev.
        # ri = (2/3*rhoi*r**2/Dw) / (Ce*Sh*W) == 7.68 / (Ce*Sh*W

        Ce = 0.01 * ((self.W + eps) / Wmaxsnow) ** (-0.4)  # exposure coeff (-)
        Sh = 1.79 + 3.0 * U**0.5  # Sherwood numbner (-)
        gi = Sh * self.W * Ce / 7.68 + eps  # m s-1
        erate[ixs] = (
            dt
            / Ls[ixs]
            * penman_monteith(
                (1.0 - tau[ixs]) * AE[ixs],
                1e3 * D[ixs],
                T[ixs],
                gi[ixs],
                Ga[ixs],
                units="W",
            )
        )

        # evaporation of intercepted water, mm
        gs = 1e6
        erate[ixr] = (
            dt
            / Lv[ixr]
            * penman_monteith(
                (1.0 - tau[ixr]) * AE[ixr], 1e3 * D[ixr], T[ixr], gs, Ga[ixr], units="W"
            )
        )

        # ---state of precipitation [as water (fW) or as snow(fS)]
        fW = np.zeros(gridshape)
        fS = np.zeros(gridshape)

        fW[T >= Tmax] = 1.0
        fS[T <= Tmin] = 1.0

        ix = np.where((T > Tmin) & (T < Tmax))
        fW[ix] = (T[ix] - Tmin) / (Tmax - Tmin)
        fS[ix] = 1.0 - fW[ix]
        del ix

        # --- Local fluxes (mm)
        Unload = np.zeros(gridshape)  # snow unloading
        Interc = np.zeros(gridshape)  # interception
        Melt = np.zeros(gridshape)  # melting
        Freeze = np.zeros(gridshape)  # freezing
        Evap = np.zeros(gridshape)

        """ --- initial conditions for calculating mass balance error --"""
        Wo = self.W  # canopy storage
        SWEo = self.SWE  # Snow water equivalent mm

        """ --------- Canopy water storage change -----"""
        # snow unloading from canopy, ensures also that seasonal LAI development does
        # not mess up computations
        ix = T >= Tmax
        Unload[ix] = np.maximum(self.W[ix] - Wmax[ix], 0.0)
        self.W = self.W - Unload
        del ix

        # Interception of rain or snow: asymptotic approach of saturation.
        # Hedstrom & Pomeroy 1998. Hydrol. Proc 12, 1611-1625;
        # Koivusalo & Kokkonen 2002 J.Hydrol. 262, 145-164.
        ix = T < Tmin
        Interc[ix] = (Wmaxsnow[ix] - self.W[ix]) * (
            1.0 - np.exp(-(self.cf[ix] / Wmaxsnow[ix]) * Prec[ix])
        )
        del ix

        # above Tmin, interception capacity equals that of liquid precip
        ix = T >= Tmin
        Interc[ix] = np.maximum(0.0, (Wmax[ix] - self.W[ix])) * (
            1.0 - np.exp(-(self.cf[ix] / Wmax[ix]) * Prec[ix])
        )
        del ix
        self.W = self.W + Interc  # new canopy storage, mm

        Trfall = Prec + Unload - Interc  # Throughfall to field layer or snowpack

        # evaporate from canopy and update storage
        Evap = np.minimum(erate, self.W)  # mm
        self.W = self.W - Evap

        """ Snowpack (in case no snow, all Trfall routed to floor) """
        ix = np.where(T >= Tmelt)
        Melt[ix] = np.minimum(self.SWEi[ix], Kmelt[ix] * dt * (T[ix] - Tmelt))  # mm
        del ix
        ix = np.where(T < Tmelt)
        Freeze[ix] = np.minimum(self.SWEl[ix], Kfreeze * dt * (Tmelt - T[ix]))  # mm
        del ix

        # amount of water as ice and liquid in snowpack
        Sice = np.maximum(0.0, self.SWEi + fS * Trfall + Freeze - Melt)
        Sliq = np.maximum(0.0, self.SWEl + fW * Trfall - Freeze + Melt)

        PotInf = np.maximum(0.0, Sliq - Sice * cpara.snow.r)  # mm
        Sliq = np.maximum(0.0, Sliq - PotInf)  # mm, liquid water in snow

        # update Snowpack state variables
        self.SWEl = Sliq
        self.SWEi = Sice
        self.SWE = self.SWEl + self.SWEi

        # mass-balance error mm
        MBE = (self.W + self.SWE) - (Wo + SWEo) - (Prec - Evap - PotInf)

        return PotInf, Trfall, Evap, Interc, MBE, erate, Unload, fS + fW

    def create_outarrays(self, nrounds, ndays, ncols):
        self.intercs = np.zeros((nrounds, ndays, ncols), dtype=float)
        self.evaps = np.zeros((nrounds, ndays, ncols), dtype=float)
        self.ETs = np.zeros((nrounds, ndays, ncols), dtype=float)
        self.transpis = np.zeros((nrounds, ndays, ncols), dtype=float)
        self.efloors = np.zeros((nrounds, ndays, ncols), dtype=float)
        self.SWEs = np.zeros((nrounds, ndays, ncols), dtype=float)

        return (
            self.intercs,
            self.evaps,
            self.ETs,
            self.transpis,
            self.efloors,
            self.SWEs,
        )

    def update_outarrays(self, scen, d, interc, evap, ET, transpi, efloor, SWE):
        self.intercs[scen, d, :] = interc
        self.evaps[scen, d, :] = evap
        self.ETs[scen, d, :] = ET
        self.transpis[scen, d, :] = transpi
        self.efloors[scen, d, :] = efloor
        self.SWEs[scen, d, :] = SWE

        return (
            self.intercs,
            self.evaps,
            self.ETs,
            self.transpis,
            self.efloors,
            self.SWEs,
        )


""" *********** utility functions ******** """


# @staticmethod
def e_sat(T, P=101300):
    """
    Computes saturation vapor pressure (Pa), slope of vapor pressure curve
    [Pa K-1]  and psychrometric constant [Pa K-1]
    IN:
        T - air temperature (degC)
        P - ambient pressure (Pa)
    OUT:
        esa - saturation vapor pressure in Pa
        s - slope of saturation vapor pressure curve (Pa K-1)
        g - psychrometric constant (Pa K-1)
    """
    NT = 273.15
    cp = 1004.67  # J/kg/K

    Lambda = 1e3 * (3147.5 - 2.37 * (T + NT))  # lat heat of vapor [J/kg]
    esa = 1e3 * (0.6112 * np.exp((17.67 * T) / (T + 273.16 - 29.66)))  # Pa

    s = 17.502 * 240.97 * esa / ((240.97 + T) ** 2)
    g = P * cp / (0.622 * Lambda)
    return esa, s, g


# @staticmethod
def penman_monteith(AE, D, T, Gs, Ga, P=101300.0, units="W"):
    """
    Computes latent heat flux LE (Wm-2) i.e evapotranspiration rate ET (mm/s)
    from Penman-Monteith equation
    INPUT:
       AE - available energy [Wm-2]
       VPD - vapor pressure deficit [Pa]
       T - ambient air temperature [degC]
       Gs - surface conductance [ms-1]
       Ga - aerodynamic conductance [ms-1]
       P - ambient pressure [Pa]
       units - W (Wm-2), mm (mms-1=kg m-2 s-1), mol (mol m-2 s-1)
    OUTPUT:
       x - evaporation rate in 'units'
    """
    # --- constants
    cp = 1004.67  # J kg-1 K-1
    rho = 1.25  # kg m-3
    Mw = 18e-3  # kg mol-1
    _, s, g = e_sat(T, P)  # slope of sat. vapor pressure, psycrom const
    L = 1e3 * (3147.5 - 2.37 * (T + 273.15))

    x = (s * AE + rho * cp * Ga * D) / (s + g * (1.0 + Ga / Gs))  # Wm-2

    if units == "mm":
        x = x / L  # kgm-2s-1 = mms-1
    if units == "mol":
        x = x / L / Mw  # mol m-2 s-1

    x = np.maximum(x, 0.0)
    return x


def aerodynamics(LAI, hc, Uo, w=0.01, zm=2.0, zg=0.5, zos=0.01):
    """
    computes wind speed at ground and canopy + boundary layer conductances
    Computes wind speed at ground height assuming logarithmic profile above and
    exponential within canopy
    Args:
        LAI - one-sided leaf-area /plant area index (m2m-2)
        hc - canopy height (m)
        Uo - mean wind speed at height zm (ms-1)
        w - leaf length scale (m)
        zm - wind speed measurement height above canopy (m)
        zg - height above ground where Ug is computed (m)
        zos - forest floor roughness length, ~ 0.1*roughness element height (m)
    Returns:
        ra - canopy aerodynamic resistance (sm-1)
        rb - canopy boundary layer resistance (sm-1)
        ras - forest floor aerod. resistance (sm-1)
        ustar - friction velocity (ms-1)
        Uh - wind speed at hc (ms-1)
        Ug - wind speed at zg (ms-1)
    SOURCE:
       Cammalleri et al. 2010 Hydrol. Earth Syst. Sci
       Massman 1987, BLM 40, 179 - 197.
       Magnani et al. 1998 Plant Cell Env.
    """
    zm = hc + zm  # m
    kv = 0.4  # von Karman constant (-)
    beta = 285.0  # s/m, from Campbell & Norman eq. (7.33) x 42.0 molm-3
    alpha = LAI / 2.0  # wind attenuation coeff (Yi, 2008 eq. 23)
    d = 0.66 * hc  # m
    zom = 0.123 * hc  # m
    zov = 0.1 * zom
    zosv = 0.1 * zos

    # solve ustar and U(hc) from log-profile above canopy
    ustar = Uo * kv / np.log((zm - d) / zom)
    Uh = ustar / kv * np.log((hc - d) / zom)

    # U(zg) from exponential wind profile
    zn = np.minimum(zg / hc, 1.0)  # zground can't be above canopy top
    Ug = Uh * np.exp(alpha * (zn - 1.0))

    # canopy aerodynamic & boundary-layer resistances (sm-1). Magnani et al. 1998 PCE eq. B1 & B5
    ra = 1.0 / (kv**2.0 * Uo) * np.log((zm - d) / zom) * np.log((zm - d) / zov)
    rb = 1.0 / LAI * beta * ((w / Uh) * (alpha / (1.0 - np.exp(-alpha / 2.0)))) ** 0.5

    # soil aerodynamic resistance (sm-1)
    ras = 1.0 / (kv**2.0 * Ug) * (np.log(zg / zos)) * np.log(zg / (zosv))

    ra = ra + rb
    return ra, rb, ras, ustar, Uh, Ug
