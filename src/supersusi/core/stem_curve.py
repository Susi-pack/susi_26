# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:15:55 2023

@author: alauren
"""

import numpy as np
from scipy.integrate import quad

# Copied from Vauhkonen (2019):
""" Class StemCurve is structured to include stem taper models and stem bucking based on analyzing the stem tapering. """
""" The attributes of the class include both allowable log dimensions for stem bucking and required parameters for Laaasasenaho's models. """
""" The methods of the class are both for 1) determining the stem curve and extracting values from it; 2) virtually bucking the stem """
""" based on the tapering and allowable log dimensions, with an aim to produce as much saw wood logs as possible. """  # (Vauhkonen 2019)

# Niemi, M.T. modified (2025):
""" Allowable saw log and pulpwood lengths removed. """
""" Calculate theoretical maximum of log and pulp assortments using functions of Vauhkonen (2019). """
""" Part of theoretical saw log volume converted to pulpwood volume by saw log reduction functions """
""" of Mehtätalo (2002), and species-spesific correction factors of Malinen et al. (2007). """


class StemCurve:
    def __init__(self):
        self.min_diams = {"log": (0, 15, 16, 18), "pulp": (0, 7, 7, 7)}
        self.min_lens = {"log": (0, 3.7, 3.7, 3.1), "pulp": (0, 3, 3, 3)}
        self.stemcurve_coefs = {
            1: (2.1288, -0.63157, -1.6082, 2.4886, -2.4147, 2.3619, -1.7539, 1.0817),
            2: (2.3366, -3.2684, 3.6513, -2.2608, 0, 2.1501, -2.7412, 1.8876),
            3: (0.93838, 4.106, -7.8517, 7.8993, -7.5018, 6.3863, -4.3918, 2.1604),
        }
        self.stumpheight_coefs = {
            1: (0.09522, 0.4456),
            2: (0.56, 0.5089),
            3: (0.497936, 0.4862),
        }

    def stemCurve(self, hx, h, species):
        x = 1 - hx / h
        b = self.stemcurve_coefs[species]
        return (
            b[0] * x
            + b[1] * x**2
            + b[2] * x**3
            + b[3] * x**5
            + b[4] * x**8
            + b[5] * x**13
            + b[6] * x**21
            + b[7] * x**34
        )

    def integrand(self, hx, h, d20, sp):
        return np.pi / 4 * (self.stemCurve(hx, h, sp) * d20 / 100) ** 2 * 1000

    def calculateVolume(self, d13, h, sp, lower, upper):
        d20 = d13 / self.stemCurve(1.3, h, sp)
        return quad(self.integrand, lower, upper, args=(h, d20, sp))[0]

    def detectCutHeight(self, dcut, d13, h, sp):
        d20 = d13 / self.stemCurve(1.3, h, sp)
        htest = 0
        diff = h / float(2)
        hsol = -1
        while hsol != htest:
            if htest > h:
                htest = h
            dtest = self.stemCurve(htest, h, sp) * d20
            if dtest < dcut + 0.001 and dtest > dcut - 0.001:
                hsol = htest
                break
            if dtest > dcut:
                htest = htest + diff
            else:
                htest = htest - diff
            diff = diff * 0.5
        return hsol

    def predictStumpHeight(self, d13, h, sp):
        stumph = self.stumpheight_coefs[sp][0] * h + self.stumpheight_coefs[sp][1] * d13
        stumph = stumph / 100
        if stumph < 0.1:
            return 0.10
        else:
            return stumph

    def buckStem(self, d13, h, sp):
        StumpHeight = self.predictStumpHeight(d13, h, sp)
        cutPoints = {"stump": StumpHeight, "pulp": StumpHeight, "log": StumpHeight}
        if d13 > self.min_diams["log"][sp]:
            cutLog = self.detectCutHeight(self.min_diams["log"][sp], d13, h, sp)
            if cutLog - cutPoints["stump"] >= self.min_lens["log"][sp]:
                cutPoints["log"] = cutLog
        if d13 > self.min_diams["pulp"][sp]:
            cutPulp = self.detectCutHeight(self.min_diams["pulp"][sp], d13, h, sp)
            if cutPulp - cutPoints["stump"] >= self.min_lens["pulp"][sp]:
                cutPoints["pulp"] = cutPulp
        return cutPoints

    def sawlogReduction(
        self, sp, t, d, y, x, altitude, DDY, fertility_class, peatland=0, planted=0
    ):
        rich = 1 if fertility_class == 2 else 0
        poor = 1 if fertility_class == 5 else 0

        para = {
            "1": {
                "c": -7.301,
                "t": 0.03814,
                "ln_t": -3.568,
                "t_p2": 0,
                "d": -0.2184,
                "ln_d": 0,
                "inv_d": 0,
                "d_p2": 0.003090,
                "d_per_t": 0,
                "planted": 0,
                "y": 0.004176,
                "ln_y": -0.7974,
                "ln_x": 0,
                "altitude": 0,
                "ln_alt": -0.4549,
                "alt60": 0,
                "alt120": 0,
                "peatland": 0.2832,
                "rich": 0.4607,
                "poor": 0.6323,
                "dd_rich": 0,
                "alt_rich": 0,
            },
            "2": {
                "c": 20.61,
                "t": -0.004853,
                "ln_t": 0,
                "t_p2": 0.0000556,
                "d": 0.1893,
                "ln_d": -8.186,
                "inv_d": 0,
                "d_p2": 0,
                "d_per_t": 1.493,
                "planted": 0.2884,
                "y": 0,
                "ln_y": 0,
                "ln_x": 0,
                "altitude": 0,
                "ln_alt": 0,
                "alt60": -0.01600,
                "alt120": 0,
                "peatland": 0.1538,
                "rich": 3.638,
                "poor": 0,
                "dd_rich": -0.002493,
                "alt_rich": -0.003826,
            },
            "3": {
                "c": -13.21,
                "t": -0.02215,
                "ln_t": 0,
                "t_p2": 0.0001727,
                "d": 0.1046,
                "ln_d": 0,
                "inv_d": 73.49,
                "d_p2": 0,
                "d_per_t": 0,
                "planted": 0,
                "y": 0.001167,
                "ln_y": 0,
                "ln_x": 0,
                "altitude": 0.006862,
                "ln_alt": 0,
                "alt60": 0,
                "alt120": -0.01092,
                "peatland": 0,
                "rich": 0,
                "poor": 2.239,
                "dd_rich": 0,
                "alt_rich": 0,
            },
            "4": {
                "c": -34.42,
                "t": 0.05320,
                "ln_t": -3.675,
                "t_p2": 0,
                "d": 0.1460,
                "ln_d": 0,
                "inv_d": 81.25,
                "d_p2": 0,
                "d_per_t": 0,
                "planted": 0,
                "y": 0.006610,
                "ln_y": -1.106,
                "ln_x": 0,
                "altitude": 0,
                "ln_alt": 0,
                "alt60": 0,
                "alt120": 0,
                "peatland": 0,
                "rich": 0,
                "poor": 0,
                "dd_rich": 0,
                "alt_rich": 0,
            },
            "5": {
                "c": -6.205,
                "t": 0.01137,
                "ln_t": 0,
                "t_p2": 0,
                "d": 0.02394,
                "ln_d": 0,
                "inv_d": 0,
                "d_p2": 0,
                "d_per_t": 0,
                "planted": 0,
                "y": 0,
                "ln_y": 0.7191,
                "ln_x": 0.7884,
                "altitude": 0.01686,
                "ln_alt": -1.186,
                "alt60": 0,
                "alt120": 0,
                "peatland": -0.7854,
                "rich": 0,
                "poor": 0,
                "dd_rich": 0,
                "alt_rich": 0,
            },
        }

        sp = str(sp)
        lx = (
            para[sp]["c"]
            + para[sp]["t"] * t
            + para[sp]["ln_t"] * np.log(t)
            + para[sp]["t_p2"] * t**2
            + para[sp]["d"] * d
            + para[sp]["ln_d"] * np.log(d)
            + para[sp]["inv_d"] * (1 / d)
            + para[sp]["d_p2"] * d**2
            + para[sp]["d_per_t"] * (d / t)
            + para[sp]["planted"] * planted
            + para[sp]["y"] * y
            + para[sp]["ln_y"] * np.log(y - 6600)
            + para[sp]["ln_x"] * np.log(x)
            + para[sp]["altitude"] * altitude
            + para[sp]["ln_alt"] * np.log(altitude + 1)
            + para[sp]["alt60"] * min(altitude, 60)
            + para[sp]["alt120"] * min(altitude, 120)
            + para[sp]["peatland"] * peatland
            + para[sp]["rich"] * rich
            + para[sp]["poor"] * poor
            + para[sp]["dd_rich"] * DDY * rich
            + para[sp]["alt_rich"] * altitude * rich
        )

        sawlogRed = np.exp(lx) / (1 + np.exp(lx))

        # Correction factors of Malinen et al. (2007):
        if sp == "1":
            sawlogRed = 0.7 * sawlogRed
        if sp == "2":
            sawlogRed = 0.4 * sawlogRed
        if sp == "3":
            sawlogRed = 1.2 * sawlogRed

        return sawlogRed

    def predictAssortmentVolumes(
        self, d13, h, sp, t, y, x, altitude, DDY, fertility_class, peatland=0, planted=0
    ):
        cutPoints = self.buckStem(d13, h, sp)
        volume = {"log": 0, "pulp": 0, "total": 0}
        volume["total"] = self.calculateVolume(d13, h, sp, 0, h)
        volume["log"] = self.calculateVolume(
            d13, h, sp, cutPoints["stump"], cutPoints["log"]
        )
        volume["pulp"] = self.calculateVolume(
            d13, h, sp, cutPoints["log"], cutPoints["pulp"]
        )
        sawlog_to_pulp = (
            self.sawlogReduction(
                sp, t, d13, y, x, altitude, DDY, fertility_class, peatland, planted
            )
            * volume["log"]
        )
        volume["log"] = volume["log"] - sawlog_to_pulp
        volume["pulp"] = volume["pulp"] + sawlog_to_pulp
        return volume
