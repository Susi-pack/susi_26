# -*- coding: utf-8 -*-
"""
This module applies diameter increment and survival rate models of Pukkala et al. (2021)
to produce similar input array to Motti-simulator (see Laurén et al. (2021), chapter 2.3.8).

@authors: Mikko Niemi & Annamari Laurén
"""

import numpy as np
import pandas as pd
from supersusi.core.stem_curve import StemCurve
from supersusi.core.weibull_recovery import generate_weibull_tree_list
from supersusi.core.metsi.naslund import naslund_height, naslund_correction
from supersusi.core.metsi.biomass_repola import (
    stem_wood_biomass_1,
    stem_bark_biomass_1,
    living_branches_biomass_1,
    dead_branches_biomass_1,
    foliage_biomass_1,
    stump_biomass_1,
    roots_biomass_1,
)


class Growth_and_Yield_Table:
    def __init__(
        self,
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
        fertility_class,
        peat,
        y,
        x,
        altitude,
        n_trees,
    ):
        self.n_trees = n_trees  # number of reference trees to generate per tree stratum
        self.age_1 = age_1  # age of pine, years
        self.G_1 = G_1  # basal area of pine, m2/ha
        self.N_1 = N_1  # stem number of pine, /ha
        self.Dg_1 = Dg_1  # mean diameter of pine, cm
        self.Hg_1 = Hg_1  # mean height of pine, m
        self.age_2 = age_2  # age of spruce, years
        self.G_2 = G_2  # basal area of spruce, m2/ha
        self.N_2 = N_2  # stem number of spruce, /ha
        self.Dg_2 = Dg_2  # mean diameter of spruce, cm
        self.Hg_2 = Hg_2  # mean height of spruce, m
        self.age_3 = age_3  # age of deciduous trees, years (assuming Betula pubescens)
        self.G_3 = (
            G_3  # basal area of deciduous trees, m2/ha (assuming Betula pubescens)
        )
        self.N_3 = (
            N_3  # stem number of deciduous trees, /ha (assuming Betula pubescens)
        )
        self.Dg_3 = (
            Dg_3  # mean diameter of deciduous trees, cm (assuming Betula pubescens)
        )
        self.Hg_3 = (
            Hg_3  # mean height of deciduous trees, m (assuming Betula pubescens)
        )
        self.DDY = DDY  # temperature sum, Degree Days per Year
        self.fertility_class = fertility_class  # as integer 1...6
        self.peat = peat  # peat soil = 1
        self.y = y  # y-coordinate
        self.x = x  # x-coordinate
        self.altitude = altitude  # altitude above the sea level

        self.G = sum([self.G_1, self.G_2, self.G_3])
        self.N = sum([self.N_1, self.N_2, self.N_3])
        self.age = round(
            sum(
                a * g
                for a, g in zip(
                    [self.age_1, self.age_2, self.age_3], [self.G_1, self.G_2, self.G_3]
                )
            )
            / self.G
        )
        self.Dg = round(
            sum(
                d * g
                for d, g in zip(
                    [self.Dg_1, self.Dg_2, self.Dg_3], [self.G_1, self.G_2, self.G_3]
                )
            )
            / self.G,
            1,
        )
        self.Hg = round(
            sum(
                h * g
                for h, g in zip(
                    [self.Hg_1, self.Hg_2, self.Hg_3], [self.G_1, self.G_2, self.G_3]
                )
            )
            / self.G,
            1,
        )

    def calculate_basal_area_weighted_attributes(self, ReferenceTrees):
        Hg = sum(
            h * g
            for h, g in zip(
                ReferenceTrees["H"],
                (
                    ReferenceTrees["Nd"]
                    * np.pi
                    * np.power((ReferenceTrees["D"] / 2) / 100, 2)
                ),
            )
        ) / sum(
            (
                ReferenceTrees["Nd"]
                * np.pi
                * np.power((ReferenceTrees["D"] / 2) / 100, 2)
            )
        )
        Dg = sum(
            d * g
            for d, g in zip(
                ReferenceTrees["D"],
                (
                    ReferenceTrees["Nd"]
                    * np.pi
                    * np.power((ReferenceTrees["D"] / 2) / 100, 2)
                ),
            )
        ) / sum(
            (
                ReferenceTrees["Nd"]
                * np.pi
                * np.power((ReferenceTrees["D"] / 2) / 100, 2)
            )
        )
        return Hg, Dg

    def dominant_height(self, ReferenceTrees):
        """
        output:
            dominant height, m, mean height of 100 thickest trees
        """

        if sum(ReferenceTrees["Nd"]) > 100:
            df_sorted = ReferenceTrees.sort_values(by="D", ascending=False).reset_index(
                drop=True
            )
            df_sorted["cum_N"] = df_sorted["Nd"].cumsum()
            df_top = df_sorted[df_sorted["cum_N"] <= 100]

            # Total N so far
            total_N = df_top["Nd"].sum()

            # If we haven't reached 100 yet, add a fractional row
            if total_N < 100:
                next_row = df_sorted.iloc[len(df_top)].copy()
                remaining = 100 - total_N
                next_row["Nd"] = remaining
                next_row["cum_N"] = 100
                df_top = pd.concat(
                    [df_top, pd.DataFrame([next_row])], ignore_index=True
                )

            # Calculate dominant height
            dominant_height = (df_top["H"] * df_top["Nd"]).sum() / 100

        else:
            dominant_height = np.mean(ReferenceTrees["H"])

        return dominant_height

    def get_biomass_components(self, ReferenceTrees):
        stem_wood, living_branches, dead_branches, foliage, stump, coarse_roots = (
            0,
            0,
            0,
            0,
            0,
            0,
        )

        for index, row in ReferenceTrees.iterrows():
            stem_wood += row["Nd"] * (
                stem_wood_biomass_1(row["sp"], row["D"], row["H"])
                + stem_bark_biomass_1(row["sp"], row["D"], row["H"])
            )
            living_branches += row["Nd"] * living_branches_biomass_1(
                row["sp"], row["D"], row["H"]
            )
            dead_branches += row["Nd"] * dead_branches_biomass_1(
                row["sp"], row["D"], row["H"]
            )
            foliage += row["Nd"] * foliage_biomass_1(row["sp"], row["D"], row["H"])
            stump += row["Nd"] * stump_biomass_1(row["sp"], row["D"], row["H"])
            coarse_roots += row["Nd"] * roots_biomass_1(row["sp"], row["D"], row["H"])

        components = {
            "stem_wood": round(stem_wood, 1),
            "stem_loss": 0,
            "living_branches": round(living_branches, 1),
            "dead_branches": round(dead_branches, 1),
            "foliage": round(foliage, 1),
            "stump": round(stump, 1),
            "coarse_roots": round(coarse_roots, 1),
            "fine_roots": round(0.02 * coarse_roots, 1),
        }
        return components

    def get_assortment_volumes(
        self,
        ReferenceTrees,
        age,
        y,
        x,
        altitude,
        DDY,
        fertility_class,
        peatland=1,
        planted=0,
    ):
        log, pulp, total = 0, 0, 0

        # Reclassify all deciduous trees (sp > 3) as 3
        ReferenceTrees.loc[ReferenceTrees["sp"] > 3, "sp"] = 3

        for index, row in ReferenceTrees.iterrows():
            if row["H"] > 1.5:
                temp_assortments = StemCurve().predictAssortmentVolumes(
                    row["D"],
                    row["H"],
                    int(row["sp"]),
                    age,
                    y,
                    x,
                    altitude,
                    DDY,
                    fertility_class,
                    peatland,
                    planted,
                )  # unit: litres
                log += row["Nd"] * (temp_assortments["log"] / 1000)
                pulp += row["Nd"] * (temp_assortments["pulp"] / 1000)
                total += row["Nd"] * (temp_assortments["total"] / 1000)

        assortments = {
            "log": round(log, 1),
            "pulp": round(pulp, 1),
            "residue": round(total - log - pulp, 1),
            "total": round(total, 1),
        }
        return assortments

    # Survival model (Pukkala et al. 2021)
    def get_survival(
        self, sp, D, BAL_Total, BAL_Pine, BAL_Spruce, BAL_S_B, Peat, Aspen, Birch
    ):
        # Parameters (Pukkala et al. 2021):
        S_param = {
            "Intercept": [1.41223, 5.01677, 1.60895],
            "sqrt_d": [1.8852, 0.36902, 0.71578],
            "d": [-0.21317, -0.07504, -0.08236],
            "BAL_Total": [-0.25637, 0, 0],
            "BAL_Pine": [0, 0, -0.04814],
            "BAL_Spruce": [0, -0.2319, 0],
            "BAL_Spruce_Broadleaf": [0, 0, -0.13481],
            "Peat": [-0.39878, -0.47361, -0.31789],
            "Aspen": [0, 0, 0.56311],
            "Birch": [0, 0, 1.40145],
        }

        fx = np.zeros(len(D))
        for i in range(len(D)):
            # Reclassify species code to represent species-index 'spi': 0=pine, 1=spruce, 2=other
            if sp[i] <= 2:
                spi = sp[i] - 1
            else:
                spi = 2
            fx[i] = (
                S_param["Intercept"][spi]
                + S_param["sqrt_d"][spi] * np.sqrt(D[i])
                + S_param["d"][spi] * D[i]
                + S_param["BAL_Total"][spi] * (BAL_Total[i] / np.sqrt(D[i] + 1))
                + S_param["BAL_Pine"][spi] * (BAL_Pine[i] / np.sqrt(D[i] + 1))
                + S_param["BAL_Spruce"][spi] * (BAL_Spruce[i] / np.sqrt(D[i] + 1))
                + S_param["BAL_Spruce_Broadleaf"][spi]
                * (BAL_S_B[i] / np.sqrt(D[i] + 1))
                + S_param["Peat"][spi] * Peat[i]
                + S_param["Aspen"][spi] * Aspen[i]
                + S_param["Birch"][spi] * Birch[i]
            )
        survival = 1 / (1 + np.exp(-fx))
        return survival

    # Predict survival rate:
    def predict_survival_5_years(self, ReferenceTrees, peat=1):
        """
        Pukkala et al. 2021. https://doi.org/10.1093/forestry/cpab008
        input:
            Nd - number of trees in reference tree class, array
            sp - tree species as integer value
            D  - tree diameter at breast height
        """
        Peat = np.repeat(peat, len(ReferenceTrees))
        Aspen = np.zeros(len(ReferenceTrees))
        Birch = np.zeros(len(ReferenceTrees))
        for i in range(len(ReferenceTrees)):
            if ReferenceTrees["sp"][i] == 5:
                Aspen[i] = 1
            if (ReferenceTrees["sp"][i] == 3) | (ReferenceTrees["sp"][i] == 4):
                Birch[i] = 1

        G_class = ReferenceTrees["Nd"] * np.pi * (ReferenceTrees["D"] / 2) ** 2 / 10000

        BAL_Total = np.zeros(len(ReferenceTrees))
        BAL_Pine = np.zeros(len(ReferenceTrees))
        BAL_Spruce = np.zeros(len(ReferenceTrees))
        BAL_S_B = np.zeros(len(ReferenceTrees))
        for i in range(len(ReferenceTrees)):
            BAL_Total[i] = np.sum(G_class[ReferenceTrees["D"] > ReferenceTrees["D"][i]])
            BAL_Pine[i] = np.sum(
                G_class[
                    (ReferenceTrees["D"] > ReferenceTrees["D"][i])
                    & (ReferenceTrees["sp"] == 1)
                ]
            )
            BAL_Spruce[i] = np.sum(
                G_class[
                    (ReferenceTrees["D"] > ReferenceTrees["D"][i])
                    & (ReferenceTrees["sp"] == 2)
                ]
            )
            BAL_S_B[i] = np.sum(
                G_class[
                    (ReferenceTrees["D"] > ReferenceTrees["D"][i])
                    & (ReferenceTrees["sp"] != 1)
                ]
            )

        return self.get_survival(
            ReferenceTrees["sp"],
            ReferenceTrees["D"],
            BAL_Total,
            BAL_Pine,
            BAL_Spruce,
            BAL_S_B,
            Peat,
            Aspen,
            Birch,
        )

    # Diameter increment model (Pukkala et al. 2021):
    def get_diameter_increment(
        self,
        initialDiameter,
        sp,
        G_plot,
        BAL_Total,
        BAL_Spruce,
        BAL_S_B,
        TS,
        SiteType,
        Peat,
        Pendula_or_Aspen,
    ):
        # Parameters (Pukkala et al. 2021):
        D_param = {
            "Intercept": [-7.1552, -12.7527, -8.6306],
            "sqrt_d": [0.4415, 0.1693, 0.5097],
            "d": [-0.0685, -0.0301, -0.0829],
            "ln_G_1": [-0.2027, -0.1875, -0.3864],
            "BAL_Total": [-0.1236, -0.0563, 0],
            "BAL_Spruce": [0, -0.0870, 0],
            "BAL_Spruce_Broadleaf": [0, 0, -0.0545],
            "ln_TS": [1.1198, 1.9747, 1.3163],
            "Peat": [-0.2425, 0, 0],
            "d_Pendula_or_Aspen": [0, 0, 0.0253],
            "Fertility": {
                "Herb-rich": [0.1438, 0.2688, 0.2566],
                "Mesic": [0, 0, 0],
                "Sub-xeric": [-0.1754, -0.2145, -0.2256],
                "Xeric": [-0.5163, -0.6179, -0.3237],
            },
        }

        ln_D_increment = np.zeros(len(initialDiameter))
        for i in range(len(initialDiameter)):
            # Reclassify species code to represent species-index 'spi': 0=pine, 1=spruce, 2=other
            if sp[i] <= 2:
                spi = sp[i] - 1
            else:
                spi = 2
            ln_D_increment[i] = (
                D_param["Intercept"][spi]
                + D_param["sqrt_d"][spi] * np.sqrt(initialDiameter[i])
                + D_param["d"][spi] * initialDiameter[i]
                + D_param["ln_G_1"][spi] * np.log(G_plot[i] + 1)
                + D_param["BAL_Total"][spi]
                * (BAL_Total[i] / np.sqrt(initialDiameter[i] + 1))
                + D_param["BAL_Spruce"][spi]
                * (BAL_Spruce[i] / np.sqrt(initialDiameter[i] + 1))
                + D_param["BAL_Spruce_Broadleaf"][spi]
                * (BAL_S_B[i] / np.sqrt(initialDiameter[i] + 1))
                + D_param["ln_TS"][spi] * np.log(TS[i])
                + D_param["Fertility"][SiteType[i]][spi]
                + D_param["Peat"][spi] * Peat[i]
                + D_param["d_Pendula_or_Aspen"][spi]
                * Pendula_or_Aspen[i]
                * initialDiameter[i]
            )
        return np.exp(ln_D_increment)

    # Predict diameter increment for next 5 years:
    def predict_diameter_increment_5_years(
        self, ReferenceTrees, fertilityClass, temperatureSum=1200, peat=1
    ):
        """
        Timo Pukkala and others, Self-learning growth simulator for modelling forest stand dynamics in
        changing conditions, Forestry: An International Journal of Forest Research, Volume 94, Issue 3,
        July 2021, Pages 333–346, https://doi.org/10.1093/forestry/cpab008
        input:
            Nd - number of trees in reference tree class, array
            sp - tree species as integer value
            D - tree diameter at breast height
            fertilityClass - fertility class expressed as string (1-6)
            temperatureSum - temperature sum as degree-days
            peat - 1: peatland, 0: mineral soil
        """

        siteTypeDict = {
            1: "Herb-rich",
            2: "Herb-rich",
            3: "Mesic",
            4: "Sub-xeric",
            5: "Xeric",
            6: "Xeric",
        }
        siteType = np.repeat(siteTypeDict[fertilityClass], len(ReferenceTrees))
        TS = np.repeat(temperatureSum, len(ReferenceTrees))
        Peat = np.repeat(peat, len(ReferenceTrees))

        Pendula_or_Aspen = np.zeros(len(ReferenceTrees))
        for i in range(len(ReferenceTrees)):
            if (ReferenceTrees["sp"][i] == 3) | (ReferenceTrees["sp"][i] == 5):
                Pendula_or_Aspen[i] = 1

        G_class = ReferenceTrees["Nd"] * np.pi * (ReferenceTrees["D"] / 2) ** 2 / 10000
        G_plot = np.repeat(np.sum(G_class), len(ReferenceTrees))

        BAL_Total = np.zeros(len(ReferenceTrees))
        BAL_Spruce = np.zeros(len(ReferenceTrees))
        BAL_S_B = np.zeros(len(ReferenceTrees))
        for i in range(len(ReferenceTrees)):
            BAL_Total[i] = np.sum(G_class[ReferenceTrees["D"] > ReferenceTrees["D"][i]])
            BAL_Spruce[i] = np.sum(
                G_class[
                    (ReferenceTrees["D"] > ReferenceTrees["D"][i])
                    & (ReferenceTrees["sp"] == 2)
                ]
            )
            BAL_S_B[i] = np.sum(
                G_class[
                    (ReferenceTrees["D"] > ReferenceTrees["D"][i])
                    & (ReferenceTrees["sp"] != 1)
                ]
            )

        return self.get_diameter_increment(
            ReferenceTrees["D"],
            ReferenceTrees["sp"],
            G_plot,
            BAL_Total,
            BAL_Spruce,
            BAL_S_B,
            TS,
            siteType,
            Peat,
            Pendula_or_Aspen,
        )

    def stand_development(
        self,
        ReferenceTrees,
        h_scalar,
        time_step,
        age,
        y,
        x,
        altitude,
        DDY,
        fertility_class,
        peatland=1,
    ):
        # predict 5-year tree survival
        survival_rate = self.predict_survival_5_years(ReferenceTrees, peat=1)

        # predict 5-year diameter increment
        ReferenceTrees["D"] = ReferenceTrees[
            "D"
        ] + self.predict_diameter_increment_5_years(
            ReferenceTrees, self.fertility_class, self.DDY, self.peat
        )

        # predict height of reference trees after 5 years
        height = []
        for index, row in ReferenceTrees.iterrows():
            height_estimate = naslund_height(row["D"], row["sp"])
            # Reclassify species code to represent species-index 'spi': 0=pine, 1=spruce, 2=other
            if row["sp"] <= 2:
                spi = int(row["sp"] - 1)
            else:
                spi = 2
            height_scaled = h_scalar[spi] * height_estimate
            height.append(height_scaled)
        ReferenceTrees["H"] = height

        # update stem number per reference tree class
        ReferenceTrees["Nd"] = survival_rate * ReferenceTrees["Nd"]

        # calculate mean attributes, timber assortments and biomass components
        mean_height, mean_diameter = self.calculate_basal_area_weighted_attributes(
            ReferenceTrees
        )
        assortments = self.get_assortment_volumes(
            ReferenceTrees, age, y, x, altitude, DDY, fertility_class, peatland
        )
        biomass = self.get_biomass_components(ReferenceTrees)

        next_state = {
            "Schedule": [0],
            "Year": [time_step],
            "Age": [time_step + age],
            "N": [round(sum(ReferenceTrees["Nd"]))],
            "BA": [
                round(
                    sum(
                        ReferenceTrees["Nd"]
                        * np.pi
                        * np.power(ReferenceTrees["D"] / 2 / 100, 2)
                    ),
                    1,
                )
            ],
            "Hg": [round(mean_height, 1)],
            "Dg": [round(mean_diameter, 1)],
            "Hdom": [round(self.dominant_height(ReferenceTrees), 1)],
            "Volume": [assortments["total"]],
            "Logs": [assortments["log"]],
            "Pulp": [assortments["pulp"]],
            "Loss": [assortments["residue"]],
            "Yield": [0],
            "Mortality": [0],
            "Stem_wood": [biomass["stem_wood"]],
            "Stem_loss": [biomass["stem_loss"]],
            "Living_branches": [biomass["living_branches"]],
            "Dead_branches": [biomass["dead_branches"]],
            "Foliage": [biomass["foliage"]],
            "Stump": [biomass["stump"]],
            "Coarse_roots": [biomass["coarse_roots"]],
            "Fine_roots": [biomass["fine_roots"]],
        }

        return ReferenceTrees, next_state

    def get_table(self, start_year: int, end_year: int, step_years=5):
        """ """
        ReferenceTrees = []

        # Pine
        if self.N_1 != 0:
            ref_1 = pd.DataFrame(
                np.array(
                    generate_weibull_tree_list(
                        self.n_trees, self.G_1, self.Dg_1, self.N_1
                    )
                ),
                columns=["Nd", "D"],
            )
            ref_1.insert(0, "sp", 1)
            ReferenceTrees.append(ref_1)

        # Spruce
        if self.N_2 != 0:
            ref_2 = pd.DataFrame(
                np.array(
                    generate_weibull_tree_list(
                        self.n_trees, self.G_2, self.Dg_2, self.N_2
                    )
                ),
                columns=["Nd", "D"],
            )
            ref_2.insert(0, "sp", 2)
            ReferenceTrees.append(ref_2)

        # Deciduous trees (assuming Betula pubescens)
        if self.N_3 != 0:
            ref_3 = pd.DataFrame(
                np.array(
                    generate_weibull_tree_list(
                        self.n_trees, self.G_3, self.Dg_3, self.N_3
                    )
                ),
                columns=["Nd", "D"],
            )
            ref_3.insert(0, "sp", 4)
            ReferenceTrees.append(ref_3)

        ReferenceTrees = pd.concat(ReferenceTrees, ignore_index=True)

        # Height correction
        h_scalar = [
            0 if self.N_1 == 0 else naslund_correction(1, self.Dg_1, self.Hg_1),
            0 if self.N_2 == 0 else naslund_correction(2, self.Dg_2, self.Hg_2),
            0 if self.N_3 == 0 else naslund_correction(3, self.Dg_3, self.Hg_3),
        ]

        # Height estimates
        height = []
        for index, row in ReferenceTrees.iterrows():
            height_estimate = naslund_height(row["D"], row["sp"])
            # Reclassify species code to represent species-index 'spi': 0=pine, 1=spruce, 2=other
            if row["sp"] <= 2:
                spi = int(row["sp"] - 1)
            else:
                spi = 2
            height_scaled = h_scalar[spi] * height_estimate
            if height_scaled < 1.3:
                height_scaled = 1.3
            height.append(height_scaled)
        ReferenceTrees["H"] = height

        # Mean height and diameter based on the reference trees
        mean_height, mean_diameter = self.calculate_basal_area_weighted_attributes(
            ReferenceTrees
        )

        # Assortment volumes
        assortments = self.get_assortment_volumes(
            ReferenceTrees,
            self.age,
            self.y,
            self.x,
            self.altitude,
            self.DDY,
            self.fertility_class,
            peatland=self.peat,
        )

        # Biomass components
        biomass = self.get_biomass_components(ReferenceTrees)

        # Current state
        current_state = {
            "Schedule": [0],
            "Year": [0],
            "Age": [self.age],
            "N": [round(sum(ReferenceTrees["Nd"]))],
            "BA": [
                round(
                    sum(
                        ReferenceTrees["Nd"]
                        * np.pi
                        * (ReferenceTrees["D"] / 2) ** 2
                        / 10000
                    ),
                    1,
                )
            ],
            "Hg": [round(mean_height, 1)],
            "Dg": [round(mean_diameter, 1)],
            "Hdom": [round(self.dominant_height(ReferenceTrees), 1)],
            "Volume": [assortments["total"]],
            "Logs": [assortments["log"]],
            "Pulp": [assortments["pulp"]],
            "Loss": [assortments["residue"]],
            "Yield": [0],
            "Mortality": [0],
            "Stem_wood": [biomass["stem_wood"]],
            "Stem_loss": [biomass["stem_loss"]],
            "Living_branches": [biomass["living_branches"]],
            "Dead_branches": [biomass["dead_branches"]],
            "Foliage": [biomass["foliage"]],
            "Stump": [biomass["stump"]],
            "Coarse_roots": [biomass["coarse_roots"]],
            "Fine_roots": [biomass["fine_roots"]],
        }

        # Stand development
        susi_input = pd.DataFrame(current_state)
        for time_step in range(start_year, end_year + 1, step_years):
            ReferenceTrees, next_state = self.stand_development(
                ReferenceTrees,
                h_scalar,
                time_step,
                self.age,
                self.y,
                self.x,
                self.altitude,
                self.DDY,
                self.fertility_class,
                peatland=self.peat,
            )
            susi_input = pd.concat(
                [susi_input, pd.DataFrame(next_state)], ignore_index=True
            )

        return susi_input


"""
import os
os.chdir('C:/Users/mikkoni/AEMES/Coding/susi_2024-master-20250806')

import numpy as np
import pandas as pd

from susi.allometric_road_map import Growth_and_Yield_Table

age_1 = 54                  # age (years), pine
G_1 = 23.4                  # basal area (m2 ha-1), pine
N_1 = 1276                  # stem number (ha-1), pine
Dg_1 = 17.6                 # mean diameter (cm), pine
Hg_1 = 15.2                 # mean height (m), pine
age_2 = 44                  # age (years), spruce
G_2 = 1.1                   # basal area (m2 ha-1), spruce
N_2 = 60                    # stem number (ha-1), spruce
Dg_2 = 16                   # mean diameter (cm), spruce
Hg_2 = 14.3                 # mean height (m), spruce
age_3 = 28                  # age (years), deciduous trees
G_3 = 0.3                   # basal area (m2 ha-1), deciduous trees
N_3 = 49                    # stem number (ha-1), deciduous trees
Dg_3 = 9.2                  # mean diameter (cm), deciduous trees
Hg_3 = 11.7                 # mean height (m), deciduous trees

n_trees = 20                # number of reference trees per tree species

DDY = 1100                  # temperature sum, degree days
fertility_class = 4         # soil fertility class (integer 1-6)
peat = 1                    # peatland = 1, mineral soil = 0

# Lestijärvi
y = 7070 # (YKJ)            # y-coordinate (see https://metsatieteenaikakauskirja.fi/article/6196)
x = 326  # (YKJ)            # x-coordinate (see https://metsatieteenaikakauskirja.fi/article/6196)
altitude = 140              # altitude above sea level

gy = Growth_and_Yield_Table(
    age_1, G_1, N_1, Dg_1, Hg_1,
    age_2, G_2, N_2, Dg_2, Hg_2,
    age_3, G_3, N_3, Dg_3, Hg_3,
    DDY, fertility_class, peat, y, x, altitude, n_trees
)

susi_input = gy.get_table()
susi_input.to_clipboard()


# Write to Excel with two sheets

page2 = pd.DataFrame({
    'StandID': [1],
    'Schedule': [1],
    'Year': [0],
    'HarvestType': ['no_loggings'],
    'Species_id': [1]
    })

with pd.ExcelWriter('C:/Users/mikkoni/AEMES/Coding/susi_inputs/susi_input.xlsx', engine='xlsxwriter') as writer:
    susi_input.to_excel(writer, sheet_name='StandData', index=False)
    page2.to_excel(writer, sheet_name='Loggings', index=False)
"""
