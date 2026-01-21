# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:31:27 2022

@author: alauren
"""

import numpy as np


class Fertilization:
    def __init__(self, spara):
        self.ncols = spara.n

    def ph_effect(self, yr, spara):
        tfert = yr - spara.fertilization.application_year
        pH_increment = spara.fertilization.pH_increment * np.exp(-0.1 * tfert)
        return pH_increment

    def nutrient_release(self, yr, spara):
        self.release = {"N": 0.0, "P": 0.0, "K": 0.0}
        tfert = yr - spara.fertilization.application_year
        if tfert >= 0:
            for nutr in ["N", "P", "K"]:
                fertilization_nutrient = getattr(spara.fertilization, nutr)

                nut_efficiency = fertilization_nutrient.eff
                dose = fertilization_nutrient.dose
                decay_k = fertilization_nutrient.decay_k
                self.release[nutr] = (
                    (
                        dose * np.exp(-decay_k * tfert)
                        - dose * np.exp(-decay_k * (tfert + 1))
                    )
                    * nut_efficiency
                    * np.ones(self.ncols)
                )  # nutrient release from the fertilizer

        return self.release
