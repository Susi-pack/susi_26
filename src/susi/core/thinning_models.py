# -*- coding: utf-8 -*-
"""
Created on 5th November 2025

@author: Mikko Niemi
"""

import numpy as np

""" THINNING MODELS """
# Source: https://api.metsanhoidonsuositukset.fi/v2/docs/#/Thinning%20Model/get_thinning_models_search

thinningModels = {
    "Southern_Finland": {
        "Organic_soil": {
            2: {
                "spruce": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -3.4345},
                                {"name": "b1", "value": 30.1291},
                                {"name": "b2", "value": 0.9038},
                            ],
                            "min": 12,
                            "max": 23,
                        }
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": 5.51691},
                                {"name": "b1", "value": 29.9113},
                                {"name": "b2", "value": 1.2193},
                            ],
                            "min": 12,
                            "max": 23,
                        }
                    ],
                }
            },
            3: {
                "pine": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -4.6218},
                                {"name": "b1", "value": 23.0843},
                                {"name": "b2", "value": 2.2737},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -11.9217},
                                {"name": "b1", "value": 38.6871},
                                {"name": "b2", "value": 3},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                },
                "spruce": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -5.237},
                                {"name": "b1", "value": 30.1291},
                                {"name": "b2", "value": 0.9038},
                            ],
                            "min": 12,
                            "max": 23,
                        }
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": 2.33381},
                                {"name": "b1", "value": 29.9113},
                                {"name": "b2", "value": 1.2193},
                            ],
                            "min": 12,
                            "max": 23,
                        }
                    ],
                },
            },
            4: {
                "pine": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -5.0174},
                                {"name": "b1", "value": 23.0843},
                                {"name": "b2", "value": 2.2737},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -12.7733},
                                {"name": "b1", "value": 38.6871},
                                {"name": "b2", "value": 3},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                }
            },
            5: {
                "pine": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -7.0997},
                                {"name": "b1", "value": 23.726},
                                {"name": "b2", "value": 2.8026},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -7.6627},
                                {"name": "b1", "value": 32.0767},
                                {"name": "b2", "value": 3},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                }
            },
        }
    },
    "Central_Finland": {
        "Organic_soil": {
            2: {
                "spruce": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": 7.1122},
                                {"name": "b1", "value": 16.336},
                                {"name": "b2", "value": 1.2193},
                            ],
                            "min": 12,
                            "max": 15,
                        },
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -2.6279},
                                {"name": "b1", "value": 30.1291},
                                {"name": "b2", "value": 0.9038},
                            ],
                            "min": 15,
                            "max": 23,
                        },
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": 13.1332},
                                {"name": "b1", "value": 20.4073},
                                {"name": "b2", "value": 1.8144},
                            ],
                            "min": 12,
                            "max": 17,
                        },
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": 6.6795},
                                {"name": "b1", "value": 29.9113},
                                {"name": "b2", "value": 1.2193},
                            ],
                            "min": 17,
                            "max": 23,
                        },
                    ],
                }
            },
            3: {
                "pine": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -4.0101},
                                {"name": "b1", "value": 23.0843},
                                {"name": "b2", "value": 2.2737},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -10.8888},
                                {"name": "b1", "value": 38.6871},
                                {"name": "b2", "value": 3},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                },
                "spruce": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -4.4304},
                                {"name": "b1", "value": 30.1291},
                                {"name": "b2", "value": 0.9038},
                            ],
                            "min": 12,
                            "max": 23,
                        }
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": 3.4964},
                                {"name": "b1", "value": 29.9113},
                                {"name": "b2", "value": 1.2193},
                            ],
                            "min": 12,
                            "max": 23,
                        }
                    ],
                },
            },
            4: {
                "pine": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -4.4057},
                                {"name": "b1", "value": 23.0843},
                                {"name": "b2", "value": 2.2737},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -11.7404},
                                {"name": "b1", "value": 38.6871},
                                {"name": "b2", "value": 3},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                }
            },
            5: {
                "pine": {
                    "thinningLimit": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -6.7226},
                                {"name": "b1", "value": 23.726},
                                {"name": "b2", "value": 2.8026},
                            ],
                            "min": 14,
                            "max": 20,
                        },
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -5.8704},
                                {"name": "b1", "value": 23.0843},
                                {"name": "b2", "value": 2.2737},
                            ],
                            "min": 20,
                            "max": 23,
                        },
                    ],
                    "thinningRecommendation": [
                        {
                            "function": "b0 + b1*(1-EXP(-1*(0.1*x)^b2))",
                            "parameters": [
                                {"name": "b0", "value": -7.0782},
                                {"name": "b1", "value": 32.0767},
                                {"name": "b2", "value": 3},
                            ],
                            "min": 14,
                            "max": 23,
                        }
                    ],
                }
            },
        }
    },
}


def find_matching_parameters(region, soil, fertility_class, main_sp, H_dom):
    results = []

    # Navigate to the correct region and soil
    region_data = thinningModels.get(region, {})
    soil_data = region_data.get(soil, {})
    site_data = soil_data.get(fertility_class, {})
    species_data = site_data.get(main_sp, {})

    # Check both thinningLimit and thinningRecommendation
    for key in ["thinningLimit", "thinningRecommendation"]:
        for entry in species_data.get(key, []):
            if entry["min"] <= H_dom <= entry["max"]:
                results.append(
                    {
                        "type": key,
                        "function": entry["function"],
                        "parameters": {
                            p["name"]: p["value"] for p in entry["parameters"]
                        },
                        "range": (entry["min"], entry["max"]),
                    }
                )

    return results


def extract_parameters(parameters, param_type):
    for entry in parameters:
        if entry["type"] == param_type:
            return {
                "b0": entry["parameters"].get("b0"),
                "b1": entry["parameters"].get("b1"),
                "b2": entry["parameters"].get("b2"),
            }


def calculate_thinning_recommendation(region, soil, fertility_class, main_sp, H_dom):
    parameters = find_matching_parameters(region, soil, fertility_class, main_sp, H_dom)
    if len(parameters) > 0:
        thinningLimit = extract_parameters(parameters, "thinningLimit")
        BA_limit = thinningLimit["b0"] + thinningLimit["b1"] * (
            1 - np.exp(-1 * np.power(0.1 * H_dom, thinningLimit["b2"]))
        )
        thinningRecommendation = extract_parameters(
            parameters, "thinningRecommendation"
        )
        BA_recommendation = thinningRecommendation["b0"] + thinningRecommendation[
            "b1"
        ] * (1 - np.exp(-1 * np.power(0.1 * H_dom, thinningRecommendation["b2"])))
        return round(BA_limit, 2), round(BA_recommendation, 2)
    else:
        return None, None
