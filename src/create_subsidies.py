import numpy as np
import pandas as pd


def get_distribution(nuclear_percentage, renewable_percentage, total, res_distribution):
    distribution = []
    distribution.append(nuclear_percentage * total)
    for i in range(len(res_distribution)):
        distribution.append(res_distribution[i] * total * renewable_percentage)

    return distribution + [0, 0]


# Nuclear, Solar, Wind, Biomass, Hydro, rest is 0
total = 125.0
res_distribution = np.array([0.50, 0.15, 0.30, 0.05])
ffs_distribution = np.array([2.23, 7.87])

nuclear_percentages = np.linspace(0, 1, 41)
renewable_percentages = np.linspace(0, 1, 41)

columns = [
    "Nuclear",
    "Solar",
    "Wind",
    "Biomass",
    "Hydro",
    "Gas",
    "Coal",
    "RES%",
    "NUC%",
]
data = [
    get_distribution(nuclear_percentage, renewable_percentage, total, res_distribution)
    + [renewable_percentage, nuclear_percentage]
    for nuclear_percentage in nuclear_percentages
    for renewable_percentage in renewable_percentages
]
df = pd.DataFrame(data, columns=columns)

df.to_csv("subsidies.csv", index=False)
