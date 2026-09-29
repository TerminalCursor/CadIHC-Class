#!/usr/bin/env python
# coding: utf-8

from glob import glob
from numpy import load as np_load
from json import load as json_load
from pandas import read_csv as pd_load
from numpy import vstack as stack
from numpy import std


def _std(x, axis=0):
    return std(x.tolist(), axis=axis)


data_sources = sorted(glob("../Datasets/HT-*/"))
scaled_data_columns = [
    "Area",
    "MjrAxisLength",
    "MnrAxisLength",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Solidity",
    "Extent",
    "Perimeter",
    "ConvexPerim",
    "FibLen",
    "InscribeR",
    "BlebLen",
]
nuclei_columns = [
    "area",
    "area_convex",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "extent",
    "solidity",
    "equivalent_diameter_area",
]

metric_datum = []
nuclei_datum = []
energy_datum = []
for data_source in data_sources:
    meta_path = data_source + "metadata.json"
    metric_path = data_source + "Metrics.csv"
    nuclei_path = data_source + "CNJoinedMetrics.csv"
    energy_path = data_source + "Energetics.npy"

    # meta_data = {}
    # with open(meta_path, "w") as fhandle:
    #    meta_data = json_load(fhandle)

    metric_data = pd_load(metric_path)[scaled_data_columns]
    nuclei_data = pd_load(nuclei_path)[nuclei_columns]
    energy_data = np_load(energy_path)
    metric_datum.append(metric_data)
    nuclei_datum.append(nuclei_data)
    energy_datum.append(energy_data)
metric_datum = stack(metric_datum)
nuclei_datum = stack(nuclei_datum)
energy_datum = stack(energy_datum)

metric_mu = metric_datum.mean(axis=0)
metric_si = _std(metric_datum, axis=0)
nuclei_mu = nuclei_datum.mean(axis=0)
nuclei_si = _std(nuclei_datum, axis=0)
energy_mu = energy_datum.mean(axis=0)
energy_si = _std(energy_datum, axis=0)

normalized_metric_datum = (metric_datum - metric_mu) / metric_si
normalized_nuclei_datum = (nuclei_datum - nuclei_mu) / nuclei_si
normalized_energy_datum = (energy_datum - energy_mu) / energy_si

print(metric_mu)
print(metric_si)
print(nuclei_mu)
print(nuclei_si)
print(energy_mu)
print(energy_si)
