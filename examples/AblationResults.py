#!/usr/bin/env python
# coding: utf-8

# from pc_util import *
import torch
import torch.nn.functional as F
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
from json import dumps
import pickle
from histclass.backends.pytorch.data.normalization import normalization_attributes

data_sources = sorted(glob("ihc/data/classification/*.*/"))
# loc = "dump/models/nuclei_classifier_stage_basic-20260201-234551-857813.pth"
# loc = "models_cqls/energetics_classifier_tn-20260205-192935-286827.pth"
# loc = "models_cqls/image_vt_classifier_stage_basic-20260208-015352-598828.pth"

from histclass.extensions.glob_ext import globs
from histclass.tools.env import get_env
from histclass.extensions.sqlite3_ext import legacy_sql_load as sql_load

BASE = get_env("DATA_BASE")
BASE = BASE.catch("dump/").unwrap()

SQLITE_FILE = f"{BASE}Results.db"
results_df = sql_load(SQLITE_FILE, "RESULTS")


def metric(df, x_var, y_var):
    """# A standard L2 metric but the distance from (1,1)"""
    df["XY"] = ((1 - df[[x_var, y_var]]) ** 2.0).sum(axis=1)
    # max_row = df.loc[df["XY"] == df["XY"].min()].copy()
    max_row = df.copy().sort_values(by="XY", ascending=True)
    return max_row


def parse(model_file, BEST):
    from sqlite3 import connect
    import pandas as pd

    base = model_file.split("/")[-1]
    dtype = base.split("-")[0]
    mtype = base.split("-attention2")[1].split("-")[0]
    other = "-".join(base.split("-attention2")[1].split("-")[1:])
    # other = "-".join(base.split("-")[1:]).split(".")[0]
    # base = base.split("-")[0]
    # dtype, mtype = base.split("-attention2")
    con = connect(f"{BEST}Results.db")
    df = pd.read_sql_query(f"SELECT * FROM RESULTS WHERE TIMESTAMP = '{other}'", con)
    con.close()
    patches = None
    if df.shape[0] == 1:
        dtype = df["NAME"].head(1).item()
        mtype = df["CATEGORY"].head(1).item()
    if dtype[: len("image_vt")] == "image_vt":
        patches = int(dtype[len("image_vt") :])
        dtype = "image_vt"
    return dtype, mtype, patches, other


res_data = {}
for timestamp in sorted(results_df["TIMESTAMP"]):
    file_l = globs(f"{BASE}*{timestamp}.pth")
    if len(file_l) == 0:
        file_l = globs(f"{BASE}*/*{timestamp}.pth")
    assert len(file_l) == 1, f"{BASE}*/*{timestamp}.pth not found!"
    loc = file_l[0]
    data_type, meta_type, patches, ext = parse(loc, f"{BASE}")
    is_ablation = "_" in data_type
    ext = ext[: -len(".pth")]
    if not is_ablation:
        continue
    ablations = "_".join(data_type.split("_")[1:])
    data_type = data_type.split("_")[0]
    data_type = data_type + "%" + meta_type
    if data_type not in res_data.keys():
        res_data[data_type] = {}
    if ablations not in res_data[data_type].keys():
        res_data[data_type][ablations] = {
            "to_accuracy": [],
            "tn_accuracy": [],
            "vo_accuracy": [],
            "vn_accuracy": [],
        }

    with open(f"{BASE}{ext}.txt", "r") as fhandle:
        data = fhandle.read()
        json_data = data.split("}\n")[0] + "}"
        remnants = data.split("}\n")[-1]
        boot_data = remnants.split("Bootstrapped ")
        reg_data = boot_data[1].split("\n")[-3:]
        con_data = boot_data[2].split("\n")[-3:]
        if reg_data[-1] == "":
            reg_data = reg_data[:2]
        else:
            reg_data = reg_data[1:]
        if con_data[-1] == "":
            con_data = con_data[:2]
        else:
            con_data = con_data[1:]
        reg_data = [float(v) for v in reg_data]
        con_data = [float(v) for v in con_data]
        # print(data_type)
        # print(reg_data)
        # print(con_data)
        res_data[data_type][ablations]["to_accuracy"].append(reg_data[0])
        res_data[data_type][ablations]["tn_accuracy"].append(con_data[0])
        res_data[data_type][ablations]["vo_accuracy"].append(reg_data[1])
        res_data[data_type][ablations]["vn_accuracy"].append(con_data[1])

bulk_statistics = {
    data_type: {
        ablations: {
            k: {
                "min": float(np.min(res_data[data_type][ablations][k])),
                "avg": float(np.mean(res_data[data_type][ablations][k])),
                "max": float(np.max(res_data[data_type][ablations][k])),
            }
            for k in res_data[data_type][ablations].keys()
        }
        for ablations in res_data[data_type].keys()
    }
    for data_type in res_data.keys()
}


def _ablation_to_name(data_type, ablations):
    abl = ablations.split("_")
    abl = [int(v) for v in abl]
    data_type = data_type.split("%")[0]
    if data_type == "latent":
        names = ["Latent-0", "Latent-1"]
        return [names[a] for a in abl]
    if data_type == "energetics":
        names = [
            "Internal Pressure",
            "Surface Tension",
            "Cell-Cell Adhesion",
            "Cell-Stroma Adhesion",
        ]
        return [names[a] for a in abl]
    if data_type == "morphology":
        names = normalization_attributes["metric"]["columns"]
        return [names[a] for a in abl]
    if data_type == "morphologynuclei":
        names1 = normalization_attributes["metric"]["columns"]
        names2 = normalization_attributes["nuclei"]["columns"]
        output_values = []
        for a in abl:
            if a < 13:
                output_values.append(names1[a])
            else:
                output_values.append(names2[a - 13])
        return output_values
    if data_type == "morphologyenergetics":
        names1 = normalization_attributes["metric"]["columns"]
        names2 = [
            "Internal Pressure",
            "Surface Tension",
            "Cell-Cell Adhesion",
            "Cell-Stroma Adhesion",
        ]
        output_values = []
        for a in abl:
            if a < 13:
                output_values.append(names1[a])
            else:
                output_values.append(names2[a - 13])
        return output_values
    if data_type in normalization_attributes.keys():
        names = normalization_attributes[data_type]["columns"]
        return [names[a] for a in abl]
    return [abl]


def ablation_to_name(data_type, ablations):
    return "_".join([str(v) for v in _ablation_to_name(data_type, ablations)])


flattened_statistics = {
    data_type: {
        ablation_to_name(data_type, ablations): float(
            np.max(
                [
                    float(
                        np.round(
                            bulk_statistics[data_type][ablations][k]["max"] * 100, 1
                        )
                    )
                    for k in bulk_statistics[data_type][ablations].keys()
                ]
            )
        )
        for ablations in bulk_statistics[data_type].keys()
    }
    for data_type in bulk_statistics.keys()
}

for k in flattened_statistics.keys():
    print(k)
    data = flattened_statistics[k]
    data = dict(sorted(data.items(), key=lambda item: item[1]))
    print(dumps(data, indent=2))
#print(dumps(flattened_statistics, indent=2))

for k in flattened_statistics.keys():
    print(k)
    data = flattened_statistics[k]
    data = dict(sorted(data.items(), key=lambda item: item[1]))
    print(dumps(data, indent=2))

for k in flattened_statistics.keys():
    print(f"{k:35} ", end='')
    data = flattened_statistics[k]
    data = dict(sorted(data.items(), key=lambda item: item[1]))
    span_data = [min(data.values()), max(data.values())]
    print(span_data, np.round(span_data[1] - span_data[0], 2))
