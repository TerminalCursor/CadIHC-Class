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
    if is_ablation:
        continue
    data_type = data_type.split("_")[0]
    data_type = data_type + "%" + meta_type
    if data_type not in res_data.keys():
        res_data[data_type] = {
            "accuracies": [],
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
        out = (reg_data[0], con_data[0], reg_data[1], con_data[1])
        res_data[data_type]["accuracies"].append((*out, sum([reg_data[0],con_data[0]]), sum([reg_data[1],con_data[1]]), sum(out)))

for k in res_data.keys():
    data = res_data[k]["accuracies"]
    data = list(sorted(data, key=lambda item: item[-2]))
    data = data[-1][:4]
    print(f"{k:35} ", end='')
    print(data[1], data[3])
    #print(dumps(data, indent=2))

#print(dumps(res_data, indent=2))

#bulk_statistics = {
#    data_type: {
#        k: {
#            "min": float(np.min(res_data[data_type][k])),
#            "avg": float(np.mean(res_data[data_type][k])),
#            "max": float(np.max(res_data[data_type][k])),
#        }
#        for k in res_data[data_type].keys()
#    }
#    for data_type in res_data.keys()
#}
#
#print(dumps(bulk_statistics, indent=2))
