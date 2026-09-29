#!/usr/bin/env python
# coding: utf-8

from pc_util import *
import torch
import torch.nn.functional as F
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
from json import dumps
import pickle

data_sources = sorted(glob("../Datasets/HT-*/"))
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
    other = "-".join(base.split("-")[1:]).split(".")[0]
    base = base.split("-")[0]
    dtype, mtype = base.split("_classifier_")
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


def load_config(data_type, meta_type, patches):
    config = {
        "input_data_type": data_type,
        "output_data_type": meta_type,
        "epochs": 20_000,
        "learning_rate": 5e-6,
        "device": "cuda",
        "batch_size": 8,
        "train_val_split": 0.8,
        "loss_fn": "mse_loss",
        "image_values": {
            "window_width": 1920,
            "window_height": 2560,
        },
        "pointcloud_values": {
            "max_points": 96,
            "min_points": 64,
            "dropout": 0.15,
            "dropout_output": False,
            "n_ff_nodes": 32,
            "n_ff_layers": 3,
        },
        "early_exit_kwargs": {
            "start_after": 100,
            "lookback": 600,
            "p_cutoff": 0.05,
            "on_validation": True,
        },
        "early_exit_max_failures": 200,
        "vision_transformer_config": {
            "image_size": patches,
            "patch_size": 16,
            "num_layers": 4,
            "num_heads": 4,
            "hidden_dim": 16,
            "mlp_dim": 16,
            "dropout": 0.05,
            "attention_dropout": 0.05,
        },
    }
    if config["input_data_type"] == "image_vt":
        assert patches is not None
        config["image_values"]["window_width"] = config["vision_transformer_config"][
            "image_size"
        ]
        config["image_values"]["window_height"] = config["vision_transformer_config"][
            "image_size"
        ]
    if config["input_data_type"] in [
        "metrics",
        "energetics",
        "metricenergetics",
        "latent",
        "random",
    ]:
        # Point Cloud Model
        del config["image_values"]
        del config["vision_transformer_config"]
    elif config["input_data_type"] == "image":
        del config["vision_transformer_config"]
        del config["pointcloud_values"]
    elif config["input_data_type"] == "image_vt":
        del config["pointcloud_values"]
    return config


def model_load(model, checkpoint_file):
    from torch import load

    checkpoint = load(checkpoint_file)
    return model.load_state_dict(checkpoint)


def model_forward(model, data):
    with torch.no_grad():
        out = model(data)
        if not np.isclose(out.sum().item(), 1):
            out = F.softmax(out, dim=1)
        return out


def dataset_length(dataloader):
    return len(dataloader.dataset)


def get_data(dataloader, index, label_indices=[0]):
    data, *labels = dataloader.dataset[index]
    data = data[None, ...]
    return data, [labels[idx] for idx in label_indices]


def scores(model, dataloader, n_tests=1):
    from sklearn.metrics import roc_auc_score

    roc_results = []
    for data_index in range(dataset_length(dataloader)):
        for _ in range(n_tests):
            data, label = get_data(dataloader, data_index)
            label = int(label[0].item())
            label_prob = model_forward(model, data)
            label_prob = label_prob.cpu().numpy().flatten()
            roc_results.append((label, *label_prob))
    roc_results = np.array(roc_results)
    y_true = roc_results[:, 0]
    y_pred_raw = roc_results[:, 1:]
    y_pred = y_pred_raw.argmax(axis=1)
    print(np.unique(y_true, return_counts=True))
    print(np.unique(y_pred, return_counts=True))
    aucs = []
    label_scores = []
    for label in range(y_pred_raw.shape[-1]):
        auc = roc_auc_score(y_true == label, y_pred_raw[:, label])
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        tn = ((y_true != label) & (y_pred != label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        aucs.append((float(auc),))
        label_scores.append((int(tp), int(fp), int(tn), int(fn)))
    return aucs, label_scores


def find_best(x_var: str, y_var: str, category: str, results_df: DataFrame, n=3):
    from pandas import concat

    metric_name = "L2Q"
    results_df = results_df.filter_on("CATEGORY", category)

    # New columns
    results_df["MODEL"] = results_df["NAME"].map(results_df["NAME"].enumerate())

    M = results_df.get("NAME")
    models = M.unique()  # pyright: ignore
    rows = []
    for m_idx, model in enumerate(models):
        df = results_df.filter_on("NAME", model)
        max_row = metric(df, x_var, y_var)
        max_row = max_row.head(n)
        rows.append(max_row)
    return concat(rows)


data_labels = ["t", "v"]
categories = ["stage_basic", "tn"]
data_assoc = {tv: {c: {} for c in categories} for tv in data_labels}
for data_label in data_labels:
    for category in categories:
        best_df = find_best(
            "VALIDATION_ACCURACY", "TOTAL_ACCURACY", category, results_df
        )
        print(best_df)

        for timestamp in sorted(best_df["TIMESTAMP"]):
            file_l = globs(f"{BASE}*{timestamp}.pth")
            if len(file_l) == 0:
                file_l = globs(f"{BASE}*/*{timestamp}.pth")
            assert len(file_l) == 1, f"{BASE}*/*{timestamp}.pth not found!"
            loc = file_l[0]
            data_type, meta_type, patches, ext = parse(loc, f"{BASE}")

            config = load_config(data_type, meta_type, patches)

            histology_dataset, N_data, N_class, histology_datasets = get_dataset(
                data_sources, config
            )

            t_dataloader, v_dataloader, histology_dataloader = train_eval_split(
                histology_dataset, config
            )

            model = get_model(config, N_data, N_class)
            model_load(model, loc)

            N_tests = 100

            # Compute RoC curves for each label
            patches = "" if patches is None else f"_{patches}"
            dataloader = None
            if data_label == "t":
                dataloader = t_dataloader
            elif data_label == "v":
                dataloader = v_dataloader
            else:
                dataloader = histology_dataloader
            aucs, counts = scores(model, dataloader, n_tests=N_tests)
            counts = np.array(counts)
            aucs = np.array(aucs)
            sensitivity = counts[:, [0]] / (counts[:, [0]] + counts[:, [1]])
            specificity = counts[:, [2]] / (counts[:, [2]] + counts[:, [3]])
            # Normalizing by True Label Counts for Each
            ncounts = counts.copy().astype(float)
            ncounts[:, 0] = counts[:, 0] / (counts[:, 0] + counts[:, 3])
            ncounts[:, 1] = counts[:, 1] / (counts[:, 1] + counts[:, 2])
            ncounts[:, 2] = counts[:, 2] / (counts[:, 1] + counts[:, 2])
            ncounts[:, 3] = counts[:, 0] / (counts[:, 0] + counts[:, 3])
            nsensitivity = ncounts[:, [0]] / (ncounts[:, [0]] + ncounts[:, [1]])
            nspecificity = ncounts[:, [2]] / (ncounts[:, [2]] + ncounts[:, [3]])
            score = np.hstack(
                [aucs, sensitivity, specificity, nsensitivity, nspecificity]
            ).T
            print(aucs)
            print(counts)
            print(counts.shape)
            print(score)
            print(score.shape)
            counts = np.hstack([counts, ncounts])
            counts = counts.tolist()
            score = score.tolist()
            if data_type not in data_assoc[data_label][meta_type].keys():
                data_assoc[data_label][meta_type][data_type] = {
                    "counts": [],
                    "score": {
                        "auc": [],
                        "tpr": [],
                        "tnr": [],
                        "ntpr": [],
                        "ntnr": [],
                    },
                }
            data_assoc[data_label][meta_type][data_type]["counts"].append(counts)
            data_assoc[data_label][meta_type][data_type]["score"]["auc"].append(
                score[0][:]
            )
            data_assoc[data_label][meta_type][data_type]["score"]["tpr"].append(
                score[1][:]
            )
            data_assoc[data_label][meta_type][data_type]["score"]["tnr"].append(
                score[2][:]
            )
            data_assoc[data_label][meta_type][data_type]["score"]["ntpr"].append(
                score[3][:]
            )
            data_assoc[data_label][meta_type][data_type]["score"]["ntnr"].append(
                score[4][:]
            )
with open("outputs/scores.pickle", "wb+") as fhandle:
    pickle.dump(data_assoc, fhandle)
