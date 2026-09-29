#!/usr/bin/env python
# coding: utf-8

from pc_util import *
import torch
import torch.nn.functional as F
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay

data_sources = sorted(glob("../Datasets/HT-*/"))
# loc = "dump/models/nuclei_classifier_stage_basic-20260201-234551-857813.pth"
# loc = "models_cqls/energetics_classifier_tn-20260205-192935-286827.pth"
# loc = "models_cqls/image_vt_classifier_stage_basic-20260208-015352-598828.pth"


def parse(model_file):
    from sqlite3 import connect
    import pandas as pd

    base = model_file.split("/")[-1]
    other = "-".join(base.split("-")[1:]).split(".")[0]
    base = base.split("-")[0]
    dtype, mtype = base.split("_classifier_")
    con = connect("dump/Results.db")
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


def get_roc_data(model, dataloader, n_tests=1):
    roc_results = []
    for data_index in range(dataset_length(dataloader)):
        for _ in range(n_tests):
            data, label = get_data(dataloader, data_index)
            label = int(label[0].item())
            label_prob = model_forward(model, data)
            label_prob = label_prob.cpu().numpy().flatten()
            roc_results.append((label, *label_prob))
    roc_results = np.array(roc_results)
    return roc_results


def plot_rocs(fname_base, roc_data):
    for label_index in range(1, roc_data.shape[-1]):
        plt.clf()
        label_matches = roc_data[:, 0] == label_index - 1
        label_prob = roc_data[:, label_index]
        fpr, tpr, _ = roc_curve(label_matches, label_prob, pos_label=True)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.plot([0, 1], [0, 1], "r--")
        plt.savefig(f"{fname_base}{label_index}.png")


for loc in sorted(glob("dump/models/*.pth")):
    data_type, meta_type, patches, ext = parse(loc)

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
    for data_label, dataloader in zip(("t", "v"), (t_dataloader, v_dataloader)):
        data_results = get_roc_data(model, dataloader, n_tests=N_tests)
        plot_rocs(
            f"../out_img/{data_type}{patches}_class_{meta_type}_T{ext}S_{data_label}",
            data_results,
        )
