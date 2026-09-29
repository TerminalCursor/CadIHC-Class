#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
import numpy as np

from glob import glob

data_sources = sorted(glob("../Datasets/HT-*/"))

from pc_util import *

import json
from sklearn.metrics import roc_curve, RocCurveDisplay

data_type = "nuclei"
meta_type = "tn"
meta_type = "stage_basic"
# meta_type = "stage_advanced"
from itertools import product

_N = 10
_N = 1
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

histology_dataset, N_data, N_class, histology_datasets = get_dataset(
    data_sources, config
)

_REGEN = False
assert not _REGEN, "Are you sure you want to regen?"
t_dataloader, v_dataloader, histology_dataloader = train_eval_split(
    histology_dataset, config, regen=_REGEN
)

# Regenerate Dataset Assignment
if _REGEN:
    import json

    for ds in histology_datasets:
        _json = {}
        with open(ds.name + "/metadata.json", "r") as fhandle:
            _json = json.load(fhandle)
            _json["assignment"] = ds.assignment
        with open(ds.name + "/metadata.json", "w") as fhandle:
            fhandle.write(json.dumps(_json, indent=2))
    print("Regenerated!")
