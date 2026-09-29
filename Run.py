#!/usr/bin/env python
# coding: utf-8

import os

SLURM_ID = os.getenv("SLURM_ARRAY_TASK_ID")

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

data_type_possibilities = [
    "nuclei_only",
    "metrics",
    "nuclei",
    "energetics",
    "metricenergetics",
    "latent",
    "image",
    "image_vt",
    # "image", "image_vt",
]
meta_type_possibilities = [
    "grade",
    "stage_basic",
    "tn",
    # "stage_advanced",
]
from itertools import product

# _N = 35
# _N = 5
_N = 10
combos = list(
    product(
        range(1),
        data_type_possibilities,
        meta_type_possibilities,
        [256, 512, 1024],
        range(_N),
    )
)
print(len(combos))
# assert False
# If running this as an array task, only one one possibility
if SLURM_ID is not None:
    SLURM_ID = int(SLURM_ID) - 1
    combos = [combos[SLURM_ID]]
for _, data_type, meta_type, patches, _ in combos:
    if (data_type != "image_vt") and (patches in [512, 1024]):
        continue
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

    # In[34]:

    if config["input_data_type"] == "image_vt":
        config["image_values"]["window_width"] = config["vision_transformer_config"][
            "image_size"
        ]
        config["image_values"]["window_height"] = config["vision_transformer_config"][
            "image_size"
        ]

    # In[35]:

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

    # ## Runtime Configuration Validation

    # In[225]:

    loss_fn = None
    if config["loss_fn"] == "mse_loss":
        loss_fn = F.mse_loss
    elif config["loss_fn"] == "cross_entropy":
        loss_fn = F.cross_entropy
    else:
        assert False, f"Loss Function Not Recognized! {config['loss_fn']}"

    # ## Logger Setup and Record Keeping

    # In[226]:

    logger = Logger("dump/")

    # In[227]:

    logger.addnl(json.dumps(config, indent=4))
    head = (
        config["input_data_type"]
        + f'{config["vision_transformer_config"]["image_size"]}'
        if (config["input_data_type"] == "image_vt")
        else config["input_data_type"]
    )
    logger.meta(
        head, config["output_data_type"], config["learning_rate"], config["loss_fn"]
    )

    # # Preparing the Data

    # In[228]:

    histology_dataset, N_data, N_class, histology_datasets = get_dataset(
        data_sources, config
    )

    # In[229]:

    t_dataloader, v_dataloader, histology_dataloader = train_eval_split(
        histology_dataset, config
    )

    # # Preparing the Model

    # In[230]:

    print(f"N classes: {N_class}")
    model = get_model(config, N_data, N_class)

    # In[231]:

    logger.addnl(f"{model}")

    # ## Optimizer

    # In[232]:

    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    logger.addnl(f"{optim}")

    # # Training

    # In[233]:

    losses = []
    cat_accs = []

    # In[234]:

    # for data, label in histology_dataloader:
    #    print(data.shape, label.shape)

    # ## Pre-training evaluation

    # In[235]:

    t_l, t_a, t_am = eval_epoch_ohe(
        model, t_dataloader, N_class, device=config["device"], loss_fn=loss_fn
    )
    v_l, v_a, v_am = eval_epoch_ohe(
        model, v_dataloader, N_class, device=config["device"], loss_fn=loss_fn
    )
    losses.append((t_l, v_l))
    cat_accs.append((t_a, v_a))
    output_data = {
        "tca": cat_accs[-1][0],
        "vca": cat_accs[-1][1],
    }
    logger.add(f"Epoch {-1: 5}: ")
    logger.addnl(f"{output_data}")
    t_l, v_l

    # ## Training Loop

    # In[ ]:

    pbar = tqdm(range(config["epochs"]), miniters=100)
    current_sequential_failures = 0
    logger.echo = False
    for epoch in pbar:
        t_l, t_a, t_am = train_epoch_ohe(
            model,
            t_dataloader,
            optim,
            N_class,
            device=config["device"],
            loss_fn=loss_fn,
        )
        v_l, v_a, v_am = eval_epoch_ohe(
            model, v_dataloader, N_class, device=config["device"], loss_fn=loss_fn
        )

        losses.append((t_l, v_l))
        cat_accs.append((t_a, v_a))
        output_data = {
            "tca": cat_accs[-1][0],
            "vca": cat_accs[-1][1],
        }
        logger.add(f"Epoch {epoch: 5}: ")
        logger.addnl(f"{output_data}")

        pbar.set_postfix(output_data)
        if early_exit_detector(losses, **config["early_exit_kwargs"]):
            current_sequential_failures += 1
        else:
            current_sequential_failures = 0
        if current_sequential_failures >= config["early_exit_max_failures"]:
            break
    logger.echo = True

    # ## Post-training Evaluation

    # In[ ]:

    t_l, t_a, t_am = eval_epoch_ohe(
        model, t_dataloader, N_class, device=config["device"], loss_fn=loss_fn
    )
    v_l, v_a, v_am = eval_epoch_ohe(
        model, v_dataloader, N_class, device=config["device"], loss_fn=loss_fn
    )
    T_l, T_a, T_am = eval_epoch_ohe(
        model, histology_dataloader, N_class, device=config["device"], loss_fn=loss_fn
    )
    losses.append((t_l, v_l))
    cat_accs.append((t_a, v_a))
    output_data = {
        "tca": cat_accs[-1][0],
        "vca": cat_accs[-1][1],
    }
    logger.add(f"Epoch {epoch+1: 5}: ")
    logger.addnl(f"{output_data}")

    # # Training Graphs

    # In[ ]:

    from matplotlib import pyplot as plt

    # get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(cat_accs, lw=1)

    # In[ ]:

    from matplotlib import pyplot as plt

    # get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(losses, lw=1)

    # # Log Evaluation Results

    # In[ ]:

    from torch import no_grad

    with no_grad():
        from numpy import eye, array, mean
        from torch import no_grad

        logger.addnl(f"{T_am}")
        logger.addnl(f"{T_a}\n")
        logger.addnl(f"{t_am}")
        logger.addnl(f"{t_a}\n")
        logger.addnl(f"{v_am}")
        logger.addnl(f"{v_a}\n")
        logger.assoc(total_accuracy=float(T_a))
        logger.assoc(training_accuracy=float(t_a))
        logger.assoc(validation_accuracy=float(v_a))
        logger.assoc(total_raw=T_am)
        logger.assoc(training_raw=t_am)
        logger.assoc(validation_raw=v_am)

        N_tests = 100
        pass_results = [0, 0]
        for output in analyze_model_performance_on_datasets(
            model, histology_datasets, n_tests=N_tests
        ):
            name, thresh, p = output
            logger.addnl(f"{name}: {thresh} {p}")
            if thresh:
                pass_results[0] += 1
            pass_results[1] += 1
        logger.addnl(f"\n{pass_results[0]/pass_results[1]}")
        logger.assoc(total_accuracy_c=float(pass_results[0] / pass_results[1]))

        # Train Consensus
        pass_results = [0, 0]
        is_image = config["input_data_type"][: len("image")] == "image"
        dataloader = t_dataloader
        for data_idx in range(len(dataloader.dataset)):
            for _ in range(N_tests):
                data, label = dataloader.dataset[data_idx]
                data = data[None, ...]
                label = label.item()
                label_pred = model(data).argmax(dim=-1).item()
                if label == label_pred:
                    pass_results[0] += 1
                pass_results[1] += 1
        # logger.add_data(float(pass_results[0]/pass_results[1]))
        logger.assoc(training_accuracy_c=float(pass_results[0] / pass_results[1]))

        # Validation Consensus
        pass_results = [0, 0]
        dataloader = v_dataloader
        for data_idx in range(len(dataloader.dataset)):
            for _ in range(N_tests):
                data, label = dataloader.dataset[data_idx]
                data = data[None, ...]
                label = label.item()
                label_pred = model(data).argmax(dim=-1).item()
                if label == label_pred:
                    pass_results[0] += 1
                pass_results[1] += 1
        # logger.add_data(float(pass_results[0]/pass_results[1]))
        logger.assoc(validation_accuracy_c=float(pass_results[0] / pass_results[1]))

    logger.dump()

    torch.save(
        model.state_dict(),
        f"dump/models/{config['input_data_type']}_classifier_{head}-{logger.timestamp}.pth",
    )
