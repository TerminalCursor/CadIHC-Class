#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[217]:


# get_ipython().system('nvidia-smi')


# ## Imports

# In[218]:

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F

# In[219]:


from tqdm.auto import tqdm

# In[220]:


from glob import glob

data_sources = sorted(glob("../Datasets/HT-*/"))


# In[221]:


from pc_util import *

# In[222]:


import json

# ## Runtime Configuration

# In[224]:


data_type_possibilities = [
    # "image", "metrics", "energetics", "random",
    "image",
    "metrics",
    "energetics",
    "metricenergetics",
    "latent",
    "random",
    "image_vt",
    # "metrics", "energetics", "metricenergetics", "latent",
    # "image", "image_vt",
    # "metrics",
    # "image", "image_vt",
]
meta_type_possibilities = [
    "tp",
    "stage_basic",
    "stage_advanced",
]
from itertools import product

_N = 5
_N = 1
data_type = data_type_possibilities[1]
meta_type = "tp"
# meta_type = "stage_basic"
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
        "image_size": 1,
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


# loss_fn = None
# if config['loss_fn'] == "mse_loss":
#    loss_fn = F.mse_loss
# elif config['loss_fn'] == "cross_entropy":
#    loss_fn = F.cross_entropy
# else:
#    assert False, f"Loss Function Not Recognized! {config['loss_fn']}"


# ## Logger Setup and Record Keeping

# In[226]:


# logger = Logger("dump/")


# In[227]:


# logger.addnl(json.dumps(config, indent=4))
# head = config["input_data_type"] + f"{config["vision_transformer_config"]["image_size"]}" if (config["input_data_type"] == "image_vt") else config["input_data_type"]
# logger.set_header(head,
#        config["output_data_type"],
#        config["learning_rate"],
#        config["loss_fn"],
#        )


# # Preparing the Data

# In[228]:


histology_dataset, N_data, N_class, histology_datasets = get_dataset(
    data_sources, config
)


# In[229]:


t_dataloader, v_dataloader, histology_dataloader = train_eval_split(
    histology_dataset, config
)

umap_dataset = []

BASE_IDX = 0
N_contrib = 100
dataloader = t_dataloader
for data_idx in range(len(dataloader.dataset)):
    for _ in range(N_contrib):
        data, label = dataloader.dataset[data_idx]
        data = data.detach().cpu().numpy()
        data = data.ravel()
        label = label.detach().cpu().numpy()
        umap_dataset.append((data, label, BASE_IDX))
    BASE_IDX += 1
dataloader = v_dataloader
for data_idx in range(len(dataloader.dataset)):
    for _ in range(N_contrib):
        data, label = dataloader.dataset[data_idx]
        data = data.detach().cpu().numpy()
        data = data.ravel()
        label = label.detach().cpu().numpy()
        umap_dataset.append((data, label, BASE_IDX))
    BASE_IDX += 1

model = umap.UMAP()
# _data, _label, _idx = zip(umap_dataset)
_data, _label, _idx = zip(*umap_dataset)
import numpy as np

_data = np.array(_data)
_label = np.array(_label)
_label = _label.ravel()
_idx = np.array(_idx)
print(_data.shape, _label.shape, _idx.shape)
u = model.fit_transform(_data)

BASE_IDX = 0
if meta_type == "tp":
    BASE_IDX = BASE_IDX + 2 + 3

from matplotlib import pyplot as plt

plt.clf()
plt.scatter(u[:, 0], u[:, 1], c=_label.ravel(), s=1)
plt.savefig(f"/home/physics/cunnichr/out_data/{BASE_IDX+0}.png", dpi=600)

plt.clf()
plt.scatter(u[:, 0], u[:, 1], c=_idx.ravel(), s=1)
plt.savefig(f"/home/physics/cunnichr/out_data/{BASE_IDX+1}.png", dpi=600)

for i in range(len(np.unique(_label))):
    a = u[_label == i]
    b = u[_label != i]
    d = np.vstack([b, a])
    al = _label[_label == i]
    bl = _label[_label != i]
    dl = np.hstack([bl, al])
    plt.clf()
    plt.scatter(d[:, 0], d[:, 1], c=dl, s=len(np.unique(dl)) - dl, vmin=0, vmax=3)
    plt.savefig(f"/home/physics/cunnichr/out_data/{BASE_IDX+2+i}.png", dpi=600)
