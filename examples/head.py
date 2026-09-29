#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[217]:


# get_ipython().system('nvidia-smi')


# ## Imports

# In[218]:


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
    # "image", "metrics", "energetics", "metricenergetics", "latent", "random", "image_vt",
    # "metrics", "energetics", "metricenergetics", "latent",
    # "image", "image_vt",
    "metrics",
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
data_type = "metrics"
patches = 128
meta_type = "stage_basic"
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

histology_dataset, N_data, N_class, histology_datasets = get_dataset(
    data_sources, config
)


t_dataloader, v_dataloader, histology_dataloader = train_eval_split(
    histology_dataset, config
)

print(f"N classes: {N_class}")
model = get_model(config, N_data, N_class)

for out in t_dataloader:
    print(out)
