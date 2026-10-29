#!/usr/bin/env python3
# coding: utf-8
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from histclass.tools.env import get_env
from histclass.interfaces.data.types import DataframeType
from histclass.backends.pytorch.utils import load_torch
from histclass.tools.training.lib import matrix2acc, matrix2mcc, consensus
from histclass.extensions.glob_ext import globs

# 1. Load Model and Dataset

N_SAMPLES = 1000
INDEX = get_env("DATA_INDEX").catch("0")(int).unwrap()

model_list = sorted(
    globs("ihc/models/classification/*hilograde*")
    + globs("ihc/models/classification/*hilostage*")
    + globs("ihc/models/classification/*2tn*")
)
nimage_model_list = [ m for m in model_list if "image" not in m ]
image_model_list = [ m for m in model_list if "image" in m ]
model_list = image_model_list
assert INDEX < len(model_list)

print(INDEX)
print(len(model_list))
model_name = model_list[INDEX]
model_name = model_name.split("/")[-1]
print(model_name)

# Load Model and Dataset
model, dataset = load_torch(
    model_name
)
model = model.to("cuda")
data_loaders = dataset.get_dataloaders()
tdl, vdl, Tdl, tcdl, vcdl = data_loaders

results_dict = {
        "T":{0: {}, 1: {}},
        "V":{0: {}, 1: {}},
        }

for dataframe in tqdm(dataset.data_frames):
    _key = "T"
    if dataframe.usage == DataframeType.Validation:
        _key = "V"
    elif dataframe.usage == DataframeType.Training:
        _key = "T"
    else:
        continue
    for _ in range(N_SAMPLES):
        x, l, n = dataframe[0]
        xshape = x.shape
        x = x.to("cuda").reshape((1,*xshape))
        p = model.forward(x)
        l = l.detach().cpu().numpy().argmax()
        p = p.detach().cpu().numpy().flatten().tolist()
        if n not in results_dict[_key][l].keys():
            results_dict[_key][l][n] = []
        results_dict[_key][l][n].append(p)

import pickle
with open(f"raw_probabilities/{model_name}.pickle", "wb") as fhandle:
    pickle.dump(results_dict, fhandle)
