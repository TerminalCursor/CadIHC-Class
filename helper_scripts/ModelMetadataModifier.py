#!/usr/bin/env python3
import pickle
import numpy as np
from random import seed
from histclass.tools.env import get_env
from histclass.extensions.glob_ext import globs

_K = 5
_SEED = 42
# _SEED = None
print(_SEED)

N_MAX = 1  # Maximum number of experts to add to pool in MoE
N_BOOTSTRAP = 50
N_SAMPLES = 100
N_CLASSES = 2  # All are binary categories
BAYESIAN = True
TOP_K = _K
DATA_KEYS = [
    "ACC",
    "ACC_CI_L",
    "ACC_CI_H",
    "AUROC",
    "AUROC_CI_L",
    "AUROC_CI_H",
    "MCC",
    "MCC_CI_L",
    "MCC_CI_H",
    "Score",
]

BAYESIAN_SCALE = True
BAYESIAN_BEFORE = True
WEIGHTS = False
NORMALIZED_WEIGHTS = False
BS_T = "Scaled" if BAYESIAN_SCALE else "Unscaled"
BS_B = "Outside" if BAYESIAN_BEFORE else "Inside"
BS_N = "NormalizedWeights" if NORMALIZED_WEIGHTS else ""
BS_W = "Weights" if WEIGHTS else ""
PREFIX = f"{BS_T}{BS_B}{BS_N}{BS_W}-"


def compute_acc(matrix):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)
        confidences = np.diag(matrix) / np.sum(matrix, axis=0)
    return accuracy, confidences


if _SEED is None:
    _SEED = np.random.randint(10000)
    print(_SEED)
seed(_SEED)
np.random.seed(_SEED)

# For the sake of re-running this analysis, draw from outputs of each model running on each image 1000 times
# Structured dict as:
#  "T" or "V"  for training or validation
#  0 or 1 for class label of the images
#  Image name like "TMA00.A01" for image name
#  Enclosed is a 1000 length list of outputs
BASE = get_env("FILE_BASE").catch("raw_probabilities/").unwrap()

file_list = globs(BASE + "*.pickle")

model_data = dict()
full_data = dict()
for fname in file_list:
    if "latent" in fname:
        continue
    mname = fname.split("/")[-1][: -len(".pickle")]
    with open(fname, "rb") as fhandle:
        full_data[mname] = pickle.load(fhandle)

# Compute the Accuracy and TPR on the training data
for model in full_data.keys():
    category = model.split(".")[0].split("2")[-1]
    model_type = model.split("-")[0]
    matrix = np.array(
        [
            [
                0,
            ]
            * N_CLASSES
        ]
        * N_CLASSES
    )
    data = full_data[model]["T"]
    for class_i in range(N_CLASSES):
        cdata = data[class_i]
        data_names = cdata.keys()
        data_indicies = {k: v for (k, v) in enumerate(data_names)}
        for _ in range(N_SAMPLES):
            p = [
                0,
            ] * N_CLASSES
            dindex = np.random.randint(len(data_names))
            dname = data_indicies[dindex]
            ddata = cdata[dname]
            for _ in range(9):
                index = np.random.randint(len(ddata))
                mx = ddata[index]
                p = [float(p[i] + mx[i]) for i in range(N_CLASSES)]
            t = class_i
            p = np.argmax(p)
            matrix[t][p] += 1
    accuracy, confidences = compute_acc(matrix)
    # accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)
    # confidences = np.diag(matrix) / np.sum(matrix, axis=0)
    model_data.setdefault(category, {})[model] = {
        "acc": float(accuracy),
        "conf": confidences.tolist(),
    }

# Find the best N_MAX performing models of each type
# to add to MoE model pool
model_pool = []
moe_model_pool = []
for category in model_data.keys():
    models = list(model_data[category].keys())
    model_pool.extend(models)
    model_types = set([m.split("2")[0] for m in models])
    models_to_keep = []
    for model_type in model_types:
        accuracies = [
            (m, model_data[category][m]["acc"])
            for m in models
            if m.startswith(model_type)
        ]
        accuracies = sorted(accuracies, key=lambda x: x[1])
        accuracies = accuracies[-N_MAX:]
        for m, _ in accuracies:
            models_to_keep.append(m)
            moe_model_pool.append(m)


def add_metadata(model_name, moe_model=False):
    from json import load, dumps
    from numpy import nan_to_num

    model_file = f"ihc/models/classification/{model_name}/metadata.json"
    model_metadata = dict()
    with open(model_file, "r") as fhandle:
        model_metadata = load(fhandle)
    try:
        del model_metadata["non-consensus"]
    except:
        pass
    try:
        del model_metadata["consensus"]
    except:
        pass
    cat = model_file.split("/")[-2].split(".")[0].split("2")[1]
    model_data[cat][model_name]["conf"] = nan_to_num(
        model_data[cat][model_name]["conf"]
    ).tolist()
    model_metadata["confidences"] = model_data[cat][model_name]["conf"]
    model_metadata["accuracy"] = model_data[cat][model_name]["acc"]
    model_metadata["moe_model"] = moe_model
    with open(model_file, "w+") as fhandle:
        fhandle.write(dumps(model_metadata, indent=2))


# Add the metadata
for model in model_pool:
    add_metadata(model, moe_model=model in moe_model_pool)
