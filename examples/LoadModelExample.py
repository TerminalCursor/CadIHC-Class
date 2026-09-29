#!/usr/bin/env python3
# coding: utf-8

from histclass.interfaces.data.types import DataTypes, CategoryType
from histclass.interfaces.models.types import ModelType

# Global Parameters
DATASETS_GLOB = "../Datasets/HT-*/"
MODEL_TYPE = ModelType.Attention
DATA_TYPE = DataTypes.Nuclei | DataTypes.Morphology
# DATA_TYPE = DataTypes.ImageViT
# CATEGORY_TYPE = CategoryType.Grade
CATEGORY_TYPE = CategoryType.TripleNegative
PATCH = 128
LOG_LOCATION = "dump/"

# SLURM ID For Array Tasks
from histclass.extensions.glob_ext import globs
from histclass.tools.configuration import load_config
from histclass.backends.pytorch.models.utils import load_model
from histclass.backends.pytorch.data.dataset import TorchDataset

# Configuration
config = load_config(DATA_TYPE, CATEGORY_TYPE, MODEL_TYPE, patches=PATCH)

# Load Data
## Get Data Sources
data_sources = globs(DATASETS_GLOB)
dataset = TorchDataset(data_sources, config)

# Load Model
model = load_model(
    config,
    "dump.migrated/models/morphologynuclei_classifier_tn-20260217-181845-490846.pth",
).to("cuda")
print(model.model)
print(model)

from histclass.backends.pytorch.models.utils import upload_model

upload_model(
    model=model,
    name="MN2TN_Classifier",
)
