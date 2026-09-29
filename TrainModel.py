#!/usr/bin/env python3
# coding: utf-8

from lib.interfaces.data.types import DataTypes, CategoryTypes

# Global Parameters
DATASETS_GLOB = "../Datasets/HT-*/"
DATA_TYPE = [DataTypes.Nuclei]
CATEGORY_TYPE = CategoryTypes.TripleNegative
PATCH = 1
LOG_LOCATION = "dump/"

# from lib.tools.slurm import get_sat_id
# from itertools import product
# NUMBER_ITERATIONS = 1
# training_combinations = list(
#    product(range(NUMBER_ITERATIONS), DATA_TYPES, CATEGORY_TYPES, PATCHES)
# )
## Slurm Array Task ID
# SATID = get_sat_id()
# if SATID is not None:
#    training_combinations = [training_combinations[SATID]]

# SLURM ID For Array Tasks
from lib.extensions.glob_ext import globs
from lib.tools.configuration import load_config
from lib.backends.pytorch.losses import get_loss
from lib.interfaces.logger import Logger

logger = Logger(LOG_LOCATION)

# Configuration
config = load_config(DATA_TYPE, CATEGORY_TYPE, PATCH)
loss_fn = get_loss(config.loss_fn)

# Record Configuration Used
logger.addnl(config.to_dict())
logger.db_set_meta(
    name=config.model_name,
    category=config.output_data_type.to_str(),
    lr=config.learning_rate,
    loss=config.loss_fn,
)

# Load Data
## Get Data Sources
data_sources = globs(DATASETS_GLOB)
dataset = get_dataset(data_sources, config)
