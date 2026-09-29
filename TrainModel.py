#!/usr/bin/env python3
# coding: utf-8

from lib.interfaces.data.types import DataTypes, CategoryType, DataframeType
from lib.interfaces.models.types import ModelType

# Global Parameters
DATASETS_GLOB = "../Datasets/HT-*/"
MODEL_TYPE = ModelType.Recurrent
DATA_TYPE = DataTypes.Nuclei | DataTypes.Morphology
# DATA_TYPE = DataTypes.ImageViT
CATEGORY_TYPE = CategoryType.Grade
#CATEGORY_TYPE = CategoryType.TripleNegative
PATCH = 128
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
from lib.tools.training.lib import early_exit_detector
from lib.backends.pytorch.losses import get_loss
from lib.backends.pytorch.optimizers import get_optimizer
from lib.backends.pytorch.models.utils import load_model
from lib.tools.logger import Logger
from lib.interfaces.monad.maybe import Maybe
from lib.backends.pytorch.data.dataset import TorchDataset
from tqdm.auto import tqdm
from numpy import array

logger = Logger(LOG_LOCATION)

# Configuration
config = load_config(DATA_TYPE, CATEGORY_TYPE, MODEL_TYPE, patches=PATCH)
# config.loss_fn = "cross_entropy"
config.loss_fn = "mse_loss"
config.loss_fn = "focal_loss"
config.optimizer = "Adam"
config.batch_size = 64
config.learning_rate = 1e-5
config.early_exit_config.kwargs["lookback"] = 1_500
loss_fn = get_loss(config)

# Record Configuration Used
logger.addnl(config.to_dict())
logger.db_add(
    name=config.model_name,
    category=config.output_data_type.to_str(),
    learning_rate=config.learning_rate,
    loss=config.loss_fn,
)

# Load Data
## Get Data Sources
data_sources = globs(DATASETS_GLOB)
dataset = TorchDataset(data_sources, config)

# Load Model
model = load_model(dataset, config).to("cuda")
model.set_optimizer(Maybe(get_optimizer(config)).unwrap())
model.optimizer = model.optimizer(model.model.parameters(), lr=config.learning_rate)
print(model.model)

t_dataloader, v_dataloader, T_dataloader = dataset.get_dataloaders()

# Metrics to save
epoch_metrics = [[], [], [], []]

# Pretraining statistics
epoch = 0
t_loss, t_acc, t_mat = model.eval_batch(t_dataloader, loss_fn)
v_loss, v_acc, v_mat = model.eval_batch(v_dataloader, loss_fn)
epoch_metrics[0].append(epoch)
epoch_metrics[1].append((t_loss, v_loss))
epoch_metrics[2].append((t_acc, v_acc))
epoch_metrics[3].append((t_mat, v_mat))

early_exit_detections = 0
logger.echo_stdout = False
for epoch in tqdm(range(config.epochs)):
    t_loss, t_acc, t_mat = model.train_batch(t_dataloader, loss_fn)
    v_loss, v_acc, v_mat = model.eval_batch(v_dataloader, loss_fn)
    epoch_metrics[0].append(epoch + 1)
    epoch_metrics[1].append((t_loss, v_loss))
    epoch_metrics[2].append((t_acc, v_acc))
    epoch_metrics[3].append((t_mat, v_mat))

    logger.addnl(
        f"{epoch_metrics[0][-1]:>5}: {t_loss:7.4f} {v_loss:7.4f}  {t_acc*100:6.2f} {v_acc*100:6.2f}"
    )

    eed, slope = early_exit_detector(
        epoch_metrics[1], **config.early_exit_config.kwargs
    )
    if eed:
        early_exit_detections += 1
    else:
        early_exit_detections = 0
    if early_exit_detections > config.early_exit_config.max_failures:
        break
logger.echo_stdout = True

# Posttraining Metrics
epoch = epoch + 1
t_loss, t_acc, t_mat = model.eval_batch(t_dataloader, loss_fn)
v_loss, v_acc, v_mat = model.eval_batch(v_dataloader, loss_fn)
T_loss, T_acc, T_mat = model.eval_batch(T_dataloader, loss_fn)
epoch_metrics[0].append(epoch)
epoch_metrics[1].append((t_loss, v_loss))
epoch_metrics[2].append((t_acc, v_acc))
epoch_metrics[3].append((t_mat, v_mat))

logger.db_add(
    total_accuracy=T_acc,
    training_accuracy=t_acc,
    validation_accuracy=v_acc,
    total_raw=T_mat.tolist(),
    training_raw=t_mat.tolist(),
    validation_raw=v_mat.tolist(),
).db_write()
epochs = array(epoch_metrics[0])
losses = array(epoch_metrics[1])
accuracies = array(epoch_metrics[2])
epoch_matrices = array(epoch_metrics[3])

from matplotlib import pyplot as plt

plt.clf()
plt.plot(epochs, losses)
plt.savefig("outputs/losses.png", dpi=600)
plt.clf()
plt.plot(
    epochs[-config.early_exit_config.kwargs["lookback"] - 1 : -1],
    losses[-config.early_exit_config.kwargs["lookback"] - 1 : -1, :],
)
plt.savefig("outputs/last_losses.png", dpi=600)
plt.clf()
plt.plot(epochs, accuracies)
plt.savefig("outputs/accuracies.png", dpi=600)

logger.addnl(str(epochs))
logger.addnl(str(losses))
logger.addnl(str(accuracies))
logger.addnl(str(epoch_matrices))

from numpy import zeros
from torch.utils.data import Dataset

buckets = zeros((3, epoch_matrices.shape[-1], epoch_matrices.shape[-1]))
for _ in range(100):
    for data_frame in dataset.data_frames:
        usage = data_frame.usage
        assert isinstance(data_frame, Dataset)
        data, y_true = data_frame[0]
        data = data.to(model.device)
        y_true = y_true.to(model.device)
        y_true = y_true.argmax(-1).item()
        y_pred = model.forward(data[None, ...])
        y_pred = y_pred.argmax(-1).item()
        buckets[0][y_true, y_pred] += 1
        if usage == DataframeType.Training:
            buckets[1][y_true, y_pred] += 1
        elif usage == DataframeType.Validation:
            buckets[2][y_true, y_pred] += 1
logger.addnl(str(buckets))

buckets = zeros((3, epoch_matrices.shape[-1], epoch_matrices.shape[-1]))
for _ in range(100):
    for data, label in T_dataloader:
        data = data.to(model.device)
        y_true = label.to(model.device)
        y_trues = y_true.argmax(-1)
        y_pred = model.forward(data)
        y_preds = y_pred.argmax(-1)
        for y_true, y_pred in zip(y_trues, y_preds):
            buckets[0][y_true, y_pred] += 1
    for data, label in t_dataloader:
        data = data.to(model.device)
        y_true = label.to(model.device)
        y_trues = y_true.argmax(-1)
        y_pred = model.forward(data)
        y_preds = y_pred.argmax(-1)
        for y_true, y_pred in zip(y_trues, y_preds):
            buckets[1][y_true, y_pred] += 1
    for data, label in v_dataloader:
        data = data.to(model.device)
        y_true = label.to(model.device)
        y_trues = y_true.argmax(-1)
        y_pred = model.forward(data)
        y_preds = y_pred.argmax(-1)
        for y_true, y_pred in zip(y_trues, y_preds):
            buckets[2][y_true, y_pred] += 1
logger.addnl(str(buckets))

from numpy import diag

[
    logger.addnl(str(diag(buckets[idx, :]) / buckets[idx, :].sum(axis=0)))
    for idx in range(3)
]

model.save(
    f"{LOG_LOCATION}/models/{config.model_name}-{config.model_type.to_str()}-{logger.timestamp}.pth"
)
