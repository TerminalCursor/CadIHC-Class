#!/usr/bin/env python3
# coding: utf-8
# pyright: basic

from histclass.tools.env import get_env
from histclass.interfaces.data.types import DataTypes, CategoryType
from histclass.interfaces.models.types import ModelType
from matplotlib import pyplot as plt

# Global Parameters
CHECKING_COUNT = False
LOG_LOCATION = get_env("DUMP_LOC").catch("dump/").unwrap()


def generate_configuration_combination(
    model_type_list,
    data_type_list,
    category_type_list,
    patch_list,
    num_ablations,
):
    model_type_configurations = model_type_list
    category_type_configurations = category_type_list
    data_type_configurations = []
    patch_configurations = []
    ablation_configurations = []
    for data_type in data_type_list:
        if data_type == DataTypes.ImageViT:
            for patch in patch_list:
                data_type_configurations.append(data_type)
                patch_configurations.append(patch)
                ablation_configurations.append([])
        elif data_type == DataTypes.ImageCNN:
            data_type_configurations.append(data_type)
            patch_configurations.append(None)
            ablation_configurations.append([])
        else:
            from itertools import combinations

            columns = data_type.dimensionality()
            ablations = combinations(range(columns), num_ablations)
            for ablation in ablations:
                data_type_configurations.append(data_type)
                patch_configurations.append(None)
                the_ablation = list(ablation)
                ablation_configurations.append(the_ablation)
    return (
        model_type_configurations,
        category_type_configurations,
        (data_type_configurations, patch_configurations, ablation_configurations),
    )

MODEL_TYPES = [
    ModelType.Attention,
    # ModelType.Recurrent,
]
DATA_TYPES = [
    DataTypes.ImageViT,
    DataTypes.ImageCNN,
    DataTypes.Morphology,
    DataTypes.Morphology | DataTypes.Nuclei,
    DataTypes.Morphology | DataTypes.Energetics,
    DataTypes.Latent,
    DataTypes.Nuclei,
    DataTypes.Energetics,
]
CATEGORY_TYPES = [
    CategoryType.BinaryStage,
    CategoryType.BinaryGrade,
    CategoryType.TripleNegative,
    #CategoryType.BasicStage,
    #CategoryType.Grade,
]
PATCHES = [128, 256, 512]

M_COMBOS, C_COMBOS, D_COMBOS = generate_configuration_combination(
    MODEL_TYPES, DATA_TYPES, CATEGORY_TYPES, PATCHES, num_ablations=0
)
DN_COMBOS = range(len(D_COMBOS[0]))

from histclass.tools.slurm import get_sat_id
from itertools import product

NUMBER_ITERATIONS = 10
training_combinations = list(
    product(range(NUMBER_ITERATIONS), M_COMBOS, DN_COMBOS, C_COMBOS)
)
print(len(training_combinations))
assert not CHECKING_COUNT

# Slurm Array Task ID
SATID = get_sat_id()
if SATID is not None:
    training_combinations = [training_combinations[SATID]]
else:
    training_combinations = list(training_combinations)

_, MODEL_TYPE, DN_IDX, CATEGORY_TYPE = training_combinations[0]
DATA_TYPE = D_COMBOS[0][DN_IDX]
PATCH = D_COMBOS[1][DN_IDX]
REMOVE_COLUMNS = D_COMBOS[2][DN_IDX]

num_columns = DATA_TYPE.dimensionality()
for v in REMOVE_COLUMNS:
    assert v < num_columns, "Column does not exist"

from histclass.tools.configuration import load_config
from histclass.tools.training.lib import early_exit_detector
from histclass.backends.pytorch.losses import get_loss
from histclass.backends.pytorch.optimizers import get_optimizer
from histclass.backends.pytorch.models.utils import get_model
from histclass.tools.logger import Logger
from histclass.backends.pytorch.data.dataset import load_data
from tqdm.auto import tqdm
from numpy import array

logger = Logger(LOG_LOCATION)

# Configuration
config = load_config(DATA_TYPE, CATEGORY_TYPE, MODEL_TYPE, patches=PATCH)
#config.loss_fn = "cross_entropy"
#config.loss_fn = "mse_loss"
config.loss_fn = "focal_loss"
config.optimizer = "AdamW"
config.batch_size = 32
config.epochs = 50_000
#config.learning_rate = 5e-6
config.learning_rate = 1e-5
config.early_exit_config.kwargs["lookback"] = 100_000
if config.pointcloud_config is not None:
    config.pointcloud_config.n_blocks = 2
loss_fn = get_loss(config).unwrap()
optim = get_optimizer(config).unwrap()

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
breakpoint()
dataset = load_data(config)
breakpoint()
assert False

# Load Model
model = get_model(config).to("cuda")
model.set_optimizer(optim)
model.optimizer = model.optimizer(model.model.parameters(), lr=config.learning_rate)
print(model.model)

dataloaders = dataset.get_dataloaders()
t_dataloader = dataloaders[0]
v_dataloader = dataloaders[1]
T_dataloader = dataloaders[2]
t_class_dataloaders = dataloaders[3]
v_class_dataloaders = dataloaders[4]

# Metrics to save
epoch_metrics = ([], [], [], [])

# Pretraining statistics
epoch = 0
t_loss, t_acc, t_mat = model.eval_epoch(t_dataloader, loss_fn)
v_loss, v_acc, v_mat = model.eval_epoch(v_dataloader, loss_fn)
epoch_metrics[0].append(epoch)
epoch_metrics[1].append((t_loss, v_loss))
epoch_metrics[2].append((t_acc, v_acc))
epoch_metrics[3].append((t_mat, v_mat))

early_exit_detections = 0
logger.echo_stdout = False
dataloaderss = [*t_class_dataloaders, t_dataloader]
for epoch in tqdm(range(config.epochs)):
    t_loss, t_acc, t_mat = model.train_epoch(t_dataloader, loss_fn)
    idx = int(epoch / 1_000)
    idx = idx % len(dataloaderss)
    #t_loss, t_acc, t_mat = model.train_epoch(dataloaderss[idx], loss_fn)
    v_loss, v_acc, v_mat = model.eval_epoch(v_dataloader, loss_fn)
    epoch_metrics[0].append(epoch + 1)
    epoch_metrics[1].append((t_loss, v_loss))
    epoch_metrics[2].append((t_acc, v_acc))
    epoch_metrics[3].append((t_mat, v_mat))

    logger.addnl(
        f"{epoch_metrics[0][-1]:>5}: {t_loss:7.4f} {v_loss:7.4f}  {t_acc*100:6.2f} {v_acc*100:6.2f}"
    )

    eed, slope = early_exit_detector(
        epoch_metrics[1], **config.early_exit_config.kwargs  # pyright: ignore
    )
    if eed:
        early_exit_detections += 1
    else:
        early_exit_detections = 0
    if early_exit_detections > config.early_exit_config.max_failures:
        break
logger.echo_stdout = True

epochs = epoch_metrics[0]
t_losses, v_losses = zip(*epoch_metrics[1])
plt.clf()
plt.plot(epochs, v_losses)
plt.plot(epochs, t_losses, alpha=0.5)
plt.savefig("outputs/losses.png")
t_losses, v_losses = zip(*epoch_metrics[2])
plt.clf()
plt.plot(epochs, v_losses)
plt.plot(epochs, t_losses, alpha=0.5)
plt.savefig("outputs/accuracies.png")

# Posttraining Metrics
epoch = epoch + 1
t_loss, t_acc, t_mat = model.eval_epoch(t_dataloader, loss_fn)
v_loss, v_acc, v_mat = model.eval_epoch(v_dataloader, loss_fn)
T_loss, T_acc, T_mat = model.eval_epoch(T_dataloader, loss_fn)
epoch_metrics[0].append(epoch)
epoch_metrics[1].append((t_loss, v_loss))
epoch_metrics[2].append((t_acc, v_acc))
epoch_metrics[3].append((t_mat, v_mat))

## logger.db_add(
##     total_accuracy=T_acc,
##     training_accuracy=t_acc,
##     validation_accuracy=v_acc,
##     total_raw=T_mat.tolist(),
##     training_raw=t_mat.tolist(),
##     validation_raw=v_mat.tolist(),
## ).db_write()
epochs = array(epoch_metrics[0])
losses = array(epoch_metrics[1])
accuracies = array(epoch_metrics[2])
epoch_matrices = array(epoch_metrics[3])

# from matplotlib import pyplot as plt
#
# plt.clf()
# plt.plot(epochs, losses)
# plt.savefig("outputs/losses.png", dpi=600)
# plt.clf()
# plt.plot(
#    epochs[-config.early_exit_config.kwargs["lookback"] - 1 : -1],
#    losses[-config.early_exit_config.kwargs["lookback"] - 1 : -1, :],
# )
# plt.savefig("outputs/last_losses.png", dpi=600)
# plt.clf()
# plt.plot(epochs, accuracies)
# plt.savefig("outputs/accuracies.png", dpi=600)

logger.addnl(str(epochs))
logger.addnl(str(losses))
logger.addnl(str(accuracies))
logger.addnl(str(epoch_matrices))

from numpy import diag


def consensus(model, dataloaders, n_samples=1000, n_rounds=1, bootstrapped=False):
    """# Get a consensus matrix"""
    from torch import zeros_like, float32, no_grad
    from numpy import array

    buckets = None
    with no_grad():
        for _ in range(n_samples):
            for dataloader in dataloaders:
                # Bootstrapped means the dataloader only applies once
                for data, label, _ in dataloader:
                    if buckets == None:
                        n_class = label.shape[-1]
                        buckets = [[0 for _ in range(n_class)] for _ in range(n_class)]
                    data = data.to(model.device)
                    y_true = label.to(model.device)
                    y_trues = y_true.argmax(-1)
                    y_pred = zeros_like(y_true, dtype=float32)
                    # Smooth over n_rounds for consensus: Max Likelyhood
                    for _ in range(n_rounds):
                        y_pred += model.forward(data)
                    y_preds = y_pred.argmax(-1)
                    for y_true, y_pred in zip(y_trues, y_preds):
                        buckets[y_true][y_pred] += 1
                        if bootstrapped:
                            break
                    if bootstrapped:
                        break
    return array(buckets)


buckets = array(
    [
        consensus(model, dataloaders, n_samples=100, n_rounds=1, bootstrapped=False)
        for dataloaders in [[T_dataloader], [t_dataloader], [v_dataloader]]
    ]
)
print(buckets.shape)

# Getting TPR, TNR ish result
[
    logger.addnl(str(diag(buckets[idx, :]) / buckets[idx, :].sum(axis=0)))
    for idx in range(3)
]

buckets = array(
    [
        consensus(model, dataloaders, n_samples=1000, n_rounds=1, bootstrapped=False)
        for dataloaders in [[T_dataloader], [t_dataloader], [v_dataloader]]
    ]
)
print(buckets.shape)
logger.addnl(str(buckets))
# Getting accuracies from each
[
    logger.addnl(str(diag(buckets[idx, ...]).sum() / buckets[idx, ...].sum()))
    for idx in range(2)
]

logger.addnl("Bootstrapped Accuracies")
buckets = array(
    [
        consensus(model, dataloaders, n_samples=1000, n_rounds=1, bootstrapped=True)
        for dataloaders in [t_class_dataloaders, v_class_dataloaders]
    ]
)
print(buckets.shape)
logger.addnl(str(buckets))
# Getting accuracies from each
[
    logger.addnl(str(diag(buckets[idx, ...]).sum() / buckets[idx, ...].sum()))
    for idx in range(2)
]
t_acc, v_acc = [
    float(diag(buckets[idx, ...]).sum() / buckets[idx, ...].sum())
    for idx in range(2)
]
logger.db_add(
    training_accuracy=t_acc,
    validation_accuracy=v_acc,
)

logger.addnl("Bootstrapped Consensus Accuracies")
buckets = array(
    [
        consensus(model, dataloaders, n_samples=1000, n_rounds=9, bootstrapped=True)
        for dataloaders in [t_class_dataloaders, v_class_dataloaders]
    ]
)
print(buckets.shape)
logger.addnl(str(buckets))
# Getting accuracies from each
[
    logger.addnl(str(diag(buckets[idx, ...]).sum() / buckets[idx, ...].sum()))
    for idx in range(2)
]
t_acc, v_acc = [
    float(diag(buckets[idx, ...]).sum() / buckets[idx, ...].sum())
    for idx in range(2)
]
logger.db_add(
    training_accuracy_c=t_acc,
    validation_accuracy_c=v_acc,
)

logger.db_write()
model.save(f"{LOG_LOCATION}/models/{config.model_name}-{logger.timestamp}.pth")
# from histclass.backends.pytorch.models.utils import upload_model
# upload_model(model, f"{config.model_name}")
