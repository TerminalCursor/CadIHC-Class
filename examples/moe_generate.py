from histclass.backends.pytorch.utils import load_torch
from histclass.backends.pytorch.data.dataset import PointcloudDataframe, ImageDataframe
from histclass.extensions.glob_ext import globs
from tqdm.auto import tqdm
from torch import Tensor, no_grad
from torch import softmax
from numpy import array
import pandas as pd
from os.path import exists

# Get model list
##folder_list = globs("ihc/models/classification/*.*")
#folder_list = globs("ihc/models/classification/*hilo*.*")
folder_list = globs("ihc/models/classification/*hilo*.*") + globs("ihc/models/classification/*tn*.*")
folder_list = [folder.split("/")[-1] for folder in folder_list]
folder_list = sorted(folder_list, key=lambda x: x.split(".")[0].split("2")[0])
# folder_list = sorted(folder_list, key=lambda x: x.split(".")[0].split("2")[1])
model_list = folder_list
model_types = set(map(lambda x: x.split(".")[0], model_list))
# print(model_list)
# print(model_types)

n_rounds = 100

moe_data = []

with no_grad():
    pbar = tqdm(model_list)
    for model_name in pbar:
        pbar.set_description(model_name.split(".")[0])
        model, dataset = load_torch(model_name)
        model.model.eval()

        for frame in dataset.data_frames:
            if not (
                isinstance(frame, PointcloudDataframe)
                or isinstance(frame, ImageDataframe)
            ):
                continue
            usage = frame.usage
            name = frame.name
            for _ in range(n_rounds):
                data, label, _ = frame[0]
                data = data.reshape(1, *data.shape).to(model.device)
                label = label.reshape(1, *label.shape).to(model.device)
                out = model.forward(data)
                out = softmax(out, dim=-1).flatten().detach().cpu().numpy()
                odata = [*out]
                label = label.flatten().detach().cpu().numpy()
                ldata = [*label]
                if len(ldata) == 2:
                    ldata.append(0)
                if len(odata) == 2:
                    odata.append(0.0)
                moe_data.append(
                    (
                        model_name.split(".")[0].split("2")[0],
                        model_name.split(".")[0].split("2")[1],
                        model_name.split(".")[1],
                        name,
                        usage.value,
                        label.argmax(-1),
                        out.argmax(-1),
                        *ldata,
                        *odata,
                    )
                )
columns = [
    "ModelName",
    "Category",
    "ModelId",
    "DataName",
    "Usage",
    "Label",
    "Prediction",
]
base_column_count = len(columns)
for index in range((len(moe_data[0]) - base_column_count) // 2):
    columns.append(f"Label-{index}")
for index in range((len(moe_data[0]) - base_column_count) // 2):
    columns.append(f"Prediction-{index}")
moe_data = pd.DataFrame(moe_data, columns=columns)
file_name = "MoeData.boot"
if exists(f"{file_name}.pickle"):
    old_moe_data = pd.read_pickle(f"{file_name}.pickle")
    if not exists(f".{file_name}.pickle.old"):
        old_moe_data.to_pickle(f".{file_name}.pickle.old")
    else:
        exit(1)
    moe_data = pd.concat([moe_data, old_moe_data], ignore_index=True)
moe_data.to_csv(f"{file_name}.csv")
moe_data.to_pickle(f"{file_name}.pickle")
