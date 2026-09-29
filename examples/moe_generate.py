from histclass.backends.pytorch.utils import load_torch
from histclass.backends.pytorch.data.dataset import PointcloudDataframe, ImageDataframe
from histclass.extensions.glob_ext import globs
from tqdm.auto import tqdm
from torch import Tensor, no_grad
from torch import softmax
from numpy import array
import pandas as pd

# Get model list
folder_list = globs("ihc/models/classification/*.*")
folder_list = [folder.split("/")[-1] for folder in folder_list]
folder_list = sorted(folder_list, key=lambda x: x.split(".")[0].split("2")[0])
# folder_list = sorted(folder_list, key=lambda x: x.split(".")[0].split("2")[1])
model_list = folder_list
model_types = set(map(lambda x: x.split(".")[0], model_list))
# print(model_list)
# print(model_types)

n_rounds = 10

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
                data, label = frame[0]
                data = data.reshape(1, *data.shape).to(model.device)
                label = label.reshape(1, *label.shape).to(model.device)
                out = model.forward(data)
                out = softmax(out, dim=-1).flatten().detach().cpu().numpy()
                label = label.flatten().detach().cpu().numpy()
                moe_data.append(
                    (
                        model_name,
                        name,
                        usage.value,
                        label.argmax(-1),
                        out.argmax(-1),
                        *label,
                        *out,
                    )
                )
columns = ["Model Name", "Data Name", "Usage", "Label", "Prediction"]
base_column_count = len(columns)
for index in range((len(moe_data[0]) - base_column_count) // 2):
    columns.append(f"Label-{index}")
for index in range((len(moe_data[0]) - base_column_count) // 2):
    columns.append(f"Prediction-{index}")
moe_data = pd.DataFrame(moe_data, columns=columns)
moe_data.to_csv("MoeData.csv")
moe_data.to_pickle("MoeData.pickle")
