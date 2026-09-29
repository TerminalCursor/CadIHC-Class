def add_field(name, local_base="./", **kwargs):
    from json import load, dumps

    original_data = {}
    with open(
        f"{local_base}ihc/models/classification/{name}/metadata.json", "r"
    ) as fhandle:
        original_data = load(fhandle)
    for k, v in kwargs.items():
        original_data[k] = v
    with open(
        f"{local_base}ihc/models/classification/{name}/metadata.json", "w+"
    ) as fhandle:
        fhandle.write(dumps(original_data, indent=2))


def consensus(model, dataloaders, n_samples=100, n_rounds=9, bootstrapped=True):
    """# Get a consensus matrix"""
    from torch import zeros_like, float32, no_grad, softmax
    from numpy import array

    buckets = None
    raw_results = []
    names = []
    with no_grad():
        for dataloader in dataloaders:
            n_iters = n_samples
            while True:
                # Bootstrapped means the dataloader only applies once
                for data, label, name in dataloader:
                    # Assume that the labels are the same in the bootstrapped version
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
                    y_pred /= n_rounds
                    y_pred = softmax(y_pred, dim=-1)
                    for idx in range(y_true.shape[0]):
                        y_t = y_true[idx,...].detach().cpu().numpy()
                        y_p = y_pred[idx,...].detach().cpu().numpy()
                        _name = name[idx]
                        raw_results.append((*y_t, *y_p))
                        names.append([_name])
                    y_preds = y_pred.argmax(-1)
                    for y_true, y_pred in zip(y_trues, y_preds):
                        buckets[y_true][y_pred] += 1
                        if bootstrapped:
                            n_iters -= 1
                            if n_iters <= 0:
                                break
                    if bootstrapped and (n_iters <= 0):
                        break
                if bootstrapped and (n_iters <= 0):
                    break
    return array(buckets), array(raw_results), names


from histclass.extensions.glob_ext import globs
from histclass.backends.pytorch.utils import load_torch
from histclass.interfaces.data.types import DataframeType
import pandas as pd
from numpy import diag, nan_to_num
from tqdm.auto import tqdm

model_names = globs("ihc/models/classification/*.*")
model_names = [model_name.split("/")[-1] for model_name in model_names]
model_names = sorted(model_names, key=lambda x: x.split(".")[0].split("2")[0])

raw_data = []
pbar = tqdm(model_names)
for model_name in pbar:
    model, dataset = load_torch(model_name)
    _, _, _, t_dataloaders, v_dataloaders = dataset.get_dataloaders()
    val_acc, raw_results, names = consensus(model, t_dataloaders)
    for idx in range(len(raw_results)):
        res = [*raw_results[idx,...]]
        if len(res) == 4:
            res = (res[0],res[1],0,res[2],res[3],0.0)
        raw_data.append(
            (
                model_name,
                DataframeType.Training.value,
                model_name.split(".")[0].split("2")[0],
                model_name.split(".")[0].split("2")[1],
                model_name.split(".")[1],
                *names[idx],
                *res,
            )
        )
    tpr = diag(val_acc) / val_acc.sum(axis=0)
    tpr = tpr.flatten().tolist()
    tpr = nan_to_num(tpr)
    tpr = [float(_tpr) for _tpr in tpr]
    val_acc = diag(val_acc).sum() / val_acc.sum()
    #add_field(model_name, confidences=tpr)
    val_acc, raw_results, names = consensus(model, v_dataloaders)
    for idx in range(len(raw_results)):
        res = [*raw_results[idx,...]]
        if len(res) == 4:
            res = (res[0],res[1],0,res[2],res[3],0.0)
        raw_data.append(
            (
                model_name,
                DataframeType.Validation.value,
                model_name.split(".")[0].split("2")[0],
                model_name.split(".")[0].split("2")[1],
                model_name.split(".")[1],
                *names[idx],
                *res,
            )
        )
    val_acc = diag(val_acc).sum() / val_acc.sum()
    pbar.set_postfix(name=model_name.split(".")[0], acc=val_acc)
raw_data = pd.DataFrame(raw_data, columns=[
    "FullName", "Usage", "ModelName", "Category", "ModelId", "DataName",
    "Label-0", "Label-1", "Label-2",
    "Prediction-0", "Prediction-1", "Prediction-2",
    ])
raw_data.to_csv("MoeData.boot.csv")
raw_data.to_pickle("MoeData.boot.pickle")
print(raw_data)
