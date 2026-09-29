#!/usr/bin/env python3
from glob import glob
from histclass.tools.env import get_env
from histclass.extensions.sqlite3_ext import legacy_sql_load as sql_load
from histclass.interfaces.config import CompleteConfig
from histclass.backends.pytorch.models.utils import load_model
from histclass.backends.pytorch.data.dataset import load_data
from os.path import exists
from json import loads

def matrix2acc(matrix):
    from numpy import diag
    return diag(matrix).sum() / matrix.sum()

# Compute consensus on data: bootstrapped. Use the triple positive rate/ triple negative rate for confidence
def consensus(model, dataloaders, n_samples=1000, n_rounds=1, bootstrapped=False):
    """# Get a consensus matrix"""
    from torch import zeros_like, float32, no_grad
    from numpy import array
    from tqdm.auto import tqdm

    buckets = None
    with no_grad():
        for _ in tqdm(range(n_samples)):
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

INDEX = 2129
DATA_LOCAL_BASE = "./"

BASE = get_env("DATA_BASE")
BASE = BASE.catch("dump.hilo/").unwrap()
DB_LOC = f"{BASE}/Results.db"

results_df = sql_load(DB_LOC, "RESULTS")
print(results_df.shape)
model_data = results_df.iloc[INDEX]
debug_file = f"{BASE}/{model_data['TIMESTAMP']}.txt"
model_file = glob(f"{BASE}/models/*{model_data['TIMESTAMP']}*.pth")
assert len(model_file) == 1
model_file = model_file[0]

print(model_data)

assert exists(debug_file), "Debug File missing"
assert exists(model_file), "Model File missing"

with open(debug_file, "r") as fhandle:
    debug_info = fhandle.read().split("\n}")[0] + "\n}"
model_json = loads(debug_info)
print(model_json)
try:
    model_config = CompleteConfig.from_json(model_json)
    model = load_model(model_config, model_file)

    dataset = load_data(
        model_config, local_base=DATA_LOCAL_BASE, classic_norm=model.classic_data_norm
    )

    data_loaders = dataset.get_dataloaders()
    tdl,vdl,Tdl,tcdl,vcdl = data_loaders

    matrix = consensus(model, vcdl, n_samples=100, n_rounds=1, bootstrapped=True)
    print(matrix)
    print(matrix2acc(matrix))
    matrix = consensus(model, vcdl, n_samples=100, n_rounds=9, bootstrapped=True)
    print(matrix)
    print(matrix2acc(matrix))

    print(model_config.output_data_type)
    print(dataset)
except Exception as e:
    print(f"Error: {e}")
