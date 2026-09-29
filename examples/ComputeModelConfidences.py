#!/usr/bin/env python3
# coding: utf-8
# pyright: basic

# Load Model
from histclass.backends.pytorch.models.utils import download_model_with_config
from histclass.backends.pytorch.data.dataset import load_data
from numpy import array, diag
from numpy import all as npall
from numpy import isfinite as npisfinite
from json import dumps, loads
from glob import glob

for _dir in sorted(glob("ihc/models/classification/*")):
    #name = "energetics-attention2hilograde.ed117b364528f83a6a8035e224265f18cf15a408"
    print(_dir)
    name = _dir.split("/")[-1]
    model, config = download_model_with_config(
        name
    )
    model.model = model.model.eval()

    metadata_json = {}
    with open(f"ihc/models/classification/{name}/metadata.json", "r") as fhandle:
        metadata_json = loads(fhandle.read())
    if "confidences" in metadata_json.keys():
        continue

    # Load Data
    dataset = load_data(config)
    dataloaders = dataset.get_dataloaders()
    t_dataloader = dataloaders[0]
    v_dataloader = dataloaders[1]
    T_dataloader = dataloaders[2]
    t_class_dataloaders = dataloaders[3]
    v_class_dataloaders = dataloaders[4]

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


    best = 0
    tpr = [0,0]
    tpr_last = None
    for _ in range(9):
        buckets = array(
            [
                consensus(model, dataloaders, n_samples=100, n_rounds=9, bootstrapped=True)
                for dataloaders in [v_class_dataloaders]
            ]
        )


        b = buckets[0,...]
        #buckets = array(
        #    [
        #        consensus(model, dataloaders, n_samples=100, n_rounds=9, bootstrapped=True)
        #        for dataloaders in [t_class_dataloaders, v_class_dataloaders]
        #    ]
        #)


        #b = buckets[0,...]
        #b = buckets[1,...]
        acc = diag(b).sum() / b.sum()
        _tpr = (diag(b)/b.sum(axis=0))
        tpr_last = _tpr
        if (acc > best) and (npall(npisfinite(_tpr))):
            tpr = _tpr.tolist()
            best = acc
    if (tpr[0] == 0) and (tpr[1] == 0):
        tpr = tpr_last.tolist()
        tpr[0] = tpr[0] if npisfinite(tpr[0]) else 0
        tpr[1] = tpr[1] if npisfinite(tpr[1]) else 0
    metadata_json["confidences"] = tpr

    print(name)
    print(best)
    print(tpr)
    #print(dumps(metadata_json, indent=2))
    with open(f"ihc/models/classification/{name}/metadata.json", "w+") as fhandle:
        fhandle.write(dumps(metadata_json, indent=2))
