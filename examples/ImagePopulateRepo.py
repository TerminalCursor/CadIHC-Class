TOP_N = 2
IN_REPO_BASE = "dump.20260228/models/*.pth"
IN_REPO_BASE = "dump.full/models/image_vt*tn*.pth"
DATASETS_GLOB = "../Datasets/HT-*/"

from glob import glob
from histclass.extensions.glob_ext import globs
from datetime import datetime
from os.path import exists
from numpy import diag

from histclass.tools.configuration import load_config
from histclass.interfaces.config import CompleteConfig
from histclass.backends.pytorch.models.utils import load_model, upload_model
from histclass.interfaces.data.types import DataTypes, CategoryType
from histclass.interfaces.models.types import ModelType
from histclass.backends.pytorch.data.dataset import TorchDataset

def consensus(model, dataloaders, n_samples=100, n_rounds=1, bootstrapped=False):
    """# Get a consensus matrix"""
    from torch import zeros_like, float32, no_grad
    from numpy import array

    buckets = None
    with no_grad():
        for dataloader in dataloaders:
            n_iters = n_samples
            while True:
                # Bootstrapped means the dataloader only applies once
                for data, label in dataloader:
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
                    y_preds = y_pred.argmax(-1)
                    for y_true, y_pred in zip(y_trues, y_preds):
                        buckets[y_true][y_pred] += 1
                        if bootstrapped:
                            n_iters -= 1
                            if n_iters <= 0:
                                break
                    if (bootstrapped and (n_iters <= 0)):
                        break
                if (bootstrapped and (n_iters <= 0)):
                    break
    return array(buckets)

def parse_ctype(in_s):
    if in_s == "tn":
        return CategoryType.TripleNegative
    if in_s == "stage_basic":
        return CategoryType.BasicStage
    if in_s == "stage_advanced":
        return CategoryType.BasicStage
    if in_s == "grade":
        return CategoryType.Grade
    return CategoryType.Unknown

def parse_dtype(in_str):
    data_type = DataTypes.Unknown
    if in_str == "nuclei_only":
        data_type = DataTypes.Nuclei
    elif in_str == "nuclei":
        data_type = DataTypes.Morphology | DataTypes.Nuclei
    elif in_str == "metrics":
        data_type = DataTypes.Morphology
    elif in_str == "energetics":
        data_type = DataTypes.Energetics
    elif in_str == "latent":
        data_type = DataTypes.Latent
    elif in_str == "metricenergetics":
        data_type = DataTypes.Morphology | DataTypes.Energetics
    elif in_str == "image":
        data_type |= DataTypes.ImageCNN
    elif in_str == "image_vt":
        data_type |= DataTypes.ImageViT
    return data_type

def parse_log(log_file, chk_file):
    from json import loads, dumps
    from re import match
    data = ""
    with open(log_file, "r") as fhandle:
        data = fhandle.read()
    lines = data.split("\n")
    curl_lines = [ idx
            for idx in range(len(lines))
            if lines[idx] == "}"
            ]
    assert len(curl_lines) == 1
    json_data_s = '\n'.join(lines[:curl_lines[0]+1])
    json_data = loads(json_data_s)
    if "image" not in json_data["input_data_type"]:
        return None

    epoch_data = [ idx
            for idx in range(len(lines))
            if match(r"Epoch *[0-9]+:", lines[idx])
            ]
    last_epoch = lines[epoch_data[-1]]
    epoch = int(lines[epoch_data[-1]][len("Epoch"):].split(":")[0])

    loss_fn = json_data["loss_fn"]
    optim = "Adam"
    lr = json_data["learning_rate"]
    batch_size = json_data["batch_size"]

    data_type = parse_dtype(json_data["input_data_type"])
    category_type = parse_ctype(json_data["output_data_type"])
    model_type = ModelType.Attention

    patches = 1
    if data_type == DataTypes.ImageViT:
        patches = json_data["vision_transformer_config"]["image_size"]
    print(chk_file, patches)

    model_name = f"{data_type.to_str()}-{model_type.to_str()}2{category_type.to_str()}"

    config = load_config(data_type, category_type, model_type, patches=patches)
    config.batch_size = 64
    model = load_model(config, chk_file)
    model = model.to("cuda")
    print("Loaded model")
    data_sources = globs(DATASETS_GLOB)
    dataset = TorchDataset(data_sources, config, classic_norm=True)
    print("Loaded dataset")
    _,_,_,_,v_dataloaders = dataset.get_dataloaders()
    nc_val_acc = consensus(model, v_dataloaders, n_rounds=1, bootstrapped=True)
    nc_val_acc = float(diag(nc_val_acc).sum() / nc_val_acc.sum())
    print("Finished non-consensus accuracies")
    c_val_acc = consensus(model, v_dataloaders, n_rounds=9, bootstrapped=True)
    c_val_acc = float(diag(c_val_acc).sum() / c_val_acc.sum())
    print("Finished consensus accuracies")

    data_dict = {model_name: {
            "meta": {
                "data_type": data_type.value,
                "data_type_s": data_type.to_str(),
                "model_type": model_type.value,
                "model_type_s": model_type.to_str(),
                "category_type": category_type.value,
                "category_type_s": category_type.to_str(),
                "epochs": epoch,
                "batch_size": batch_size,
                "learning_rate": lr,
                "loss_fn": loss_fn,
                "optimizer": optim,
                "non-consensus": nc_val_acc,
                "consensus": c_val_acc,
                }
                }
            }
    if patches > 1:
        data_dict[model_name]["meta"]["patches"] = patches
    print(data_dict)
    return data_dict

model_data = {}
for file in glob(IN_REPO_BASE):
    timestamp = '-'.join(file.split("-")[1:])[:-len('.pth')]
    #time_format = "%Y%m%d-%H%M%S-%f"
    #log_file = f"dump.20260228/{timestamp}.txt"
    log_file = f"dump.full/{timestamp}.txt"
    if not exists(log_file):
        continue
    _dict = None
    try:
        _dict = parse_log(log_file, file)
    except:
        pass
    if _dict is None:
        continue
    print(file)
    _key = list(_dict.keys())[0]
    if _key not in model_data.keys():
        model_data[_key] = []
    _dict[_key]["name"] = file
    model_data[_key].append(_dict[_key])

from json import dumps

best_models = {
        "nc": {},
        "c": {},
        }

def sort_fn(a):
    va = a["meta"]["consensus"]
    return va
for key in model_data.keys():
    _sorted = sorted(model_data[key], key=sort_fn, reverse=True)
    best_models["c"][key] = _sorted[:TOP_N]
def sort_fn(a):
    va = a["meta"]["non-consensus"]
    return va
for key in model_data.keys():
    _sorted = sorted(model_data[key], key=sort_fn, reverse=True)
    best_models["nc"][key] = _sorted[:TOP_N]

for _type in best_models.keys():
    for _model in best_models[_type].keys():
        for data in best_models[_type][_model]:
            chk = data['name']
            dtype = DataTypes(data['meta']['data_type'])
            ctype = CategoryType(data['meta']['category_type'])
            mtype = ModelType(data['meta']['model_type'])
            #conf = CompleteConfig(dtype, ctype, mtype)
            patches = 1 if 'patches' not in data['meta'].keys() else data['meta']['patches']
            conf = load_config(dtype, ctype, mtype, patches=patches)
            model = load_model(conf, chk)
            upload_model(model, _model,
                    epochs=data['meta']['epochs'],
                    optimizer=data['meta']['epochs'],
                    loss_fn=data['meta']['epochs'],
                    batch_size=data['meta']['batch_size'],
                    lr=data['meta']['learning_rate'],
                    rest={
                        "use_classic": "yes",
                        "non-consensus": data['meta']['non-consensus'],
                        "consensus": data['meta']['consensus'],
                        "patches": patches,
                        },
                    to_remote = False,
                    )
