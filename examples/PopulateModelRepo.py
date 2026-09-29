TOP_N = 2
IN_REPO_BASE = "dump/models/*.pth"

from glob import glob
from datetime import datetime
from os.path import exists

from histclass.interfaces.data.types import DataTypes, CategoryType
from histclass.interfaces.models.types import ModelType

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
    if ("latent") in in_str:
        data_type |= DataTypes.Latent
    if ("morphology") in in_str:
        data_type |= DataTypes.Morphology
    if ("energetics") in in_str:
        data_type |= DataTypes.Energetics
    if ("nuclei") in in_str:
        data_type |= DataTypes.Nuclei
    if in_str == "image":
        data_type |= DataTypes.ImageCNN
    if in_str == "image_vt":
        data_type |= DataTypes.ImageViT
    return data_type

def parse_log(log_file):
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

    epoch_data = [ idx
            for idx in range(len(lines))
            if match(r" *[0-9]+:", lines[idx])
            ]
    last_epoch = lines[epoch_data[-1]]
    epoch = int(lines[epoch_data[-1]].split(":")[0])

    loss_fn = json_data["loss_fn"]
    optim = "Adam"
    lr = json_data["learning_rate"]
    batch_size = json_data["batch_size"]
    model_name = json_data["model_name"]

    final = [
            idx for idx in range(len(lines))
            if lines[idx] == "Bootstrapped Accuracies"
            ]
    assert len(final) == 1
    fin_lines = lines[final[0]:]
    nc_lines = fin_lines[:len(fin_lines)//2]
    c_lines = fin_lines[len(fin_lines)//2:]
    assert nc_lines[0].startswith("Bootstrapped")
    assert c_lines[0].startswith("Bootstrapped")
    data_lines = [idx for idx in range(len(nc_lines))
            if match(r"[0-1]", nc_lines[idx])]
    nc_val_acc = float(nc_lines[data_lines[-1]])
    data_lines = [idx for idx in range(len(c_lines))
            if match(r"[0-1]", c_lines[idx])]
    c_val_acc = float(c_lines[data_lines[-1]])

    data_type_s = model_name.split("2")[0].split("-")[0]
    model_type_s = model_name.split("2")[0].split("-")[1]
    category_type_s = model_name.split("2")[1]

    data_dict = {model_name: {
            "meta": {
                "data_type": parse_dtype(data_type_s).value,
                "data_type_s": data_type_s,
                "model_type": ModelType.Attention.value,
                "model_type_s": model_type_s,
                "category_type_s": category_type_s,
                "category_type": parse_ctype(category_type_s).value,
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
    return data_dict

model_data = {}
for file in glob(IN_REPO_BASE):
    timestamp = '-'.join(file.split("-")[2:])[:-len('.pth')]
    time_format = "%Y%m%d-%H%M%S-%f"
    assert exists(f"dump/{timestamp}.txt")
    _dict = parse_log(f"dump/{timestamp}.txt")
    _key = list(_dict.keys())[0]
    if _key not in model_data.keys():
        model_data[_key] = []
    _dict[_key]["name"] = file
    model_data[_key].append(_dict[_key])
print(model_data)

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

from histclass.interfaces.config import CompleteConfig
from histclass.backends.pytorch.models.utils import load_model, upload_model
for _type in best_models.keys():
    for _model in best_models[_type].keys():
        for data in best_models[_type][_model]:
            chk = data['name']
            dtype = DataTypes(data['meta']['data_type'])
            ctype = CategoryType(data['meta']['category_type'])
            mtype = ModelType(data['meta']['model_type'])
            conf = CompleteConfig(dtype, ctype, mtype)
            model = load_model(conf, chk)
            upload_model(model, _model,
                    epochs=data['meta']['epochs'],
                    optimizer=data['meta']['epochs'],
                    loss_fn=data['meta']['epochs'],
                    batch_size=data['meta']['batch_size'],
                    lr=data['meta']['learning_rate'],
                    rest={
                        "non-consensus": data['meta']['non-consensus'],
                        "consensus": data['meta']['consensus'],
                        },
                    to_remote = False,
                    )
