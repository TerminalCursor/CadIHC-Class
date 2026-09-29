import pandas as pd
import sqlite3
from matplotlib import pyplot as plt
from os import remove, makedirs
from os.path import exists
from json import loads, dumps
from histclass.backends.pytorch.models.utils import load_model
from histclass.interfaces.data.types import DataTypes, CategoryType
from histclass.interfaces.models.types import ModelType
from histclass.interfaces.config import CompleteConfig

EXPORT = True

odf = None
with sqlite3.connect("Results.db") as connection:
    odf = pd.read_sql_query("SELECT * FROM RESULTS", connection)
odf = odf[
    (odf["NAME"] == "image_vt-attention2hilostage")
    | (odf["NAME"] == "image_vt-attention2hilograde")
]

df = None
with sqlite3.connect("dump/Results.db") as connection:
    df = pd.read_sql_query("SELECT * FROM RESULTS", connection)
assert df is not None
if odf is not None:
    df = pd.concat([df, odf], ignore_index=True)
full_data = df.copy()


def _hash(fpath: str) -> str:
    """# Compute hash of file"""
    from hashlib import sha1

    hash_obj = sha1()
    _the_hash = ""
    with open(fpath, "rb") as fhandle:
        while chunk := fhandle.read(8192):
            hash_obj.update(chunk)
        _the_hash = hash_obj.hexdigest()
    return _the_hash


def parse_type(type_str, TypeClass):
    from itertools import combinations

    for i in range(1, 2 + 1):
        for combination in combinations(TypeClass, i):
            combo = list(combination)
            test_type = None
            if len(combo) == 1:
                test_type = combo[0]
            else:
                test_type = combo[0] | combo[1]
            try:
                if test_type.to_str() == type_str:
                    return test_type
            except:
                pass
    return None


categories = [
    "hilostage",
    "hilograde",
]

skey = "VALIDATION_ACCURACY"
mkey = "total_accuracy"


full_data = full_data[full_data["CATEGORY"].isin(categories)]
print(full_data.head())

full_data["DIST"] = (
    (1 - full_data["TRAINING_ACCURACY_C"]) ** 2.0
    + (1 - full_data["TRAINING_ACCURACY"]) ** 2.0
    + (1 - full_data["VALIDATION_ACCURACY_C"]) ** 2.0
    + (1 - full_data["VALIDATION_ACCURACY"]) ** 2.0
)

full_data = full_data.groupby("NAME")

for name, gdf in full_data:
    gdf = gdf.sort_values("DIST").head(3)
    for timestamp in gdf["TIMESTAMP"]:
        name = gdf[gdf["TIMESTAMP"] == timestamp]["NAME"].head(1).item()
        nco = gdf[gdf["TIMESTAMP"] == timestamp]["VALIDATION_ACCURACY"].head(1).item()
        con = gdf[gdf["TIMESTAMP"] == timestamp]["VALIDATION_ACCURACY_C"].head(1).item()
        model_path_base = "dump.imarm/"
        model_log_path = f"{model_path_base}{timestamp}.txt"
        if not exists(model_log_path):
            model_path_base = "dump.earlate/"
            model_log_path = f"{model_path_base}{timestamp}.txt"
        model_chk_path = f"{model_path_base}/models/{name}-{timestamp}.pth"
        assert exists(model_log_path), timestamp
        assert exists(model_chk_path), timestamp
        model_json = None
        with open(model_log_path, "r") as f:
            log_data = f.read()
            log_data = log_data.split("\n}\n")
            assert len(log_data) > 1
            log_data = log_data[0] + "}"
            model_json = loads(log_data)
        config = CompleteConfig.from_json(model_json)
        model_type = config.model_type
        data_type = config.input_data_types
        category_type = config.output_data_type
        metadata_json = {
            "model_type": model_type.value,
            "model_type_s": model_type.to_str(),
            "data_type": data_type.value,
            "data_type_s": data_type.to_str(),
            "category_type": category_type.value,
            "category_type_s": category_type.to_str(),
            "optimizer": "AdamW",
            "loss_fn": model_json["loss_fn"],
            "epochs": model_json["epochs"],
            "lr": model_json["learning_rate"],
            "batch_size": model_json["batch_size"],
            "use_classic": "no",
            "non-consensus": nco,
            "consensus": con,
            "model_json": model_json,
        }
        model = load_model(config, model_chk_path)
        model.model = model.model.eval()

        # Export
        model.save(".tmp.pth")
        model_hash = _hash(".tmp.pth")
        remove(".tmp.pth")

        new_path = f"ihc/models/classification/{name}.{model_hash}/"
        if EXPORT:
            try:
                makedirs(new_path)
                model.save(new_path + "checkpoint.pth")
                with open(new_path + "metadata.json", "w+") as fhandle:
                    fhandle.write(dumps(metadata_json, indent=2))
            except Exception as e:
                print(f"Error exporting model: {name}")
                print(e)
