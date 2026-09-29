from histclass.backends.pytorch.models.utils import download_model
from histclass.interfaces.data.types import DataTypes
from glob import glob
from itertools import product
from os.path import exists
from json import load


def random_sample(data, model):
    from numpy.random import randint
    from numpy import zeros
    from torch import Tensor

    if DataTypes.ImageViT in model.data_types:
        im_size = model.image_size
        w, h, c = -1, -1, -1
        extra_c = -1
        if len(data.shape) == 4:
            extra_c, c, w, h = data.shape
        else:
            c, w, h = data.shape
        new_data = None
        if extra_c > 0:
            new_data = zeros((extra_c,c,im_size,im_size))
        else:
            new_data = zeros((c,im_size,im_size))
        assert c <= 4
        if w <= im_size:
            s_x = 0
            s_X = w
        else:
            s_x = randint(w - im_size)
            s_X = s_x + im_size
        if h <= im_size:
            s_y = 0
            s_Y = h
        else:
            s_y = randint(h - im_size)
            s_Y = s_y + im_size
        new_data[..., :s_X-s_x,:s_Y-s_y] = data[..., s_x:s_X, s_y:s_Y]
        return Tensor(new_data)
    return data


def embedding_name(model_name):
    return f"Embedding-{model_name}"


def add_embedding(fname, non_replace=True, **kwargs):
    from json import loads, dumps

    data = dict()
    try:
        with open(f"{fname}embedding.json", "r") as fhandle:
            data = loads(fhandle.read())
    except:
        pass
    for key, value in kwargs.items():
        if non_replace and (key in data.keys()):
            continue
        data[key] = value
    with open(f"{fname}embedding.json", "w+") as fhandle:
        fhandle.write(dumps(data, indent=2))
    return data


model_names = [
    "image-attention2grade.8664f3958557ae1a0886e3d0f986b022b2b6d1cd",
]

meta_dataset = dict()
meta_dataset_n = dict()
keys = ["tn", "stage_advanced", "grade"]
for image_src in glob("ihc/data/classification/*.*/"):
    model_name_full = glob("ihc/models/classification/image*/")[0]
    model_name = (
        model_name_full.replace("ihc", "")
        .replace("models", "")
        .replace("classification", "")
        .replace("/", "")
    )
    model = download_model(model_name)

    # Load image
    # image_src = "ihc/data/classification/TMA00.A01/"
    source = image_src

    if not exists(source + "/metadata.json"):
        print(image_src + " empty")
        continue
    metadata_dict = {}
    with open(source + "/metadata.json", "r") as fhandle:
        metadata_dict = load(fhandle)
    metadata = metadata_dict
    if len(metadata.keys()) == 0:
        print(image_src + " empty")
        continue
    assert metadata is not None
    for key in keys:
        k = key + "_i"
        if k not in metadata:
            continue
        if key not in metadata:
            continue
        if key not in meta_dataset.keys():
            meta_dataset[key] = dict()
        if key not in meta_dataset_n.keys():
            meta_dataset_n[key] = dict()
        v = metadata[k]
        vn = metadata[key]
        if v not in meta_dataset[key].keys():
            meta_dataset[key][v] = 0
        if vn not in meta_dataset_n[key].keys():
            meta_dataset_n[key][vn] = 0
        meta_dataset[key][v] += 1
        meta_dataset_n[key][vn] += 1
print(meta_dataset_n)

for k in keys:
    print(k)
    for label in meta_dataset_n[k].keys():
        print(label, meta_dataset_n[k][label])
