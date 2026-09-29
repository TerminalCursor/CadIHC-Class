from histclass.backends.pytorch.models.utils import download_model
from histclass.backends.pytorch.data.utils import get_data, get_metadata_labels
from histclass.interfaces.data.types import DataTypes
from glob import glob
from itertools import product


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
        "ihc/models/classification/image-attention2hilograde.472cb9d6abc5f67ed226557bfaa25283c35ddae2/"
        "ihc/models/classification/image_vt-attention2hilograde.47c2669067cb2652a166a06d9f54079adb18a8f9/"
]

for model_name_full, image_src in product(
    glob(glob("ihc/models/classification/image*/")), sorted(glob("ihc/data/classification/*.*/"))
):
    model_name = (
        model_name_full.replace("ihc", "")
        .replace("models", "")
        .replace("classification", "")
        .replace("/", "")
    )
    model = download_model(model_name)

    # Load image
    # image_src = "ihc/data/classification/TMA00.A01/"
    metadata, labels = get_metadata_labels(image_src, model.category_type)
    if metadata is None:
        print(image_src + " empty")
        continue
    assert metadata is not None
    data = get_data(image_src, model.data_types, metadata)
    assert data is not None
    data = random_sample(data, model)

    embedding, _ = model.embed(data)
    embedding = embedding.flatten().tolist()

    en = embedding_name(model_name)
    model_kwargs = {
        en: embedding,
    }

    _ = add_embedding(image_src, **model_kwargs)
