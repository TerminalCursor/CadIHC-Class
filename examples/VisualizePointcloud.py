from histclass.backends.pytorch.utils import load_torch
from histclass.interfaces.data.types import DataTypes
from torch import no_grad, log, exp, randint
from torch import arange as tarange
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from numpy import array
from numpy.random import permutation
from scipy.io import loadmat


def visualize_cnn(model, dataframe):
    data = dataframe[0][0].to(model.device)[None, ...]
    im = data.detach().cpu().numpy()[0, ...].transpose(1, 2, 0)
    data = model.model.features[0](data)
    # data = model.model.features[1](data)
    # data = model.model.features[2](data)
    data = data.squeeze()
    # data = data[randint(data.shape[0],(1,)).item(),...]
    data = data.mean(0)
    data = data.detach().cpu().numpy()
    plt.imshow(im)
    plt.imshow(
        data,
        alpha=0.75,
        cmap="inferno",
        extent=[0, im.shape[1], im.shape[0], 0],
        vmax=1,
    )
    plt.show()


def visualize_vit(model, dataframe):
    from matplotlib.pyplot import imread

    data = dataframe[0][0].to(model.device)
    x = model.model.conv_proj(data)
    x = x.permute((2, 1, 0))
    x = model.model.encoder.layers.encoder_layer_0.ln_1(x)
    a, _map = model.model.encoder.layers.encoder_layer_0.self_attention(x, x, x)
    _map = _map.detach().cpu().numpy()
    im = data.detach().cpu().numpy()
    im = im.T
    plt.imshow(im)
    plt.imshow(
        _map.mean(axis=1)[::-1, :],
        cmap="inferno",
        alpha=0.5,
        extent=[0, data.shape[1], 0, data.shape[2]],
    )
    plt.show()


def visualize_pc(model, dataframe, dataset_name):
    from pandas import read_csv

    ds = ""
    if dataframe.data_types & DataTypes.Nuclei:
        ds = "CNJoinedMetrics.csv"
    elif dataframe.data_types & DataTypes.Morphology:
        ds = "Metrics.csv"
    elif dataframe.data_types & DataTypes.Energetics:
        ds = "Metrics.csv"
    raw_data = read_csv(dataset_name + ds)
    n = min(raw_data.shape[0], dataframe.data.shape[0])
    perms = permutation(n)[: model.model.max_points]
    cloud_data = dataframe.data[perms, ...].to(model.device)
    with no_grad():
        cd = model.model.cloud_fill(cloud_data)  # pyright: ignore
        attn_out, attn_weights = model.model.attention_layers[  # pyright: ignore
            0
        ].attention_layer(  # pyright: ignore
            cd, cd, cd
        )
        attn_weights = attn_weights.detach().cpu().numpy()
        # plt.imshow(attn_weights)
        # plt.show()
        weights = attn_weights.mean(axis=1)
        # plt.imshow(weights[None,:])
        # plt.show()
        plt.imshow(im)
        xy = raw_data.iloc[perms, :]
        x = xy["CentroidX"]
        y = xy["CentroidY"]
        values = []
        for idx in range(model.model.max_points):
            _x = x.iloc[idx]
            _y = y.iloc[idx]
            _c = weights[idx]
            values.append((_x, _y, _c))
        values = array(values)
        plt.scatter(values[:, 0], values[:, 1], c=values[:, 2], cmap="inferno")
        plt.show()


# Load Model and Dataset
model, dataset = load_torch(
    # "morphology-attention2tn.ccee1ddfd950348df135ee2a36ee1e0041972aca"
    # "morphologyenergetics-attention2tn.060799ec5131b9e37458ebf9d3118c800c4b7835"
    # "image_vt-attention2grade.95f2f63800c98fd335b375fa1cdfcdba9de37a5f"
    # "image_vt-attention2tn.4e27be4657982c68a24a7b146c76d7a348925df1"
    # "morphologynuclei-attention2grade.56b11ad68f09c67c855a4cf802cc67bdcbf99562"
    "image-attention2grade.8664f3958557ae1a0886e3d0f986b022b2b6d1cd"
)
model = model.to("cuda")
dataframe = None
_, _, v_d, _, _ = dataset.get_dataloaders()
dataframe = dataset.data_frames[2]
dataframe = v_d.dataset.datasets[0]
dataset_name = dataframe.name
dataset_name = f"ihc/data/classification/{dataset_name}/"

raw_data = None
cloud_data = None
im = imread(dataset_name + "image.png")
print(model.data_types)
print(model.model)
if model.data_types == DataTypes.ImageCNN:
    visualize_cnn(model, dataframe)
elif model.data_types == DataTypes.ImageViT:
    visualize_vit(model, dataframe)
elif model.data_types & DataTypes.Morphology:
    visualize_pc(model, dataframe, dataset_name)
elif model.data_types & DataTypes.Nuclei:
    visualize_pc(model, dataframe, dataset_name)
elif model.data_types & DataTypes.Energetics:
    visualize_pc(model, dataframe, dataset_name)
