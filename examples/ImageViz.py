from histclass.backends.pytorch.models.utils import download_model
from histclass.backends.pytorch.data.utils import get_data, get_metadata_labels
from histclass.interfaces.data.types import DataTypes
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from numpy import zeros

inferno = colormaps["inferno"]
BASE = "FigurePieces/Figure2-"


def upscale_match(im0, im1):
    from itertools import product
    from numpy import zeros

    # Detect upscale necessary
    w0, h0, c0 = im0.shape
    w1, h1, c1 = im1.shape
    assert c0 == 3
    assert c0 == c1
    if ((w0 > w1) and (h0 < h1)) or ((w0 < w1) and (h0 > h1)):
        raise NotImplementedError("This complicated case has not been added")
    # Second is smaller
    if w0 < w1:
        nim1, nim0 = upscale_match(im1, im0)
        return nim0, nim1
    W = w0 - (w0 % w1)
    H = h0 - (h0 % h1)
    nim0 = zeros((W, H, 3))
    nim0 = im0[:W, :H, :]
    dW = w0 // w1
    dH = h0 // h1
    nim1 = zeros((W, H, 3))
    for x, y in product(range(dW), range(dH)):
        nim1[x::dW, y::dH, :] = im1
    return nim0, nim1


def blend(im0, im1, alpha=0.5):
    from numpy import clip

    # Channel fixing
    im0 = im0[..., :3]
    # im0 /= im0.max()
    im1 = im1[..., :3]
    # im1 /= im1.max()
    im0, im1 = upscale_match(im0, im1)
    imo = im0 * (1 - alpha) + im1 * (alpha)
    return clip(imo, 0, 1)


def index2slice(index, size):
    return slice(index * size, (index + 1) * size)


def chunk(image, patch_size=256):
    from itertools import product

    c, w, h = 0, 0, 0
    if len(image.shape) == 4:
        _, c, w, h = image.shape
    else:
        c, w, h = image.shape
    assert c <= 4
    nw = image.shape[1] // patch_size
    nh = image.shape[2] // patch_size
    whs = product(range(nw), range(nh))
    return list(whs), (nw, nh)


model = download_model(
    "image-attention2tn.031a41308842262d5fb3b389e442a06e48f826dc"
    # "image-attention2grade.8664f3958557ae1a0886e3d0f986b022b2b6d1cd"
    # "image_vt-attention2grade.95f2f63800c98fd335b375fa1cdfcdba9de37a5f"
)

# Load image
# image_src = "ihc/data/classification/TMA00.A01/image.png"
# image_src = "ihc/data/classification/TMA00.A01/"
image_src = "ihc/data/classification/TMA00.C05/"
metadata, labels = get_metadata_labels(image_src, model.category_type)
assert metadata is not None
data = get_data(image_src, model.data_types, metadata)
assert data is not None

if model.data_types == DataTypes.ImageCNN:
    image, y, yl = model.visualize(data, n_levels=1)
    rng = abs(y.min())
    if rng <= 0:
        rng = 1
    image /= 255
    image = image[:,:,:,::-1]
    out_nm = Normalize(vmin=0, vmax=rng)(y)
    out_nm = Normalize(vmin=0, vmax=2)(y)
    out_nm = out_nm[0, ...]
    for _alpha in [0, 0.25, 0.5, 0.75, 1.0]:
        plt.imsave(
            f"{BASE}CNN-{int(_alpha*100):02}.png",
            blend(image[0, ...], inferno(out_nm), _alpha),
        )
        # plt.imshow(image[0, ...])
        # plt.imshow(
        #    y[0, ...],
        #    alpha=_alpha,
        #    cmap="inferno",
        #    extent=[0, image.shape[2], image.shape[1], 0],
        #    vmax=rng,
        # )
        # plt.show()
elif model.data_types == DataTypes.ImageViT:
    im_size = model.image_size
    chunks, shape = chunk(data, im_size)
    out_im = zeros((shape[0] * im_size, shape[1] * im_size, 3))
    N = im_size // 16
    out_dt = zeros(
        (
            shape[0] * N,
            shape[1] * N,
        )
    )
    partial_image = None
    for sx, sy in chunks:
        s_x = index2slice(sx, im_size)
        s_y = index2slice(sy, im_size)
        partial_image = data[:, s_x, s_y]
        image, y, _ = model.visualize(partial_image)
        out_im[s_x, s_y, :] = image[0, ...]
        s_x = index2slice(sx, N)
        s_y = index2slice(sy, N)
        out_dt[s_x, s_y] = y
    out_im /= 255
    out_im = out_im[:, :, ::-1]
    data = data.cpu().numpy()
    out_nm = Normalize(vmin=out_dt.min(), vmax=out_dt.max())(out_dt)
    for _alpha in [0, 0.25, 0.5, 0.75, 1.0]:
        plt.imsave(
            f"{BASE}ViT-{int(_alpha*100):02}.png",
            blend(out_im, inferno(out_nm), _alpha),
        )
