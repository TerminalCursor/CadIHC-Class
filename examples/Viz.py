from histclass.backends.pytorch.utils import load_torch
from histclass.interfaces.data.types import DataTypes
from matplotlib import pyplot as plt

model, dataset = load_torch(
    #"image-attention2grade.8664f3958557ae1a0886e3d0f986b022b2b6d1cd"
    "image_vt-attention2grade.95f2f63800c98fd335b375fa1cdfcdba9de37a5f"
)
print(model.model)
print(len(model.model.encoder.layers))
dls = dataset.get_dataloaders()
vd = dls[1]

y,yl = model.embed(vd)
print(y)
print(y.shape)
print(yl)
print(yl.shape)
#image = None
#label = None
#for o in vd:
#    image = o[0][0,...]
#    label = o[1][0,...].argmax().cpu().numpy()
#im, y, yl = model.visualize(image)
#image = image.cpu().numpy().transpose(1,2,0)
image, y, yl = model.visualize(vd, n_levels=1)
print(image.shape)
print(y.shape)
print(yl.shape)
print(y.min())

for _alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    plt.imshow(image[0,...])
    rng = abs(y.min())
    if rng <= 0:
        rng = 1
    plt_kwargs = {}
    if DataTypes.ImageCNN in model.data_types:
        plt_kwargs["vmax"] = rng
    plt.imshow(y[0,...],
               alpha=_alpha,
               cmap="inferno",
               extent=[0,image.shape[2], image.shape[1], 0],
               **plt_kwargs,
               )
    plt.show()
