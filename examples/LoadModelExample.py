#!/usr/bin/env python3
# coding: utf-8

from histclass.backends.pytorch.utils import load_torch

# Load Model and Dataset
model, dataset = load_torch(
    "image_vt-attention2grade.95f2f63800c98fd335b375fa1cdfcdba9de37a5f"
    # "image-attention2grade.8664f3958557ae1a0886e3d0f986b022b2b6d1cd"
    # "morphologynuclei-attention2grade.56b11ad68f09c67c855a4cf802cc67bdcbf99562"
)
model = model.to("cuda")
dataframe = dataset.data_frames[0]

print(model.model)
print(model)
print(dataframe.name)
