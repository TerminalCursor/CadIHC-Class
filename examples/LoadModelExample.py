#!/usr/bin/env python3
# coding: utf-8

from histclass.backends.pytorch.utils import load_torch

# Load Model and Dataset
model, dataset = load_torch(
    "morphologynuclei-attention2grade.56b11ad68f09c67c855a4cf802cc67bdcbf99562"
)
model = model.to("cuda")
dataframe = dataset.data_frames[0]

print(model.model)
print(model)
print(dataframe.name)
