#!/usr/bin/env python3
# coding: utf-8

from histclass.backends.pytorch.models.utils import download_model

model = download_model("MN2TN_Classifier.241aeecb08f113b2ee355b0e965c8e6813692538")

print(model.model)
print(model)
