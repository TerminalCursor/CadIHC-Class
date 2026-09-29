from ....tools.call_graph import cg
from enum import Flag, auto
from typing import Self


class DataTypes(Flag):
    """# Data Types for Model Input

    ImageCNN, ImageViT: Image Inputs
    Morphology, Latent, Energetics, Nuclei: Pointcloud Inputs

    Some of the Pointcloud Combinations are considered valid
    Image types can be the only ones to be valid
    """

    Unknown = 0
    ImageCNN = auto()
    ImageViT = auto()
    Morphology = auto()
    Latent = auto()
    Energetics = auto()
    Nuclei = auto()

    def is_valid(self) -> bool:
        """# Determine if it is a valid combination"""
        cg.add("DataTypes.is_valid")
        if self == self.ImageCNN:
            return True
        elif self == self.ImageViT:
            return True
        elif self == self.Morphology:
            return True
        elif self == self.Latent:
            return True
        elif self == self.Energetics:
            return True
        elif self == self.Nuclei:
            return True
        elif self == self.Morphology | self.Energetics:
            return True
        elif self == self.Morphology | self.Nuclei:
            return True
        return False

    def to_str(self: Self) -> str:
        """# String Representation"""
        cg.add("DataTypes.to_str")
        return_string = ""
        if not self.is_valid():
            return return_string
        if self & DataTypes.ImageCNN:
            return_string += "image"
        if self & DataTypes.ImageViT:
            return_string += "image_vt"
        if self & DataTypes.Morphology:
            return_string += "morphology"
        if self & DataTypes.Latent:
            return_string += "latent"
        if self & DataTypes.Energetics:
            return_string += "energetics"
        if self & DataTypes.Nuclei:
            return_string += "nuclei"
        return return_string

    def dimensionality(self: Self) -> int:
        """# Get Dimensionality of Dataset"""
        cg.add("DataTypes.dimensionality")
        if not self.is_valid():
            return 0
        _dim = 0
        if self & DataTypes.Morphology:
            _dim += 13
        if self & DataTypes.Latent:
            _dim += 2
        if self & DataTypes.Energetics:
            _dim += 4
        if self & DataTypes.Nuclei:
            _dim += 9
        return _dim
