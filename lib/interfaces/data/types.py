from ...tools.call_graph import cg
from enum import Enum, Flag, auto
from typing import List, Self


class DataframeType(Enum):
    """# Dataframe Usage Type"""

    Unknown = 0
    Training = 1
    Validation = 2
    Test = 3


class CategoryType(Enum):
    """# Category Types for Model Output

    TripleNegative: Whether the Sample is Triple Negative
    BasicStage: Whether the Sample should be classified into Stages Coarsely (I, II, ...)
    AdvancedStage: Whether the Sample should be classified into Stages Finely (IA, IIA, ...)
    Grade: Whether the Sample should be classified into Grades (1, 2, ...)
    """

    Unknown = 0
    TripleNegative = 1
    BasicStage = 2
    AdvancedStage = 3
    Grade = 4

    def to_str(self: Self) -> str:
        """# String Representation"""
        cg.add("CategoryType.to_str")
        if self == CategoryType.TripleNegative:
            return "tn"
        elif self == CategoryType.BasicStage:
            return "stage_basic"
        elif self == CategoryType.AdvancedStage:
            return "stage_advanced"
        elif self == CategoryType.Grade:
            return "grade"
        assert False, "Undefined Type!"  # pyright: ignore[reportUnreachable]

    def values(self: Self) -> List[str] | None:
        """# Category Values"""
        cg.add("CategoryType.values")
        if self == CategoryType.TripleNegative:
            return ["NTN", "TN"]
        elif self == CategoryType.BasicStage:
            return ["I", "II", "III"]
        elif self == CategoryType.AdvancedStage:
            return ["IA", "IIA", "IIB", "IIIC", "IIIB", "IIIC"]
        elif self == CategoryType.Grade:
            return ["One", "Two", "Three"]
        assert False, "Undefined Type!"  # pyright: ignore[reportUnreachable]


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

    def to_str(self: Self):
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
