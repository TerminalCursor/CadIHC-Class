from ...tools.call_graph import cg
from enum import Enum
from typing import List


class CategoryTypes(Enum):
    TripleNegative = 1
    BasicStage = 2
    AdvancedStage = 3

    def to_str(self) -> str:
        """# String Representation"""
        cg.add("CategoryTypes.to_str")
        if self == CategoryTypes.TripleNegative:
            return "tn"
        elif self == CategoryTypes.BasicStage:
            return "stage_basic"
        elif self == CategoryTypes.AdvancedStage:
            return "stage_advanced"
        return ""

    def values(self) -> List[str] | None:
        """# Category Values"""
        cg.add("CategoryTypes.values")
        if self == CategoryTypes.TripleNegative:
            return ["NTN", "TN"]
        elif self == CategoryTypes.BasicStage:
            return ["I", "II", "III"]
        elif self == CategoryTypes.AdvancedStage:
            return ["IA", "IIA", "IIB", "IIIC", "IIIB", "IIIC"]
        return


class DataTypes(Enum):
    ImageCNN = 1
    ImageViT = 2
    Morphology = 3
    Latent = 4
    Energetics = 5
    Nuclei = 6

    def to_str(self):
        """# String Representation"""
        cg.add("DataTypes.to_str")
        if self == DataTypes.ImageCNN:
            return "image"
        elif self == DataTypes.ImageViT:
            return "image_vt"
        elif self == DataTypes.Morphology:
            return "morphology"
        elif self == DataTypes.Latent:
            return "latent"
        elif self == DataTypes.Energetics:
            return "energetics"
        elif self == DataTypes.Nuclei:
            return "nuclei"
        return ""
