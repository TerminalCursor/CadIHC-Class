from .call_graph import cg
from ..interfaces.data.types import DataTypes, CategoryTypes
from ..interfaces.config import CompleteConfig
from typing import List


def load_config(
    data_types: List[DataTypes],
    category_type: CategoryTypes,
    patches: int | None = None,
    patch_size: int = 16,
    device: str = "cuda",
) -> CompleteConfig:
    """# Load a Dictionary Config

    data_types: Input Data Types
    category_type: Classification Data Type

    For ViT only
    patches: Number of patches
    patch_size: Size of patches
    """
    cg.add("load_config")
    config = CompleteConfig(
        data_types, category_type, vit={"patches": patches, "patch_size": patch_size}
    )
    config.device = device
    return config
