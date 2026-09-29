from .call_graph import cg
from ..interfaces.data.types import DataTypes, CategoryType
from ..interfaces.models.types import ModelType
from ..interfaces.config import CompleteConfig


def load_config(
    data_types: DataTypes,
    category_type: CategoryType,
    model_type: ModelType = ModelType.Attention,
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
        data_types,
        category_type,
        model_type,
        vit={"image_size": patches, "patch_size": patch_size},
    )
    config.device = device
    return config
