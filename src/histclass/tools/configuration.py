from ..interfaces.data.types import DataTypes, CategoryType
from ..interfaces.models.types import ModelType
from ..interfaces.config import CompleteConfig
from typing import List


def load_config(
    data_types: DataTypes,
    category_type: CategoryType,
    model_type: ModelType = ModelType.Attention,
    patches: int | None = None,
    patch_size: int = 16,
    ablation_columns: List[int] = [],
    device: str = "cuda",
) -> CompleteConfig:
    """
    Load a complete configuration for the model.

    Args:
        data_types (DataTypes): The input data types.
        category_type (CategoryType): The classification data type.
        model_type (ModelType, optional): The model type. Defaults to ModelType.Attention.
        patches (int | None, optional): The number of patches. For ViT only. Defaults to None.
        patch_size (int, optional): The size of patches. Defaults to 16.
        device (str, optional): The device to use for the model. Defaults to "cuda".

    Returns:
        CompleteConfig: The loaded configuration.

    Notes:
        This function creates a CompleteConfig object with the specified data types and model type. For ViT models,
        it also specifies the number of patches and patch size.
    """
    data_types.ablations = ablation_columns  # pyright: ignore
    config = CompleteConfig(
        data_types,
        category_type,
        model_type,
        vit={"image_size": patches, "patch_size": patch_size},
        a={"columns": ablation_columns},
    )
    config.device = device
    return config
