from .models.interface import PyTorchModel
from .models.utils import download_model
from .data.dataset import TorchDataset
from .data.dataset import load_data_from_model
from typing import Tuple


def load_torch(
    model_name: str,
    model_id: str | None = None,
    local_base: str = "./",
    remote_base: str = "./",
    use_cached: bool = True,
    metadata_only: bool = False,
) -> Tuple[PyTorchModel, TorchDataset]:
    """
    Load and return a PyTorch model and its corresponding dataset.

    This function downloads the specified model if it's not already cached,
    then loads the dataset associated with that model from local storage.

    Args:
        - model_name (str): The name of the model to load.
        - model_id (str | None): The ID of the model to load. If None,
            the latest model will be loaded if available.
        - local_base (str): The base directory for local storage. Defaults to "./".
        - remote_base (str): The base URL for remote model download. Defaults to "./".
        - use_cached (bool): Whether to use cached models instead of downloading them.
        - metadata_only (bool): Whether to only load model metadata information

    Returns:
        A tuple containing the loaded PyTorch model and dataset.
    """
    model = download_model(
        model_name,
        model_id,
        local_base=local_base,
        remote_base=remote_base,
        use_cached=use_cached,
        metadata_only=metadata_only,
    )
    dataset = load_data_from_model(model, local_base=local_base)
    return (model, dataset)
