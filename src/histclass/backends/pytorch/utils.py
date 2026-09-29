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
) -> Tuple[PyTorchModel, TorchDataset]:
    """# Load Torch Implementation

    - Load Model
    - Load Dataset
    """
    model = download_model(
        model_name,
        model_id,
        local_base=local_base,
        remote_base=remote_base,
        use_cached=use_cached,
    )
    dataset = load_data_from_model(model, local_base=local_base)
    return (model, dataset)
