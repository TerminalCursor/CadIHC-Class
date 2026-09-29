from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ..types import ActivationT
from .cnn import CnnModel
from .pointcloud import PointcloudModel
from .vit import ViTModel
from .gru import GruModel
from .interface import PyTorchModel
from torch.nn.functional import elu
from typing import Dict


def get_model(
    configuration: CompleteConfig, activation: ActivationT = elu
) -> PyTorchModel:
    """# Get a model"""
    assert configuration.input_data_types.is_valid()
    device = configuration.device
    if configuration.input_data_types == DataTypes.ImageCNN:
        return CnnModel(configuration, activation).to(device)
    elif configuration.input_data_types == DataTypes.ImageViT:
        return ViTModel(configuration, activation).to(device)
    else:
        if configuration.model_type == ModelType.Attention:
            return PointcloudModel(configuration, activation).to(device)
        return GruModel(configuration, activation).to(device)


def load_model(
    configuration: CompleteConfig,
    checkpoint_path: str,
    activation: ActivationT = elu,
) -> PyTorchModel:
    """# Load a Model"""
    device = configuration.device
    model = get_model(configuration, activation)
    return model.load(checkpoint_path).to(device)


def download_model(
    name: str,
    model_id: str | None = None,
    local_base: str = "./",
    remote_base: str = "./",
    use_cached: bool = True,
) -> PyTorchModel:
    """# Download a model from repository

    Assumed schema:
    ihc/models/classification/
          name.id/
            metadata.json
              data_type: DataTypes as a value,
              data_type_s: DataTypes as a string,
              category_type: CategoryType as a value,
              category_type_s: CategoryType as a string,
              (Optional values)
              optimizer: String of optimizer used
              loss_fn: String of loss function
              lr: Float of learning rate
              batch_size: Integer of batch_size used
              epochs: Integer of epochs trained for
            checkpoint.pth
              Checkpoint file

    Environment variables required:
     - REMOTE_HOSTNAME
     - REMOTE_UNAME
     - REMOTE_KEY_FILE
    """
    from ....tools.env import get_env
    from ....interfaces.remote_file import RemoteFile
    from os import makedirs
    from os.path import exists

    model_full_name = f"{name}"
    if model_id is not None:
        model_full_name += f".{model_id}"
    remote_path = f"{remote_base}ihc/models/classification/{model_full_name}/"
    local_path = f"{local_base}ihc/models/classification/{model_full_name}/"
    makedirs(local_path, exist_ok=True)
    is_cached = False
    if exists(f"{local_path}metadata.json") and exists(f"{local_path}checkpoint.pth"):
        is_cached = True
    if (not is_cached) or (not use_cached):
        hostname = get_env("REMOTE_HOSTNAME").unwrap()
        username = get_env("REMOTE_UNAME").unwrap()
        key_filename = get_env("REMOTE_KEY_FILE").unwrap()
        remote_meta = RemoteFile(
            remote_filename=f"{remote_path}metadata.json",
            local_filename=f"{local_path}metadata.json",
            hostname=hostname,
            username=username,
            key_filename=key_filename,
        )
        remote_meta.download()
        remote_chk = RemoteFile(
            remote_filename=f"{remote_path}checkpoint.pth",
            local_filename=f"{local_path}checkpoint.pth",
            hostname=hostname,
            username=username,
            key_filename=key_filename,
        )
        remote_chk.download()
    # Parse Metadata
    model_type = ModelType.Attention
    data_type = DataTypes.Unknown
    category_type = CategoryType.Unknown
    classic_norm = False
    model_kwargs = {}
    confidences = None
    with open(local_path + "metadata.json", "r") as fhandle:
        from json import load

        data = load(fhandle)
        assert "model_type" in data.keys(), "ModelType not found in metadata"
        assert "data_type" in data.keys(), "DataType not found in metadata"
        assert "category_type" in data.keys(), "CategoryType not found in metadata"
        if "use_classic" in data.keys():
            if data["use_classic"] == "yes":
                classic_norm = True
        if "patches" in data.keys():
            model_kwargs["vit"] = {"image_size": data["patches"]}
        if "confidences" in data.keys():
            confidences = data["confidences"]
        model_type = ModelType(data["model_type"])
        data_type = DataTypes(data["data_type"])
        category_type = CategoryType(data["category_type"])
        assert data_type.is_valid()
        assert category_type != CategoryType.Unknown
    config = CompleteConfig(
        data_type, category_type, model_type, **model_kwargs  # pyright: ignore
    )

    model = load_model(config, local_path + "checkpoint.pth")
    model.classic_data_norm = classic_norm
    model.confidences = confidences
    assert isinstance(model, PyTorchModel)
    return model


def upload_model(
    model: PyTorchModel,
    name: str,
    optimizer: str | None = None,
    loss_fn: str | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    rest: Dict[str, str | int | float] | None = None,
    local_base: str = "./",
    remote_base: str = "./",
    to_remote: bool = True,
):
    """# Upload Model to Model Repository

    Assumed schema:
    ihc/models/classification/
          name.id/
            metadata.json
              model_type: ModelType as a value,
              model_type_s: ModelType as a string,
              data_type: DataTypes as a value,
              data_type_s: DataTypes as a string,
              category_type: CategoryType as a value,
              category_type_s: CategoryType as a string,
              (Optional values)
              optimizer: String of optimizer used
              loss_fn: String of loss function
              lr: Float of learning rate
              batch_size: Integer of batch_size used
              epochs: Integer of epochs trained for
            checkpoint.pth
              Checkpoint file

    Environment variables required:
     - REMOTE_HOSTNAME
     - REMOTE_UNAME
     - REMOTE_KEY_FILE
    """
    from ....tools.env import get_env
    from ....interfaces.remote_file import RemoteFile
    from json import dumps
    from os import remove, makedirs

    def _hash(fpath: str) -> str:
        """# Compute hash of file"""
        from hashlib import sha1

        hash_obj = sha1()
        _the_hash = ""
        with open(fpath, "rb") as fhandle:
            while chunk := fhandle.read(8192):
                hash_obj.update(chunk)
            _the_hash = hash_obj.hexdigest()
        return _the_hash

    metadata: Dict[str, str | int | float] = {
        "model_type": model.model_type.value,
        "model_type_s": model.model_type.to_str(),
        "data_type": model.data_types.value,
        "data_type_s": model.data_types.to_str(),
        "category_type": model.category_type.value,
        "category_type_s": model.category_type.to_str(),
    }
    if optimizer is not None:
        metadata["optimizer"] = optimizer
    if loss_fn is not None:
        metadata["loss_fn"] = loss_fn
    if lr is not None:
        metadata["lr"] = lr
    if batch_size is not None:
        metadata["batch_size"] = batch_size
    if epochs is not None:
        metadata["epochs"] = epochs
    if rest is not None:
        for key, value in rest.items():
            metadata[key] = value
    model.save(".tmp.pth")
    model_hash = _hash(".tmp.pth")
    remove(".tmp.pth")
    local_path = f"{local_base}ihc/models/classification/{name}.{model_hash}/"
    remote_path = f"{remote_base}ihc/models/classification/{name}.{model_hash}/"
    makedirs(local_path, exist_ok=True)
    with open(local_path + "metadata.json", "w+") as fhandle:
        fhandle.write(dumps(metadata, indent=2))
    model.save(local_path + "checkpoint.pth")

    if to_remote:
        hostname = get_env("REMOTE_HOSTNAME").unwrap()
        username = get_env("REMOTE_UNAME").unwrap()
        key_filename = get_env("REMOTE_KEY_FILE").unwrap()
        remote_chk = RemoteFile(
            remote_filename=f"{remote_path}checkpoint.pth",
            local_filename=f"{local_path}checkpoint.pth",
            hostname=hostname,
            username=username,
            key_filename=key_filename,
        )
        remote_meta = RemoteFile(
            remote_filename=f"{remote_path}metadata.json",
            local_filename=f"{local_path}metadata.json",
            hostname=hostname,
            username=username,
            key_filename=key_filename,
        )
        remote_chk.upload()
        remote_meta.upload()
