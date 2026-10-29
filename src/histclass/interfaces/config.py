from copy import deepcopy as copy
from typing import Self, Dict, List, Any
from .data.types import DataTypes, CategoryType
from .models.types import ModelType


class PointcloudConfig:
    """
    Configuration for pointcloud model.

    Attributes:
        max_points (int): Maximum number of points to sample.
        min_points (int): Minimum number of points to sample.
        dropout (float): Dropout probability.
        dropout_output (bool): Whether to use dropout as output.
        n_ff_nodes (int): Number of fully connected nodes.
        n_ff_layers (int): Number of fully connected layers.
        n_blocks (int): Number of blocks.
    """

    max_points: int = 96
    min_points: int = 64
    dropout: float = 0.15
    dropout_output: bool = False
    n_ff_nodes: int = 32
    n_ff_layers: int = 3
    n_blocks: int = 1

    def __init__(self: Self, **kwargs: Any):
        """
        Initialize pointcloud configuration.

        Args:
            kwargs (Any): Keyword arguments to load.
        """
        # Default attrs
        default_attrs_k = vars(self.__class__)["__annotations__"].keys()
        default_attrs = {
            k: vars(self.__class__)[k]
            for k in default_attrs_k
            if k in vars(self.__class__).keys()
        }
        manual_attrs = kwargs
        for key, def_val in default_attrs.items():
            if key in manual_attrs.keys():
                val = kwargs[key]
                setattr(self, key, val)
            else:
                setattr(self, key, def_val)

    def to_dict(self: Self) -> Dict[str, int | float | str]:
        """
        Serialize pointcloud configuration to dictionary.

        Returns:
            Dict[str, int | float | str]: Pointcloud configuration as a dictionary.
        """
        return copy(vars(self))


class CnnConfig:
    """
    Configuration for CNN model.

    Attributes:
        window_width (int): Window width.
        window_height (int): Window height.
    """

    window_width: int = 1920
    window_height: int = 2560

    def __init__(self: Self, **kwargs: Any):
        """
        Initialize CNN configuration.

        Args:
            kwargs (Any): Keyword arguments to load.
        """
        # Default attrs
        default_attrs_k = vars(self.__class__)["__annotations__"].keys()
        default_attrs = {
            k: vars(self.__class__)[k]
            for k in default_attrs_k
            if k in vars(self.__class__).keys()
        }
        manual_attrs = kwargs
        for key, def_val in default_attrs.items():
            if key in manual_attrs.keys():
                val = kwargs[key]
                setattr(self, key, val)
            else:
                setattr(self, key, def_val)

    def to_dict(self: Self) -> Dict[str, int | float | str]:
        """
        Serialize CNN configuration to dictionary.

        Returns:
            Dict[str, int | float | str]: CNN configuration as a dictionary.
        """
        return copy(vars(self))


class ViTConfig:
    """
    Configuration for Vision Transformer model.

    Attributes:
        image_size (int): Image size.
        patch_size (int): Patch size.
        num_layers (int): Number of layers.
        num_heads (int): Number of heads.
        hidden_dim (int): Hidden dimension.
        mlp_dim (int): Multi-layer perceptron dimension.
        dropout (float): Dropout probability.
        attention_dropout (float): Attention dropout probability.
    """

    image_size: int = 1
    patch_size: int = 16
    num_layers: int = 4
    num_heads: int = 4
    hidden_dim: int = 16
    mlp_dim: int = 16
    dropout: float = 0.05
    attention_dropout: float = 0.05

    def __init__(self: Self, **kwargs: Any):
        """
        Initialize Vision Transformer configuration.

        Args:
            kwargs (Any): Keyword arguments to load.
        """
        # Default attrs
        default_attrs_k = vars(self.__class__)["__annotations__"].keys()
        default_attrs = {
            k: vars(self.__class__)[k]
            for k in default_attrs_k
            if k in vars(self.__class__).keys()
        }
        manual_attrs = kwargs
        for key, def_val in default_attrs.items():
            if key in manual_attrs.keys():
                val = kwargs[key]
                setattr(self, key, val)
            else:
                setattr(self, key, def_val)

    def to_dict(self: Self) -> Dict[str, int | float | str]:
        """
        Serialize Vision Transformer configuration to dictionary.

        Returns:
            Dict[str, int | float | str]: Vision Transformer configuration as a dictionary.
        """
        return copy(vars(self))


class EarlyExitConfig:
    """
    Configuration for early exit detector.

    Attributes:
        kwargs (Dict[str, int | float]): Keyword arguments.
        max_failures (int): Maximum failures allowed.
    """

    kwargs: Dict[str, int | float] = {
        "start_after": 100,
        "lookback": 600,
        "p_cutoff": 0.05,
        "on_index": 1,
    }
    max_failures: int = 200

    def __init__(self: Self, **kwargs: Any):
        """
        Initialize Early Exit configuration.

        Args:
            kwargs (Any): Keyword arguments to load.
        """
        # Default attrs
        default_attrs_k = vars(self.__class__)["__annotations__"].keys()
        default_attrs = {
            k: vars(self.__class__)[k]
            for k in default_attrs_k
            if k in vars(self.__class__).keys()
        }
        manual_attrs = kwargs
        for key, def_val in default_attrs.items():
            if key in manual_attrs.keys():
                val = kwargs[key]
                setattr(self, key, val)
            else:
                setattr(self, key, def_val)

    def to_dict(
        self: Self,
    ) -> Dict[str, int | float | Dict[str, int | float]]:
        """
        Serialize early exit configuration to dictionary.

        Returns:
            Dict[str, int | float | Dict[str, int | float]]: Early exit configuration as a dictionary.
        """
        return copy(vars(self))


class AblationConfig:
    """
    Configuration for ablation study.

    Attributes:
        columns (List[int]): Columns to remove
    """

    columns: List[int] = []

    def __init__(self: Self, **kwargs: Any):
        """
        Initialize Ablation configuration.

        Args:
            kwargs (Any): Keyword arguments to load.
        """
        # Default attrs
        default_attrs_k = vars(self.__class__)["__annotations__"].keys()
        default_attrs = {
            k: vars(self.__class__)[k]
            for k in default_attrs_k
            if k in vars(self.__class__).keys()
        }
        manual_attrs = kwargs
        for key, def_val in default_attrs.items():
            if key in manual_attrs.keys():
                val = kwargs[key]
                setattr(self, key, val)
            else:
                setattr(self, key, def_val)

    def to_dict(
        self: Self,
    ) -> Dict[str, int | float | Dict[str, int | float]]:
        """
        Serialize ablation configuration to dictionary.

        Returns:
            Dict[str, int | float | Dict[str, int | float]]: Early exit configuration as a dictionary.
        """
        return copy(vars(self))


class CompleteConfig:
    """
    Configuration for complete model.

    Attributes:
        model_name (str): Model name.
        model_type (ModelType): Model type.
        input_data_types (DataTypes): Input data types.
        output_data_type (CategoryType): Output data type.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        device (str): Device to use.
        batch_size (int): Batch size.
        train_val_split (float): Training and validation split.
        loss_fn (str): Loss function name.
        optimizer (str): Optimizer name.
        early_exit_config (EarlyExitConfig): Early exit configuration.
        ablation_config (AblationConfig): Ablation configuration.
        pointcloud_config (PointcloudConfig | None): Pointcloud configuration.
        image_values (CnnConfig | None): CNN values.
        vision_transformer_config (ViTConfig | None): Vision transformer configuration.
    """

    model_name: str = ""
    model_type: ModelType
    input_data_types: DataTypes
    output_data_type: CategoryType
    epochs: int = 20_000
    learning_rate: float = 5e-6
    device: str = "cuda"
    batch_size: int = 8
    train_val_split: float = 0.8
    loss_fn: str = "mse_loss"
    optimizer: str = "Adam"
    early_exit_config: EarlyExitConfig = EarlyExitConfig()
    ablation_config: AblationConfig = AblationConfig()
    pointcloud_config: PointcloudConfig | None = None
    image_values: CnnConfig | None = None
    vision_transformer_config: ViTConfig | None = None

    def __init__(
        self: Self,
        input_data_types: DataTypes,
        category_type: CategoryType,
        model_type: ModelType = ModelType.Attention,
        loss_fn: str = "mse_loss",
        optimizer: str = "Adam",
        **kwargs: Any,
    ):
        """
        Initialize complete configuration.

        Args:
            input_data_types (DataTypes): Input data types.
            category_type (CategoryType): Output data type.
            model_type (ModelType, optional): Model type. Defaults to ModelType.Attention.
            loss_fn (str, optional): Loss function name. Defaults to "mse_loss".
            optimizer (str, optional): Optimizer name. Defaults to "Adam".
            kwargs (Any, optional): Keyword arguments to load.
        """
        # Default attrs
        default_attrs_k = vars(self.__class__)["__annotations__"].keys()
        default_attrs = {
            k: vars(self.__class__)[k]
            for k in default_attrs_k
            if k in vars(self.__class__).keys()
        }
        for key, def_val in default_attrs.items():
            setattr(self, key, def_val)
        self.model_type = model_type
        self.input_data_types = input_data_types
        self.output_data_type = category_type
        assert (
            self.input_data_types.is_valid()
        ), f"Invalid combination!\n{input_data_types}"
        self.model_name = (
            self.input_data_types.to_str()
            + "-"
            + self.model_type.to_str()
            + "2"
            + self.output_data_type.to_str()
        )
        if self.input_data_types in [DataTypes.ImageCNN, DataTypes.ImageViT]:
            _kwargs = {}
            if "cnn" in kwargs.keys():
                _kwargs = kwargs["cnn"]
            self.image_values = CnnConfig(**_kwargs)
            if self.input_data_types == DataTypes.ImageViT:
                _kwargs = {}
                if "vit" in kwargs.keys():
                    _kwargs = kwargs["vit"]
                self.vision_transformer_config = ViTConfig(**_kwargs)
                self.image_values.window_width = (
                    self.vision_transformer_config.image_size
                )
                self.image_values.window_height = (
                    self.vision_transformer_config.image_size
                )
        else:
            _kwargs = {}
            if "pc" in kwargs.keys():
                _kwargs = kwargs["pc"]
            self.pointcloud_config = PointcloudConfig(**_kwargs)
        if "a" in kwargs.keys():
            self.ablation_config = AblationConfig(**kwargs["a"])
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        return

    def to_dict(self: Self) -> Dict[str, Any]:
        """
        Serialize complete configuration to dictionary.

        Returns:
            Dict[str, Any]: Complete configuration as a dictionary.
        """
        input_data_types_v = [x.to_str() for x in self.input_data_types]
        if len(self.input_data_types.get_ablations()) > 0:
            input_data_types_v.append(f"{self.input_data_types.get_ablations()}")
        return_dict: Dict[str, Any] = {
            "model_name": self.model_name,
            "model_type": self.model_type.to_str(),
            "input_data_types": input_data_types_v,
            "output_data_type": self.output_data_type.to_str(),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "loss_fn": self.loss_fn,
            "batch_size": self.batch_size,
            "train_val_split": self.train_val_split,
            "device": self.device,
            "early_exit_config": self.early_exit_config.to_dict(),
            "ablation_config": self.ablation_config.to_dict(),
        }
        if self.pointcloud_config is not None:
            v = self.pointcloud_config.to_dict()
            return_dict["pointcloud_config"] = v
        if self.image_values is not None:
            return_dict["image_values"] = self.image_values.to_dict()
        if self.vision_transformer_config is not None:
            return_dict["vision_transformer_config"] = (
                self.vision_transformer_config.to_dict()
            )
        return return_dict

    @staticmethod
    def from_json(model_json: Dict) -> "CompleteConfig":
        from .generic import parse_type

        model_type: ModelType = parse_type(
            model_json["model_type"], ModelType  # pyright: ignore
        )
        assert model_type is not None
        data_type: DataTypes = parse_type(
            "".join(model_json["input_data_types"]), DataTypes  # pyright: ignore
        )
        assert model_type is not None
        category_type: CategoryType = parse_type(
            model_json["output_data_type"], CategoryType  # pyright: ignore
        )
        assert model_type is not None
        _kwargs = {}
        if "pointcloud_config" in model_json.keys():
            _kwargs["pc"] = model_json["pointcloud_config"]
        if "image_values" in model_json.keys():
            _kwargs["cnn"] = model_json["image_values"]
        if "vision_transformer_config" in model_json.keys():
            _kwargs["vit"] = model_json["vision_transformer_config"]

        config = CompleteConfig(
            data_type,  # pyright: ignore
            category_type,  # pyright: ignore,
            model_type,  # pyright: ignore
            **_kwargs,
        )
        for key in ["epochs", "learning_rate", "loss_fn", "batch_size", "train_val_split"]:
            if key in model_json:
                setattr(config, key, model_json[key])
        return config
