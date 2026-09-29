from ..tools.call_graph import cg
from typing import Self, Dict, Any
from .data.types import DataTypes, CategoryType
from .models.types import ModelType


class PointcloudConfig:
    """# Pointcloud Model Configuration"""

    max_points: int = 96
    min_points: int = 64
    dropout: float = 0.15
    dropout_output: bool = False
    n_ff_nodes: int = 32
    n_ff_layers: int = 3
    n_blocks: int = 1

    def __init__(self: Self, **kwargs):
        """# Initialization

        Load Any Passed Values
        """
        cg.add("PointcloudConfig.__init__")
        if "max_points" in kwargs.keys():
            self.max_points = kwargs["max_points"]
        if "min_points" in kwargs.keys():
            self.min_points = kwargs["min_points"]
        if "dropout" in kwargs.keys():
            self.dropout = kwargs["dropout"]
        if "dropout_output" in kwargs.keys():
            self.dropout_output = kwargs["dropout_output"]
        if "n_ff_nodes" in kwargs.keys():
            self.n_ff_nodes = kwargs["n_ff_nodes"]
        if "n_ff_layers" in kwargs.keys():
            self.n_ff_layers = kwargs["n_ff_layers"]
        if "n_blocks" in kwargs.keys():
            self.n_blocks = kwargs["n_blocks"]

    def to_dict(self: Self) -> dict:
        """# Serialize to Dictionary"""
        cg.add("PointcloudConfig.to_dict")
        return {
            "max_points": self.max_points,
            "min_points": self.min_points,
            "dropout": self.dropout,
            "dropout_output": self.dropout_output,
            "n_ff_nodes": self.n_ff_nodes,
            "n_ff_layers": self.n_ff_layers,
        }


class CnnConfig:
    """# CNN Configuration"""

    window_width: int = 1920
    window_height: int = 2560

    def __init__(self: Self, **kwargs):
        """# Initialization

        Load Any Passed Values
        """
        cg.add("CnnConfig.__init__")
        if "window_width" in kwargs.keys():
            self.window_width = kwargs["window_width"]
        if "window_height" in kwargs.keys():
            self.window_height = kwargs["window_height"]

    def to_dict(self: Self) -> dict:
        """# Serialize to Dictionary"""
        cg.add("CnnConfig.to_dict")
        return {
            "window_width": self.window_width,
            "window_height": self.window_height,
        }


class ViTConfig:
    """# ViT Configuration"""

    image_size: int = 1
    patch_size: int = 16
    num_layers: int = 4
    num_heads: int = 4
    hidden_dim: int = 16
    mlp_dim: int = 16
    dropout: float = 0.05
    attention_dropout: float = 0.05

    def __init__(self: Self, **kwargs):
        """# Initialization

        Load Any Passed Values
        """
        cg.add("ViTConfig.__init__")
        if "image_size" in kwargs.keys():
            self.image_size = kwargs["image_size"]
        if "patch_size" in kwargs.keys():
            self.patch_size = kwargs["patch_size"]
        if "num_layers" in kwargs.keys():
            self.num_layers = kwargs["num_layers"]
        if "num_heads" in kwargs.keys():
            self.num_heads = kwargs["num_heads"]
        if "hidden_dim" in kwargs.keys():
            self.hidden_dim = kwargs["hidden_dim"]
        if "mlp_dim" in kwargs.keys():
            self.mlp_dim = kwargs["mlp_dim"]
        if "dropout" in kwargs.keys():
            self.dropout = kwargs["dropout"]
        if "attention_dropout" in kwargs.keys():
            self.attention_dropout = kwargs["attention_dropout"]

    def to_dict(self: Self) -> dict:
        """# Serialize to Dictionary"""
        cg.add("ViTConfig.to_dict")
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
        }


class EarlyExitConfig:
    """# Early Exit Detector Configuration"""

    kwargs: Dict[str, Any] = {
        "start_after": 100,
        "lookback": 600,
        "p_cutoff": 0.05,
        "on_index": 1,
    }
    max_failures: int = 200

    def to_dict(self: Self) -> dict:
        """# Serialize to Dictionary"""
        cg.add("EarlyExitConfig.to_dict")
        return {
            "kwargs": self.kwargs,
            "max_failures": self.max_failures,
        }


class CompleteConfig:
    """# Configuration"""

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
    pointcloud_config: PointcloudConfig | None = None
    image_values: CnnConfig | None = None
    vision_transformer_config: ViTConfig | None = None

    def __init__(
        self: Self,
        input_data_types: DataTypes,
        category_type: CategoryType,
        model_type: ModelType = ModelType.Attention,
        loss_fn="mse_loss",
        optimizer="Adam",
        **kwargs: Dict[str, Any],
    ):
        """# Initialization

        input_data_types: DataTypes of model inputs
        output_data_type: CategoryType of model output
        loss_fn: Loss function name to use
        optimizer: Optimizer name to use
        Load Any Additional Passed Values
        """
        cg.add("CompleteConfig.__init__")
        self.model_type = model_type
        self.input_data_types = input_data_types
        self.output_data_type = category_type
        assert (
            self.input_data_types.is_valid()
        ), f"Invalid combination!\n{input_data_types}"
        self.model_name = (
            self.input_data_types.to_str() + "_" + self.output_data_type.to_str()
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
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        return

    def to_dict(self: Self) -> dict:
        """# Serialize to Dictionary"""
        cg.add("CompleteConfig.to_dict")
        return_dict = {
            "model_name": self.model_name,
            "model_type": self.model_type.to_str(),
            "input_data_types": [x.to_str() for x in self.input_data_types],
            "output_data_type": self.output_data_type.to_str(),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "loss_fn": self.loss_fn,
            "batch_size": self.batch_size,
            "train_val_split": self.train_val_split,
            "device": self.device,
            "early_exit_config": self.early_exit_config.to_dict(),
        }
        if self.pointcloud_config is not None:
            return_dict["pointcloud_config"] = self.pointcloud_config.to_dict()
        if self.image_values is not None:
            return_dict["image_values"] = self.image_values.to_dict()
        if self.vision_transformer_config is not None:
            return_dict["vision_transformer_config"] = (
                self.vision_transformer_config.to_dict()
            )
        return return_dict
