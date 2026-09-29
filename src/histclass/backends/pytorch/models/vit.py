from ....interfaces.config import CompleteConfig
from ....interfaces.data.types import DataTypes, CategoryType
from ....interfaces.models.types import ModelType
from ....interfaces.models.types import LossT, OptimizerT
from .interface import PyTorchModel
from ..types import ActivationT, DataloaderT
from typing import Self, Any
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import elu
from torchvision.models.vision_transformer import (  # pyright: ignore[reportMissingTypeStubs]
    VisionTransformer,
)


class ViTModel(PyTorchModel):
    """
    A PyTorch model for Vision Transformer (ViT) image classification.

    Attributes:
        model (Module): The underlying VisionTransformer model.
        optimizer (OptimizerT): The optimizer used to train the model.
        loss_fn (LossT): The loss function used to evaluate the model's performance.
        model_type (ModelType): The type of model (Attention).
        data_types (DataTypes): The input and output data types.
        category_type (CategoryType): The category type of the output data.
        image_size (int): The size of the images fed into the model.

    Raises:
        AssertionError: If configuration.image_values or vision_transformer_config is None.
        ValueError: If classes or n_classes are not defined.
    """

    model: Module
    optimizer: OptimizerT
    loss_fn: LossT

    model_type: ModelType = ModelType.Attention
    data_types: DataTypes
    category_type: CategoryType
    image_size: int

    def __init__(
        self: Self,
        configuration: CompleteConfig,
        activation: ActivationT = elu,
    ):
        """
        Initialize the ViTModel.

        Args:
            configuration (CompleteConfig): The configuration for the model.
            activation (ActivationT, optional): The activation function used in the model. Defaults to elu.

        Raises:
            AssertionError: If configuration.image_values or vision_transformer_config is None.
            ValueError: If classes or n_classes are not defined.
        """
        assert configuration.image_values is not None
        assert configuration.vision_transformer_config is not None
        classes = configuration.output_data_type.values()
        assert classes is not None
        n_classes = len(classes)
        self.data_types = configuration.input_data_types
        self.category_type = configuration.output_data_type
        self.image_size = configuration.vision_transformer_config.image_size
        self.model = VisionTransformer(
            num_classes=n_classes,
            **configuration.vision_transformer_config.to_dict(),  # pyright: ignore
        )

    def embed(self: Self, x: Tensor | DataloaderT, **kwargs: Any) -> Any:
        """
        Embed data from a model

        Args:
            x: Input
            kwargs: Embedding Arguments

        Returns:
            Any
        """
        from torch import cat, no_grad
        from numpy import concatenate

        with no_grad():
            if type(x) == Tensor:
                if len(x.shape) == 3:
                    x = x[None, ...]
                x = x.to(device=self.device, dtype=self.dtype)
            else:
                values = []
                labels = []
                for data in x:
                    _x = data[0]
                    _y = data[1]
                    values.append(self.embed(_x, **kwargs)[0])
                    labels.append(_y.argmax(-1).cpu().numpy())
                return concatenate(values), concatenate(labels)
            ### Lifted from PyTorch Vision Transformer Forward Pass
            z = x
            # Reshape and permute the input tensor
            z = self.model._process_input(z)  # pyright: ignore
            n = z.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(
                n, -1, -1
            )  # pyright: ignore
            z = cat([batch_class_token, z], dim=1)

            z = self.model.encoder(z)  # pyright: ignore

            # Classifier "token" as used by standard language architectures
            z = z[:, 0]
            ###
            embedding = z
            embedding = embedding.cpu().numpy()
            return embedding, None

    def visualize(self: Self, x: Any, **kwargs: Any) -> Any:
        """
        Visualize a model

        Args:
            x: Input
            kwargs: Visualization Arguments

        Returns:
            Data
        """
        from torch import cat, no_grad
        from numpy import concatenate

        with no_grad():
            if type(x) == Tensor:
                if len(x.shape) == 3:
                    x = x[None, ...]
                x = x.to(device=self.device, dtype=self.dtype)
            else:
                dataes = []
                values = []
                labels = []
                for data in x:
                    _x = data[0]
                    _y = data[1]
                    if len(_x.shape) == 4:
                        for idx in range(_x.shape[0]):
                            out = self.visualize(_x[[idx], ...], **kwargs)
                            dataes.append(out[0])
                            values.append(out[1])
                            label = _y[[idx], ...].argmax(-1).cpu().numpy()
                            labels.append(label)
                return concatenate(dataes), concatenate(values), concatenate(labels)
            n_levels = 1
            if "n_levels" in kwargs.keys():
                n_levels = kwargs["n_levels"]
                assert isinstance(n_levels, int)
                n_levels = (
                    n_levels
                    if n_levels <= len(self.model.encoder.layers)  # pyright: ignore
                    else len(self.model.encoder.layers)  # pyright: ignore
                )
            z = x
            z = self.model.conv_proj(z)  # pyright: ignore
            z = z.permute((0, 2, 3, 1))  # Get channel into correct position
            assert z.shape[0] == 1 # Only one image per time supported
            z = z[0, ...]
            z = self.model.encoder.layers.encoder_layer_0.ln_1(z)  # pyright: ignore
            _, _map = (
                self.model.encoder.layers.encoder_layer_0.self_attention(  # pyright: ignore
                    z, z, z
                )
            )
            visual = _map.mean(axis=1)
            ## ### Lifted from PyTorch Vision Transformer Forward Pass
            ## z = x
            ## # Reshape and permute the input tensor
            ## z = self.model._process_input(z)  # pyright: ignore
            ## n = z.shape[0]
            ## assert n == 1

            ## # Expand the class token to the full batch
            ## batch_class_token = self.model.class_token.expand(
            ##     n, -1, -1
            ## )  # pyright: ignore
            ## z = cat([batch_class_token, z], dim=1)

            ## z = z + self.model.encoder.pos_embedding  # pyright: ignore
            ## z = self.model.encoder.dropout(z)  # pyright: ignore
            ## # Replace the following for the visualization
            ## y = z
            ## y_map = None
            ## for layer_idx in range(n_levels):
            ##     if layer_idx + 1 == n_levels:
            ##         y = self.model.encoder.layers[0].ln_1(y)  # pyright: ignore
            ##         y, y_map = self.model.encoder.layers[  # pyright: ignore
            ##             0
            ##         ].self_attention(  # pyright: ignore
            ##             y, y, y
            ##         )
            ##     else:
            ##         y = self.model.encoder.layers[layer_idx](y)  # pyright: ignore
            ## z = y_map
            ## assert z is not None
            ## ###
            ## visual = z[:,:-1,:-1]
            ## visual = visual.cpu().numpy()
            _x = x.cpu().numpy().transpose(0, 2, 3, 1)
            visual = visual.cpu().numpy()[None, ...]
            return _x, visual, None
