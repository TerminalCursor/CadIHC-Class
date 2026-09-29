from ..data.types import DataTypes, CategoryType
from ..models.types import ModelType
from .types import LossT, OptimizerT
from typing import Self, Any


class Model:
    """
    # Model Interface

    This class provides a basic interface for models. It includes methods to set the optimizer,
    loss function, device, and dtype. The forward method is used to perform the forward pass of the model.

    Attributes:
        optimizer (OptimizerT): The optimizer used for training.
        loss_fn (LossT): The loss function used for training.
        device (Any): The device on which the model will be trained.
        dtype (Any): The data type used by the model.
        model (Any): The actual model implementation. Must be implemented in child classes.
        model_type (ModelType): The type of model this is.
        data_types (DataTypes): The data types used by this model.
        category_type (CategoryType): The category type used by this model.
    """

    optimizer: OptimizerT
    loss_fn: LossT
    device = None
    dtype = None
    model: Any

    model_type: ModelType
    data_types: DataTypes
    category_type: CategoryType

    def __init__(self: Self, *_args: Any, **_kwargs: Any):
        """
        # Model Initialization

        Initializes a new instance of the Model class. Must be implemented in child classes.

        Args:
            *_args (Any): Variable length argument list.
            **_kwargs (Any): Arbitrary keyword arguments.

        Raises:
            NotImplementedError: This method must be implemented by child classes.
        """
        raise NotImplementedError("Implement Me!")

    def set_optimizer(self: Self, optimizer: OptimizerT) -> Self:
        """
        # Optimizer for Model Training

        Sets the optimizer used for training.

        Args:
            optimizer (OptimizerT): The optimizer to use.

        Returns:
            Self: This model instance.
        """
        self.optimizer = optimizer
        return self

    def set_loss_fn(self: Self, loss_fn: LossT) -> Self:
        """
        # Loss Function for Model Training

        Sets the loss function used for training.

        Args:
            loss_fn (LossT): The loss function to use.

        Returns:
            Self: This model instance.
        """
        self.loss_fn = loss_fn
        return self

    def set_device(self: Self, device: Any) -> Self:
        """
        # Device to put Model On

        Sets the device on which the model will be trained.

        Args:
            device (Any): The device to use.

        Returns:
            Self: This model instance.
        """
        self.device = device
        return self

    def set_dtype(self: Self, dtype: Any) -> Self:
        """
        # Dtype for Model to use

        Sets the data type used by the model.

        Args:
            dtype (Any): The data type to use.

        Returns:
            Self: This model instance.
        """
        self.dtype = dtype
        return self

    def train_epoch(self: Self, *_args: Any, **_kwargs: Any) -> Any:
        """
        # Procedure to Train on Batch

        Trains the model on a batch.

        Args:
            *_args (Any): Variable length argument list.
            **_kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of training the batch.

        Raises:
            NotImplementedError: This method must be implemented by child classes.
        """
        raise NotImplementedError("Implement Me!")

    def eval_epoch(self: Self, *_args: Any, **_kwargs: Any) -> Any:
        """
        # Procedure to Evaluate on Batch

        Evaluates the model on a batch.

        Args:
            *_args (Any): Variable length argument list.
            **_kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of evaluating the batch.

        Raises:
            NotImplementedError: This method must be implemented by child classes.
        """
        raise NotImplementedError("Implement Me!")

    def forward(self: Self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        # Procedure for Forward Pass

        Performs the forward pass of the model.

        Args:
            x (Any): The input to the model.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of the forward pass.
        """
        assert self.model is not None
        return self.model.forward(x, *args, **kwargs)

    def to(self: Self, device: Any | None = None, dtype: Any | None = None) -> Self:
        """
        # Move Model to Device

        Moves the model to a new device and data type.

        Args:
            device (Any | None): The new device to move to. Defaults to None.
            dtype (Any | None): The new data type to use. Defaults to None.

        Returns:
            Self: This model instance.
        """
        if device is not None:
            self.set_device(device)
        if dtype is not None:
            self.set_dtype(dtype)
        return self

    def load(self: Self, checkpoint_path: str) -> Self:
        """
        # Load Model from Checkpoint

        Loads the model from a checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            Self: This model instance.
        """
        raise NotImplementedError("Implement Me!")

    def save(self: Self, checkpoint_path: str) -> Self:
        """
        # Save Model to Checkpoint

        Saves the model to a checkpoint.

        Args:
            checkpoint_path (str): The path to save the checkpoint file.

        Returns:
            Self: This model instance.
        """
        raise NotImplementedError("Implement Me!")
