from ...tools.call_graph import cg
from ..data.types import DataTypes, CategoryType
from ..models.types import ModelType
from .types import LossT, OptimizerT
from typing import Self, Any


class Model:
    """# Model Interface"""

    optimizer: OptimizerT
    loss_fn: LossT
    device = None
    dtype = None
    model: Any

    model_type: ModelType
    data_types: DataTypes
    category_type: CategoryType

    def __init__(self: Self, *_args: Any, **_kwargs: Any):
        """# Model Initialization"""
        cg.add("Model.__init__")
        raise NotImplementedError("Implement Me!")

    def set_optimizer(self: Self, optimizer: OptimizerT) -> Self:
        """# Optimizer for Model Training"""
        cg.add("Model.set_optimizer")
        self.optimizer = optimizer
        return self

    def set_loss_fn(self: Self, loss_fn: LossT) -> Self:
        """# Loss Function for Model Training"""
        cg.add("Model.set_loss_fn")
        self.loss_fn = loss_fn
        return self

    def set_device(self: Self, device: Any) -> Self:
        """# Device to put Model On"""
        cg.add("Model.set_device")
        self.device = device
        return self

    def set_dtype(self: Self, dtype: Any) -> Self:
        """# Dtype for Model to use"""
        cg.add("Model.set_dtype")
        self.dtype = dtype
        return self

    def train_batch(self: Self, *_args: Any, **_kwargs: Any) -> Any:
        """# Procedure to Train on Batch"""
        cg.add("Model.train_batch")
        raise NotImplementedError("Implement Me!")

    def eval_batch(self: Self, *_args: Any, **_kwargs: Any) -> Any:
        """# Procedure to Evaluate on Batch"""
        cg.add("Model.eval_batch")
        raise NotImplementedError("Implement Me!")

    def forward(self: Self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """# Procedure for Forward Pass"""
        cg.add("Model.forward")
        assert self.model is not None
        return self.model.forward(x, *args, **kwargs)

    def to(self: Self, device: Any | None = None, dtype: Any | None = None) -> Self:
        """# Move Model to Device"""
        cg.add("Model.to")
        if device is not None:
            self.set_device(device)
        if dtype is not None:
            self.set_dtype(dtype)
        return self

    def load(self: Self, checkpoint_path: str) -> Self:
        """# Load Model from Checkpoint"""
        cg.add("Model.load")
        raise NotImplementedError("Implement Me!")

    def save(self: Self, checkpoint_path: str) -> Self:
        """# Save Model to Checkpoint"""
        cg.add("Model.save")
        raise NotImplementedError("Implement Me!")
