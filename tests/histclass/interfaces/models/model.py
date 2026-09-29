import pytest
from histclass.interfaces.models.model import Model
from typing import Any


def test_model_default_attributes():
    """# Model Default Attributes Test"""
    model = Model()
    assert model.optimizer is None
    assert model.loss_fn is None
    assert model.device is None
    assert model.dtype is None
    assert model.model is None
    assert model.model_type is None
    assert model.data_types is None
    assert model.category_type is None


def test_model_init_not_implemented():
    """# Model Initialization Not Implemented Test"""
    with pytest.raises(NotImplementedError):
        Model()


def test_set_device():
    """# Device to put Model On Test"""
    device = "cuda"  # Replace with actual device type
    model = Model()
    result_model = model.set_device(device)
    assert result_model.device == device


def test_set_dtype():
    """# Dtype for Model to use Test"""
    dtype = "float32"  # Replace with actual data type
    model = Model()
    result_model = model.set_dtype(dtype)
    assert result_model.dtype == dtype


def test_train_epoch_not_implemented():
    """# Procedure to Train on Batch Not Implemented Test"""
    with pytest.raises(NotImplementedError):
        Model().train_epoch()


def test_eval_epoch_not_implemented():
    """# Procedure to Evaluate on Batch Not Implemented Test"""
    with pytest.raises(NotImplementedError):
        Model().eval_epoch()


def test_forward_pass():
    """# Procedure for Forward Pass Test"""

    class MyModel:
        def __call__(self, x: Any) -> Any:
            return "Forward Pass"

    model = Model()
    my_model = MyModel()
    model.model = my_model
    result = model.forward("Input", *(), **{})
    assert isinstance(result, str)


def test_to_device():
    """# Move Model to Device Test"""
    device = "simd"  # Replace with actual device type
    model = Model()
    result_model = model.to(device=device)
    assert result_model.device == device


def test_load_checkpoint_not_implemented():
    """# Load Model from Checkpoint Not Implemented Test"""
    with pytest.raises(NotImplementedError):
        Model().load("Checkpoint")


def test_save_checkpoint_not_implemented():
    """# Save Model to Checkpoint Not Implemented Test"""
    with pytest.raises(NotImplementedError):
        Model().save("Checkpoint")
