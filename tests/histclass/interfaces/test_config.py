# tests/test_config.py
import pytest
from histclass.interfaces.models.types import ModelType
from histclass.interfaces.data.types import DataTypes, CategoryType
from histclass.interfaces.config import (
    CompleteConfig,
    PointcloudConfig,
    CnnConfig,
    ViTConfig,
    EarlyExitConfig,
)


class TestConfig:
    def test_complete_config_init(self):
        config = CompleteConfig(
            input_data_types=DataTypes.ImageCNN,
            category_type=CategoryType.TripleNegative,
            model_type=ModelType.Attention,
            loss_fn="mse_loss",
            optimizer="Adam",
        )
        assert config.model_name
        assert config.model_type == ModelType.Attention
        assert config.input_data_types == DataTypes.ImageCNN
        assert config.output_data_type == CategoryType.TripleNegative
        assert config.epochs > 0
        assert config.learning_rate > 0

    def test_complete_config_to_dict(self):
        config = CompleteConfig(
            input_data_types=DataTypes.ImageCNN,
            category_type=CategoryType.BasicStage,
            model_type=ModelType.Attention,
            loss_fn="mse_loss",
            optimizer="Adam",
        )
        config_dict = config.to_dict()
        assert len(config_dict) == 13
        assert "model_name" in config_dict
        assert "early_exit_config" in config_dict

    def test_pointcloud_config_init(self):
        config = PointcloudConfig(max_points=100, dropout=0.1)
        assert config.max_points == 100
        assert config.dropout == 0.1

    def test_pointcloud_config_to_dict(self):
        config = PointcloudConfig(max_points=100, dropout=0.1)
        config_dict = config.to_dict()
        assert len(config_dict) == 7
        assert "max_points" in config_dict
        assert "dropout" in config_dict

    def test_cnn_config_init(self):
        config = CnnConfig(window_width=1920, window_height=2560)
        assert config.window_width == 1920
        assert config.window_height == 2560

    def test_cnn_config_to_dict(self):
        config = CnnConfig(window_width=1920, window_height=2560)
        config_dict = config.to_dict()
        assert len(config_dict) == 2
        assert "window_width" in config_dict
        assert "window_height" in config_dict

    def test_vit_config_init(self):
        config = ViTConfig(image_size=224, patch_size=32)
        assert config.image_size == 224
        assert config.patch_size == 32

    def test_vit_config_to_dict(self):
        config = ViTConfig(image_size=224, patch_size=32)
        config_dict = config.to_dict()
        assert len(config_dict) == 8
        assert "image_size" in config_dict
        assert "patch_size" in config_dict

    def test_early_exit_config_to_dict(self):
        config = EarlyExitConfig()
        config_dict = config.to_dict()
        assert len(config_dict) == 2
        assert "kwargs" in config_dict
        assert "max_failures" in config_dict


# tests/test_data_types.py
from histclass.interfaces.models.types import ModelType
from histclass.interfaces.data.types import DataTypes


def test_data_types_valid():
    # Test that data types are valid
    data_types = [DataTypes.ImageCNN, DataTypes.ImageViT]
    for data_type in data_types:
        assert isinstance(data_type.value, int)


# tests/test_models.py
from histclass.interfaces.models.types import ModelType


@pytest.fixture
def model_type():
    return ModelType.Attention


def test_model_type(model_type: ModelType):
    # Test that model types are valid
    assert model_type == ModelType.Attention
