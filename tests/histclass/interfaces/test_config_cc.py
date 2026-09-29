from unittest.mock import Mock
from histclass.interfaces.config import CompleteConfig
from histclass.interfaces.data.types import DataTypes, CategoryType
from histclass.interfaces.models.types import ModelType


class TestCompleteConfig:
    def test_init_with_image_cnn_data_type(self):
        # Mock the required enums and configs
        data_types = DataTypes.ImageCNN
        category_type = CategoryType.TripleNegative
        model_type = ModelType.Attention
        config = CompleteConfig(
            input_data_types=data_types,
            category_type=category_type,
            model_type=model_type,
        )
        assert config.pointcloud_config is None
        assert config.vision_transformer_config is None

    def test_init_with_image_vit_data_type(self):
        data_types = DataTypes.ImageViT
        category_type = CategoryType.TripleNegative
        model_type = ModelType.Attention

        # Test with vit config in kwargs
        config = CompleteConfig(
            input_data_types=data_types,
            category_type=category_type,
            model_type=model_type,
            cnn={"some": "param"},
            vit={"image_size": 224},
        )

        assert config.model_name == "image_vt-attention2tn"
        assert config.image_values is not None
        assert config.vision_transformer_config is not None

    def test_init_with_pointcloud_data_type(self):
        mock_data_types = Mock()
        mock_data_types.is_valid.return_value = True
        mock_data_types.to_str.return_value = "PointCloud"

        mock_category_type = Mock()
        mock_category_type.to_str.return_value = "Category"

        mock_model_type = Mock()
        mock_model_type.to_str.return_value = "Attention"

        config = CompleteConfig(
            input_data_types=mock_data_types,
            category_type=mock_category_type,
            model_type=mock_model_type,
            pc={"some": "param"},
        )

        assert config.model_name == "PointCloud-Attention2Category"
        assert config.pointcloud_config is not None
        assert config.image_values is None
        assert config.vision_transformer_config is None

    def test_init_default_values(self):
        mock_data_types = Mock()
        mock_data_types.is_valid.return_value = True
        mock_data_types.to_str.return_value = "ImageCNN"

        mock_category_type = Mock()
        mock_category_type.to_str.return_value = "Category"

        mock_model_type = Mock()
        mock_model_type.to_str.return_value = "Attention"

        config = CompleteConfig(
            input_data_types=mock_data_types,
            category_type=mock_category_type,
            model_type=mock_model_type,
        )

        assert config.epochs == 20000
        assert config.learning_rate == 5e-06
        assert config.device == "cuda"
        assert config.batch_size == 8
        assert config.train_val_split == 0.8
        assert config.loss_fn == "mse_loss"
        assert config.optimizer == "Adam"

    def test_to_dict_method(self):
        # Mock all required objects
        data_types = DataTypes.ImageCNN
        category_type = CategoryType.TripleNegative
        model_type = ModelType.Attention

        # Create a minimal config
        config = CompleteConfig(
            input_data_types=data_types,
            category_type=category_type,
            model_type=model_type,
        )

        result_dict = config.to_dict()

        assert "model_name" in result_dict
        assert "model_type" in result_dict
        assert "input_data_types" in result_dict
        assert "output_data_type" in result_dict
        assert "epochs" in result_dict
        assert "learning_rate" in result_dict
        assert "loss_fn" in result_dict
        assert "batch_size" in result_dict
        assert "train_val_split" in result_dict
        assert "device" in result_dict
        assert "early_exit_config" in result_dict
