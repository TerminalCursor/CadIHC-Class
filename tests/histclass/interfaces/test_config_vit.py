from histclass.interfaces.config import ViTConfig


class TestViTConfig:
    def test_default_values(self):
        """Test that default values are correctly set."""
        config = ViTConfig()
        assert config.image_size == 1
        assert config.patch_size == 16
        assert config.num_layers == 4
        assert config.num_heads == 4
        assert config.hidden_dim == 16
        assert config.mlp_dim == 16
        assert config.dropout == 0.05
        assert config.attention_dropout == 0.05

    def test_override_defaults(self):
        """Test that default values can be overridden."""
        config = ViTConfig(
            image_size=224,
            patch_size=32,
            num_layers=12,
            num_heads=8,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=0.1,
            attention_dropout=0.1,
        )
        assert config.image_size == 224
        assert config.patch_size == 32
        assert config.num_layers == 12
        assert config.num_heads == 8
        assert config.hidden_dim == 768
        assert config.mlp_dim == 3072
        assert config.dropout == 0.1
        assert config.attention_dropout == 0.1

    def test_partial_override(self):
        """Test that only specified values are overridden."""
        config = ViTConfig(image_size=224, patch_size=32)
        assert config.image_size == 224
        assert config.patch_size == 32
        # Check that other values remain default
        assert config.num_layers == 4
        assert config.num_heads == 4
        assert config.hidden_dim == 16
        assert config.mlp_dim == 16
        assert config.dropout == 0.05
        assert config.attention_dropout == 0.05

    def test_to_dict(self):
        """Test that to_dict method returns correct dictionary."""
        config = ViTConfig(
            image_size=224,
            patch_size=32,
            num_layers=12,
            num_heads=8,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=0.1,
            attention_dropout=0.1,
        )
        expected_dict = {
            "image_size": 224,
            "patch_size": 32,
            "num_layers": 12,
            "num_heads": 8,
            "hidden_dim": 768,
            "mlp_dim": 3072,
            "dropout": 0.1,
            "attention_dropout": 0.1,
        }
        assert config.to_dict() == expected_dict

    def test_empty_init(self):
        """Test that empty initialization works correctly."""
        config = ViTConfig()
        assert config.image_size == 1
        assert config.patch_size == 16
        assert config.num_layers == 4
        assert config.num_heads == 4
        assert config.hidden_dim == 16
        assert config.mlp_dim == 16
        assert config.dropout == 0.05
        assert config.attention_dropout == 0.05

    def test_to_dict_consistency(self):
        """Test that to_dict returns a copy and doesn't reference the same object."""
        config = ViTConfig(image_size=224)
        dict_result = config.to_dict()
        dict_result["image_size"] = 999
        # Original config should not be affected
        assert config.image_size == 224
