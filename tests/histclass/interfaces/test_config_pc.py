from histclass.interfaces.config import PointcloudConfig


class TestPointcloudConfig:
    def test_default_initialization(self):
        """
        Test that the PointcloudConfig initializes with default values.
        """
        config = PointcloudConfig()
        assert config.max_points == 96
        assert config.min_points == 64
        assert config.dropout == 0.15
        assert config.dropout_output is False
        assert config.n_ff_nodes == 32
        assert config.n_ff_layers == 3
        assert config.n_blocks == 1

    def test_custom_initialization(self):
        """
        Test that the PointcloudConfig can be initialized with custom values.
        """
        custom_config = {
            "max_points": 128,
            "min_points": 32,
            "dropout": 0.3,
            "dropout_output": True,
            "n_ff_nodes": 64,
            "n_ff_layers": 5,
            "n_blocks": 2,
        }
        config = PointcloudConfig(**custom_config)
        assert config.max_points == custom_config["max_points"]
        assert config.min_points == custom_config["min_points"]
        assert config.dropout == custom_config["dropout"]
        assert config.dropout_output == custom_config["dropout_output"]
        assert config.n_ff_nodes == custom_config["n_ff_nodes"]
        assert config.n_ff_layers == custom_config["n_ff_layers"]
        assert config.n_blocks == custom_config["n_blocks"]

    def test_partial_initialization(self):
        """
        Test that the PointcloudConfig can be initialized with only some values.
        """
        config = PointcloudConfig(max_points=200, dropout=0.5)
        assert config.max_points == 200
        assert config.min_points == 64  # default value
        assert config.dropout == 0.5
        assert config.dropout_output is False  # default value
        assert config.n_ff_nodes == 32  # default value
        assert config.n_ff_layers == 3  # default value
        assert config.n_blocks == 1  # default value

    def test_to_dict(self):
        """
        Test that the to_dict method returns a correct dictionary representation.
        """
        config = PointcloudConfig(
            max_points=100,
            min_points=50,
            dropout=0.2,
            dropout_output=True,
            n_ff_nodes=40,
            n_ff_layers=4,
            n_blocks=2,
        )
        expected_dict = {
            "max_points": 100,
            "min_points": 50,
            "dropout": 0.2,
            "dropout_output": True,
            "n_ff_nodes": 40,
            "n_ff_layers": 4,
            "n_blocks": 2,
        }
        assert config.to_dict() == expected_dict

    def test_to_dict_default_values(self):
        """
        Test that to_dict returns default values when no custom values are set.
        """
        config = PointcloudConfig()
        expected_dict = {
            "max_points": 96,
            "min_points": 64,
            "dropout": 0.15,
            "dropout_output": False,
            "n_ff_nodes": 32,
            "n_ff_layers": 3,
            "n_blocks": 1,
        }
        assert config.to_dict() == expected_dict
