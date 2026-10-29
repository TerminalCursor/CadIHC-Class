from histclass.interfaces.config import CnnConfig


class TestCnnConfig:
    def test_default_initialization(self):
        """
        Test that CnnConfig initializes with default values.
        """
        config = CnnConfig()
        assert config.window_width == 1920
        assert config.window_height == 2560

    def test_custom_initialization(self):
        """
        Test that CnnConfig can be initialized with custom values.
        """
        config = CnnConfig(window_width=1024, window_height=768)
        assert config.window_width == 1024
        assert config.window_height == 768

    def test_partial_initialization(self):
        """
        Test that CnnConfig can be initialized with only one parameter.
        """
        config = CnnConfig(window_width=512)
        assert config.window_width == 512
        assert config.window_height == 2560  # Should keep default

    def test_to_dict(self):
        """
        Test that to_dict returns correct dictionary representation.
        """
        config = CnnConfig(window_width=1024, window_height=768)
        result = config.to_dict()
        expected = {"window_width": 1024, "window_height": 768}
        assert result == expected

    def test_to_dict_default_values(self):
        """
        Test that to_dict returns correct dictionary with default values.
        """
        config = CnnConfig()
        result = config.to_dict()
        expected = {"window_width": 1920, "window_height": 2560}
        assert result == expected

    def test_to_dict_with_kwargs(self):
        """
        Test that to_dict works correctly when initialized with kwargs.
        """
        config = CnnConfig(**{"window_width": 800, "window_height": 600})
        result = config.to_dict()
        expected = {"window_width": 800, "window_height": 600}
        assert result == expected

    def test_empty_kwargs(self):
        """
        Test that CnnConfig handles empty kwargs correctly.
        """
        config = CnnConfig(**{})
        assert config.window_width == 1920
        assert config.window_height == 2560
        result = config.to_dict()
        expected = {"window_width": 1920, "window_height": 2560}
        assert result == expected
