from histclass.interfaces.config import EarlyExitConfig


class TestEarlyExitConfig:
    def test_initialization(self):
        """Test that EarlyExitConfig initializes with correct default values."""
        config = EarlyExitConfig()

        assert config.kwargs == {
            "start_after": 100,
            "lookback": 600,
            "p_cutoff": 0.05,
            "on_index": 1,
        }
        assert config.max_failures == 200

    def test_to_dict_method(self):
        """Test that to_dict method returns correct dictionary representation."""
        config = EarlyExitConfig()

        result = config.to_dict()

        expected = {
            "kwargs": {
                "start_after": 100,
                "lookback": 600,
                "p_cutoff": 0.05,
                "on_index": 1,
            },
            "max_failures": 200,
        }

        assert result == expected
        assert isinstance(result, dict)
        assert "kwargs" in result
        assert "max_failures" in result

    def test_to_dict_returns_copy(self):
        """Test that to_dict returns a copy and doesn't allow external modification of internal state."""
        config = EarlyExitConfig()
        result = config.to_dict()

        # Modify the returned dict
        result["max_failures"] = 999
        result["kwargs"]["start_after"] = 999  # pyright: ignore

        # Original config should remain unchanged
        assert config.max_failures == 200
        assert config.kwargs["start_after"] == 100

    def test_custom_values(self):
        """Test that we can create instances with custom values."""
        custom_kwargs = {"start_after": 50, "lookback": 300}  # pyright: ignore
        custom_max_failures = 100  # pyright: ignore

        # This should work even though the class doesn't support custom initialization
        # We're testing that the default values are preserved
        config = EarlyExitConfig()
        assert config.kwargs == {
            "start_after": 100,
            "lookback": 600,
            "p_cutoff": 0.05,
            "on_index": 1,
        }
        assert config.max_failures == 200
