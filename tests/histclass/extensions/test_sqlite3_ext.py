# Import necessary modules
from histclass.extensions.sqlite3_ext import *
from histclass.interfaces.data.types import DataTypes
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]


# Function to test wash2strlist
def test_wash2strlist():
    """Test wash2strlist function"""
    # Test with integer list
    result = wash2strlist([1, 2, 3])
    assert isinstance(result, list), "Expected list"
    for item in result:
        assert isinstance(item, int), "Expected integers"

    # Test with string list
    result = wash2strlist(["a", "b", "c"])
    assert isinstance(result, list), "Expected list"
    for item in result:
        assert isinstance(item, str), "Expected strings"


# Function to test legacy_enumerate_model
def test_legacy_enumerate_model():
    """Test legacy_enumerate_model function"""
    # Create a sample series with different types of data
    series = pd.Series(["image", "image_vt256", "morphology", "latent", "energetics"])
    dtypes, extras = legacy_enumerate_model(series)
    for dtype in dtypes:
        assert DataTypes(dtype).is_valid()
    for idx, ex in enumerate(extras):
        if idx == 1:
            assert ex == 256
        else:
            assert ex == -1
