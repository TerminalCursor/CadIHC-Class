from pandas import DataFrame, Series  # pyright: ignore[reportMissingTypeStubs]
from numpy import ndarray
from histclass.extensions.pandas_ext import *


def test_filter_on_column_value():
    """Test filter_on function"""
    df = DataFrame({"A": [1, 2, 3], "B": ["hello", "world", "test"]})
    filtered_df = filter_on(df, "A", 2)
    assert filtered_df.shape[0] == 1
    assert filtered_df.shape[1] == 2


def test_filter_on_column_value_all():
    """Test filter_on function with all rows"""
    df = DataFrame({"A": [1, 2, 3], "B": ["hello", "world", "test"]})
    filtered_df = filter_on(df, "B", "")
    assert filtered_df.shape[0] == 0


def test_filter_on_column_value_none():
    """Test filter_on function with None value"""
    df = DataFrame({"A": [1, 2, 3], "B": ["hello", "world", "test"]})
    filtered_df = filter_on(df, "B", None)
    assert filtered_df.shape[0] == 0


def test_get_unique_series():
    """Test get_unique function"""
    series = Series([4, 5, 6])
    result = get_unique(series)
    assert isinstance(result, ndarray)


def test_get_unique_dataframe_column():
    """Test get_unique function with DataFrame and column"""
    df = DataFrame({"A": [1, 2, 3]})
    result = get_unique(df, "A")
    assert isinstance(result, ndarray)


def test_get_enumerate_series():
    """Test get_enumerate function"""
    series = Series([4, 5, 6])
    result = get_enumerate(series)
    assert isinstance(result, Dict)


def test_convert_timestamp_column_out():
    """Test convert_timestamp function with column out"""
    df = DataFrame({"TIMESTAMP": ["20220113-221201-100000", "20220115-221201-100000"]})
    result = convert_timestamp(df, col="TIMESTAMP")
    assert result.shape[0] == 2
    assert result.shape[1] == 2


def test_convert_timestamp_time_format():
    """Test convert_timestamp function with time format"""
    df = DataFrame({"TIMESTAMP": ["20220113-221201-100000"]})
    result = convert_timestamp(df)
    assert isinstance(result["TIME"].iloc[0], float)  # pyright: ignore


def test_convert_timestamp_relative():
    """Test convert_timestamp function with relative"""
    df = DataFrame({"TIMESTAMP": ["20220113-221201-100000", "20220115-221201-100000"]})
    result = convert_timestamp(df, relative=False)
    assert isinstance(result["TIME"].iloc[0], float)  # pyright: ignore
