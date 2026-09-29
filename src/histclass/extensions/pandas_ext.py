# Pandas Extensions
from pandas import DataFrame, Series  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from typing import Any, Dict
from pandera.typing import Series as SeriesT
from ..typecheck import *


def filter_on(self: DataFrame, column: str, value: Any) -> Series | DataFrame:
    """
    Filter a DataFrame on column values.

    Args:
        self (DataFrame): Input DataFrame
        column (str): Column name to filter
        value (Any): Value to filter

    Returns:
        Series | DataFrame: Filtered series or DataFrame with filtered values
    """
    df = self.copy()
    _filter: Series = wash2series(df[column] == value)
    filtered = wash2df(df[_filter]).reset_index(drop=True)
    return wash2strseries(filtered)


def get_unique(self: DataFrame | Series, column: str | None = None) -> NDArray[Any]:
    """
    Get unique values from a series or DataFrame.

    For dataframes, a desired column can be provided.

    Args:
        self (DataFrame | Series): Input series or DataFrame
        column (str | None, optional): Desired column name. Defaults to None.

    Returns:
        NDArray[Any]: Unique values as numpy array
    """
    from numpy import unique

    if isinstance(self, DataFrame):
        assert column is not None
        return unique(wash2series(self[column]))
    return unique(self)


def get_enumerate(self: Series) -> Dict[Any, int]:
    """
    Enumerate unique values in a series.

    Args:
        self (Series): Input series

    Returns:
        Dict[Any, int]: Dictionary with enumated unique values
    """
    values = self.unique()
    return {v: i for i, v in enumerate(values)}


def convert_timestamp(
    self: DataFrame,
    col: str = "TIMESTAMP",
    col_out: str = "TIME",
    time_format: str = "%Y%m%d-%H%M%S-%f",
    relative: bool = True,
) -> DataFrame:
    """
    Convert timestamp column to float.

    Return the new DataFrame with a new `col_out` column containing time as an float.

    Args:
        self (DataFrame): Input DataFrame
        col (str, optional): Timestamp column name. Defaults to "TIMESTAMP".
        col_out (str, optional): Output column name. Defaults to "TIME".
        time_format (str, optional): Time format string. Defaults to "%Y%m%d-%H%M%S-%f".
        relative (bool, optional): Whether to make values relative. Defaults to True.

    Returns:
        DataFrame: Output DataFrame with converted timestamp column
    """
    from datetime import datetime
    from numpy import array

    df = self.copy()
    column_data: SeriesT[str] = wash2strseries(df[col])
    times = array(
        [
            datetime.strptime(wash2str(column_data.iloc[idx]), time_format).timestamp()
            for idx in range(column_data.shape[0])
        ]
    )
    df[col_out] = times
    if relative:
        df[col_out] -= df[col_out].min()  # pyright: ignore
    return df


# Register custom methods onto pandas DataFrame and Series Classes
DataFrame.filter_on = filter_on  # pyright: ignore
DataFrame.unique = get_unique  # pyright: ignore
DataFrame.time = convert_timestamp  # pyright: ignore
Series.unique = get_unique  # pyright: ignore
Series.enumerate = get_enumerate  # pyright: ignore
