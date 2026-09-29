# Pandas Extensions
from pandas import DataFrame, Series
from numpy.typing import NDArray
from typing import Any, Dict, Any


def filter_on(self: DataFrame, column: str, value: Any) -> DataFrame | Series | Any:
    """# Filter a Dataframe on Column Value"""
    df = self.copy()
    return df[df[column] == value].reset_index(drop=True)


def get_unique(self, col: str | None = None) -> NDArray[Any]:
    """# Get Unique Values from Series or DataFrame

    For dataframes, a desired column can be provided
    """
    from pandas import DataFrame
    from numpy import unique

    if isinstance(self, DataFrame):
        assert col is not None
        return unique(self[col])
    return unique(self)


def get_enumerate(self: Series) -> Dict[int, Any]:
    """# Enumerate Unique Values in Series"""
    values = self.unique()
    return {v: i for i, v in enumerate(values)}


def convert_timestamp(
    self: DataFrame,
    col: str = "TIMESTAMP",
    col_out: str = "TIME",
    time_format: str = "%Y%m%d-%H%M%S-%f",
    relative: bool = True,
) -> DataFrame:
    """# Convert Timestamp to Int

    Return the new dataframe with new `col_out` column with time as an `int`
    """
    from datetime import datetime
    from numpy import array

    df = self.copy()
    times = array(
        [
            datetime.strptime(df[col].iloc[idx], time_format).timestamp()
            for idx in range(df[col].shape[0])
        ]
    )
    df[col_out] = times
    if relative:
        df[col_out] -= df[col_out].min()
    return df


DataFrame.filter_on = filter_on  # pyright: ignore
DataFrame.unique = get_unique  # pyright: ignore
DataFrame.time = convert_timestamp  # pyright: ignore
Series.unique = get_unique  # pyright: ignore
Series.enumerate = get_enumerate  # pyright: ignore
