# Pandas Extensions
from pandas import DataFrame, Series  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from typing import Any, Dict
from pandera.typing import Series as SeriesT


def wash2intseries(x: Any) -> SeriesT[int]:
    """# Washing function for Type Checking"""
    return x


def wash2strseries(x: Any) -> SeriesT[str]:
    """# Washing function for Type Checking"""
    return x


def wash2str(x: Any) -> str:
    """# Washing function for Type Checking"""
    return x


def wash2df(x: Any) -> DataFrame:
    """# Washing Function for Type Checking"""
    return x


def wash2series(x: Any) -> Series:
    """# Washing Function for Type Checking"""
    return x


def filter_on(self: DataFrame, column: str, value: Any) -> Series | DataFrame:
    """# Filter a Dataframe on Column Value"""
    df = self.copy()
    _filter: Series = wash2series(df[column] == value)
    filtered = wash2df(df[_filter]).reset_index(drop=True)
    return wash2strseries(filtered)


def get_unique(self: DataFrame | Series, col: str | None = None) -> NDArray[Any]:
    """# Get Unique Values from Series or DataFrame

    For dataframes, a desired column can be provided
    """
    from numpy import unique

    if isinstance(self, DataFrame):
        assert col is not None
        return unique(wash2series(self[col]))
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


DataFrame.filter_on = filter_on  # pyright: ignore
DataFrame.unique = get_unique  # pyright: ignore
DataFrame.time = convert_timestamp  # pyright: ignore
Series.unique = get_unique  # pyright: ignore
Series.enumerate = get_enumerate  # pyright: ignore
