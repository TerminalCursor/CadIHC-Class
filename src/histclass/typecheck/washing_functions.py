from typing import Any, List
from pandas import DataFrame, Series  # pyright: ignore[reportMissingTypeStubs]
from pandera.typing import Series as SeriesT


def wash2intseries(x: Any) -> SeriesT[int]:
    """
    Washing function to convert input series to int type.

    Args:
        x (Any): Input series

    Returns:
        SeriesT[int]: Output series with int type
    """
    return x


def wash2strseries(x: Any) -> SeriesT[str]:
    """
    Washing function to convert input series to str type.

    Args:
        x (Any): Input series

    Returns:
        SeriesT[str]: Output series with str type
    """
    return x


def wash2str(x: Any) -> str:
    """
    Washing function to convert single value to str type.

    Args:
        x (Any): Single input value

    Returns:
        str: Output str value
    """
    return x


def wash2df(x: Any) -> DataFrame:
    """
    Washing function to convert input series or scalar to DataFrame.

    Args:
        x (Any): Input series or scalar

    Returns:
        DataFrame: Output DataFrame
    """
    return x


def wash2series(x: Any) -> Series:
    """
    Washing function to convert input series or scalar to DataFrame.

    Args:
        x (Any): Input series or scalar

    Returns:
        Series: Output Series
    """
    return x


def wash2strlist(x: Any) -> List[str]:
    """
    Washing function to convert input to list of strings type.

    Args:
        x (Any): Input list

    Returns:
        List[str]: Output List of Strings
    """
    return x
