from typing import List


def globs(glob_string: str) -> List[str]:
    """
    Sort the matches of a glob string

    Args:
        glob_string (str): Glob String

    Returns:
        List[str]: Sorted list of globbing matches
    """
    from glob import glob as _glob

    return sorted(_glob(glob_string))
