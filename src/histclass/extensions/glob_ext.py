from typing import List


def globs(glob_string: str) -> List[str]:
    """# Get the matching files from glob string"""
    from glob import glob as _glob

    return sorted(_glob(glob_string))
