from ..tools.call_graph import cg


def globs(glob_string: str):
    """# Get the matching files from glob string"""
    cg.add("globs")
    from glob import glob as _glob

    return sorted(_glob(glob_string))
