def test_globs() -> None:
    """# Test glob string"""
    from histclass.extensions.glob_ext import globs

    g = globs("tests/inputs/Data/D*/*.txt")
    assert len(g) == 3, "Is not capturing all of the files"
    assert g[0] == "tests/inputs/Data/D1/Test.txt", "Unsorted!"
