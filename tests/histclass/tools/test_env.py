def test_env() -> None:
    """# Test Getting Environment Variables"""
    from histclass.tools.env import get_env

    assert get_env("_THIS_DOES_NOT_EXIST").data is None
    assert get_env("PATH").data is not None
