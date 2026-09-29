def test_db() -> None:
    """# Test Database Interface"""
    from histclass.interfaces.db import Database

    fname = "tests/outputs/DbTest.db"
    db = Database(fname)
    assert db.filename == fname, "Filename set"
