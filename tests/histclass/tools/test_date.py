def test_datestring() -> None:
    """# Test Getting Datestring"""
    from histclass.tools.date import get_datestring
    from datetime import datetime

    datestring = get_datestring()
    parts = datestring.split("-")
    assert len(parts) == 3, "Datestring not in 3 parts"
    assert len(parts[0]) == 8, "First part not in YYYYMMDD"
    assert len(parts[1]) == 6, "Second part not in HHMMSS"
    assert len(parts[2]) == 6, "Third part not in us"
    format_str = "%Y%m%d-%H%M%S-%f"
    _parsed = datetime.strptime(datestring, format_str)
    assert True, "Parse threw no errors"
