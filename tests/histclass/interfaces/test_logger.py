from histclass.interfaces.db import Database
from pytest import Subtests
from typing import Self, Dict, Any


class TempDatabase(Database):
    """# Test Database Class

    For testing logger
    """

    data: Dict[str, Any] = {}

    def __init__(self: Self, filename: str):
        super().__init__(filename)
        self.data = {}

    def add(self: Self, **kwargs: Any) -> Self:
        """# Add to log"""
        valid_keys = ["test_1", "test_2", "test_3"]
        for k, v in kwargs.items():
            if k in valid_keys:
                self.data[k] = v
        return self

    def write(self: Self, *args: Any, **kwargs: Any) -> Self:
        if "echo" not in kwargs.keys():
            return self
        if kwargs["echo"]:
            print(self.data)
        return self


def test_logger(subtests: Subtests) -> None:
    """# Test Logger"""
    from histclass.interfaces.logger import Logger

    bpath = "tests/outputs/"
    log = Logger(bpath, echo_stdout=False)
    with subtests.test("Logger Initialization"):
        assert log.timestamp != "", "Logger timestamp should be initialized"
        assert not log.echo_stdout, "Logger should not echo for tests"
        assert log.enabled, "Logger should be enabled by default"
    with subtests.test("Logger Database Initialization"):
        log.db = TempDatabase(bpath + "Log.db")
        assert log.db.filename == bpath + "Log.db"
    total_data: str = ""
    with subtests.test("Logger add"):
        data = "Log data here\n"
        total_data += data
        log.add(data)
        assert log.file.read() == data, "Log data inconsistent"
    with subtests.test("Logger addnl"):
        data = "More data for all"
        total_data += data
        log.addnl(data)
        assert log.file.read() == total_data + "\n", "Log data inconsistent"
    with subtests.test("Logger db add"):
        assert isinstance(log.db, TempDatabase)
        log.db_add(test_1=1, test_2="a", test_3=5.2, test_4=False)
        assert "test_1" in log.db.data.keys(), "Log failed to add data"
        assert "test_2" in log.db.data.keys(), "Log failed to add data"
        assert "test_3" in log.db.data.keys(), "Log failed to add data"
        assert "test_4" not in log.db.data.keys(), "Log added extra data"
        assert log.db.data["test_1"] == 1
        assert log.db.data["test_2"] == "a"
        assert log.db.data["test_3"] == 5.2
        log.echo_stdout = True
        log.db_write()
        assert True, "Should dump data to stdout"
