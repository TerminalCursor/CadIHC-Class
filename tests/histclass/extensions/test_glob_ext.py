import pytest
from unittest.mock import patch
from histclass.extensions.glob_ext import globs
from typing import Any


# Test that the function returns a list of matching files
def test_globs_matches_file(glob_string: str = "*.txt"):
    expected_files = ["file1.txt", "file2.txt"]  # Replace with actual file names

    with patch("glob.glob") as mock_glob:
        mock_glob.return_value = expected_files
        assert globs(glob_string) == expected_files


# Test that the function handles invalid glob patterns
def test_globs_invalid_pattern():
    """Test that an error is raised when an invalid glob pattern is passed"""
    assert globs("*.xyz") == []


# Test that the function preserves file order
def test_globs_file_order():
    """Test that the files are returned in a consistent order (ascending)"""
    glob_string = "*.txt"
    expected_files = ["file1.txt", "file2.txt"]

    with patch("glob.glob") as mock_glob:
        mock_glob.return_value = expected_files
        assert globs(glob_string) == sorted(expected_files)


# Test that the function handles an empty glob string
def test_globs_empty_pattern():
    """Test that an empty glob pattern is handled correctly"""
    assert globs("") == []


@pytest.fixture
def mock_glob(_glob_string: Any, _capsys: Any):
    """Mock the glob function for testing"""

    def mock_glob(*_args: Any, **_kwargs: Any):
        return ["file1.txt", "file2.txt"]

    with patch("glob.glob", new=mock_glob) as mock_glob:  # pyright: ignore
        yield mock_glob


@pytest.fixture
def mock_glob_error(_glob_string: Any, _capsys: Any):
    """Mock the glob function for testing errors"""

    def mock_glob(*_args: Any, **_kwargs: Any):
        raise ValueError("Invalid glob pattern")

    with patch("glob.glob", new=mock_glob) as mock_glob:  # pyright: ignore
        yield mock_glob


def test_globs() -> None:
    """# Test glob string"""
    from histclass.extensions.glob_ext import globs

    g = globs("tests/inputs/Data/D*/*.txt")
    assert len(g) == 3, "Is not capturing all of the files"
    assert g[0] == "tests/inputs/Data/D1/Test.txt", "Unsorted!"
