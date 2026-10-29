import pytest
from unittest.mock import patch
from histclass.tools.results_db import ResultsDatabase

# pyright: reportUnknownMemberType=false,reportUnknownParameterType=false,reportMissingParameterType=false


class TestResultsDatabase:
    @pytest.fixture
    def results_db(self):
        # Create a mock database instance for testing
        with patch("histclass.tools.results_db.Database"):
            db = ResultsDatabase("test.db")
            return db

    def test_init(self, results_db):
        assert results_db.filename == "test.db"
        assert results_db.data == {}
        assert results_db.meta == {}
        assert results_db.scalar_result_keys == [
            "total_accuracy",
            "training_accuracy",
            "validation_accuracy",
            "total_accuracy_c",
            "training_accuracy_c",
            "validation_accuracy_c",
        ]
        assert results_db.matrix_result_keys == [
            "total_raw",
            "training_raw",
            "validation_raw",
        ]
        assert results_db.meta_keys == [
            "timestamp",
            "name",
            "category",
            "learning_rate",
            "loss",
        ]

    def test_add_meta_keys(self, results_db):
        # Test adding meta keys
        results_db.add(
            timestamp="2023-01-01", name="test_run", category="classification"
        )
        assert results_db.meta["timestamp"] == "2023-01-01"
        assert results_db.meta["name"] == "test_run"
        assert results_db.meta["category"] == "classification"

    def test_add_scalar_result_keys(self, results_db):
        # Test adding scalar result keys
        results_db.add(total_accuracy=0.95, training_accuracy=0.92)
        assert results_db.data["total_accuracy"] == 0.95
        assert results_db.data["training_accuracy"] == 0.92

    def test_add_matrix_result_keys(self, results_db):
        # Test adding matrix result keys (should be converted to string)
        mock_array = "array([[1, 2], [3, 4]])"
        results_db.add(total_raw=mock_array)
        assert results_db.data["total_raw"] == "([[1, 2], [3, 4]])"

    def test_add_mixed_keys(self, results_db):
        # Test adding a mix of different key types
        results_db.add(
            timestamp="2023-01-01", total_accuracy=0.95, total_raw="array([[1, 2]])"
        )
        assert results_db.meta["timestamp"] == "2023-01-01"
        assert results_db.data["total_accuracy"] == 0.95
        assert results_db.data["total_raw"] == "([[1, 2]])"

    def test_add_nonexistent_keys(self, results_db):
        # Test adding keys that don't belong to any category
        results_db.add(some_random_key="random_value")
        assert "some_random_key" not in results_db.data
        assert "some_random_key" not in results_db.meta

    def test_write_with_all_meta_keys(self, results_db):
        # Test writing with all meta keys populated
        results_db.add(
            timestamp="2023-01-01",
            name="test_run",
            category="classification",
            learning_rate=0.001,
            loss=0.15,
        )

        assert results_db.meta["timestamp"] == "2023-01-01"
        assert results_db.meta["name"] == "test_run"
        assert results_db.meta["category"] == "classification"
        assert results_db.meta["learning_rate"] == 0.001
        assert results_db.meta["loss"] == 0.15
