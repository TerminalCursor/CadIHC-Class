import pytest
from histclass.interfaces.data.types.categorytype import CategoryType


def test_category_type_values():
    """# Test Category Type Values"""
    expected_values = {
        CategoryType.TripleNegative: ["NTN", "TN"],
        CategoryType.BasicStage: ["I", "II", "III"],
        CategoryType.AdvancedStage: [
            "IA",
            "IIA",
            "IIB",
            "IIIC",
            "IIIB",
            "IIIC",
        ],
        CategoryType.Grade: ["One", "Two", "Three"],
        CategoryType.BinaryStage: ["Early", "Late"],
        CategoryType.BinaryGrade: ["Early", "Late"],
    }
    for category, expected in expected_values.items():
        cvs = category.values()
        assert cvs is not None
        assert set(cvs) == set(expected)


def test_category_type_to_str():
    """# Test String Representation"""
    expected = {
        CategoryType.TripleNegative: "tn",
        CategoryType.BasicStage: "stage_basic",
        CategoryType.AdvancedStage: "stage_advanced",
        CategoryType.Grade: "grade",
        CategoryType.BinaryStage: "hilostage",
        CategoryType.BinaryGrade: "hilograde",
    }
    for category, expected_value in expected.items():
        assert (
            CategoryType(category).to_str() == expected_value
        ), f"Expected {expected_value} but got {CategoryType(category).to_str()}"


def test_category_type_unknown():
    """# Test Unknown Category Type"""
    with pytest.raises(AssertionError):
        CategoryType.Unknown.to_str()
    with pytest.raises(AssertionError):
        CategoryType.Unknown.values()


def test_category_type_values_none_returned():
    """# Test Values for Unknown Category Type"""
    with pytest.raises(AssertionError):
        assert CategoryType.Unknown.values() is None, "Expected None but got a list"


def test_category_type_order():
    """# Test Order of Enum Members"""
    # This test is more about the structure than actual value
    # but it helps to ensure that no new members are added out of order.
    expected_members = [
        CategoryType.Unknown,
        CategoryType.TripleNegative,
        CategoryType.BasicStage,
        CategoryType.AdvancedStage,
        CategoryType.Grade,
        CategoryType.BinaryStage,
        CategoryType.BinaryGrade,
    ]
    assert list(CategoryType) == [
        member for member in expected_members
    ], "Order of Enum Members is not correct"
