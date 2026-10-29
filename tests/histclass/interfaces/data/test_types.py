def test_CategoryTypeValues():
    """# Test Category Type Value Counts"""
    from histclass.interfaces.data.types import CategoryType

    results = {
        CategoryType.TripleNegative: 2,
        CategoryType.BasicStage: 3,
        CategoryType.AdvancedStage: 6,
        CategoryType.Grade: 3,
        CategoryType.BinaryStage: 2,
        CategoryType.BinaryGrade: 2,
    }
    for _type in CategoryType:
        if _type == CategoryType.Unknown:
            continue
        _values = _type.values()
        assert _values is not None
        assert len(_values) == results[_type]


def test_DataTypeCombinations():
    """# Test Data Type Valid Combinations"""
    from histclass.interfaces.data.types import DataTypes
    from itertools import combinations
    from operator import or_
    from functools import reduce
    from typing import List, Tuple

    valid_two_combos = [
        DataTypes.Morphology | DataTypes.Nuclei,
        DataTypes.Morphology | DataTypes.Energetics,
    ]
    combos: List[Tuple[DataTypes, bool]] = []
    for N in range(1, len(DataTypes) + 1):
        for _types in combinations(DataTypes, N):
            _type = reduce(or_, _types)
            if (_type in valid_two_combos) or N == 1:
                combos.append((_type, True))
            else:
                combos.append((_type, False))
    for dtype, is_valid in combos:
        valid_s = "Valid" if is_valid else "Invalid"
        assert dtype.is_valid() == is_valid, f"Combination {dtype} should be {valid_s}"
