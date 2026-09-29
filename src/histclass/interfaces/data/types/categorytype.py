from ....tools.call_graph import cg
from enum import Enum
from typing import List, Self


class CategoryType(Enum):
    """# Category Types for Model Output

    TripleNegative: Whether the Sample is Triple Negative
    BasicStage: Whether the Sample should be classified into Stages Coarsely (I, II, ...)
    AdvancedStage: Whether the Sample should be classified into Stages Finely (IA, IIA, ...)
    Grade: Whether the Sample should be classified into Grades (1, 2, ...)
    """

    Unknown = 0
    TripleNegative = 1
    BasicStage = 2
    AdvancedStage = 3
    Grade = 4

    def to_str(self: Self) -> str:
        """# String Representation"""
        cg.add("CategoryType.to_str")
        if self == CategoryType.TripleNegative:
            return "tn"
        elif self == CategoryType.BasicStage:
            return "stage_basic"
        elif self == CategoryType.AdvancedStage:
            return "stage_advanced"
        elif self == CategoryType.Grade:
            return "grade"
        assert False, "Undefined Type!"  # pyright: ignore[reportUnreachable]

    def values(self: Self) -> List[str] | None:
        """# Category Values"""
        cg.add("CategoryType.values")
        if self == CategoryType.TripleNegative:
            return ["NTN", "TN"]
        elif self == CategoryType.BasicStage:
            return ["I", "II", "III"]
        elif self == CategoryType.AdvancedStage:
            return ["IA", "IIA", "IIB", "IIIC", "IIIB", "IIIC"]
        elif self == CategoryType.Grade:
            return ["One", "Two", "Three"]
        assert False, "Undefined Type!"  # pyright: ignore[reportUnreachable]
