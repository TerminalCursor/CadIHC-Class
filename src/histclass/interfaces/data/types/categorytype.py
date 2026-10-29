from enum import Enum
from typing import List, Self


class CategoryType(Enum):
    """
    Enum representing different category types for model output.

    This enum defines the possible category types that can be used in a model.
    Each type has a corresponding string representation and values.

    Attributes:
        Unknown (int): Unknown category type (default value)
        TripleNegative (int): Whether the Sample is Triple Negative
        BasicStage (int): Whether the Sample should be classified into Stages Coarsely (I, II, ...)
        AdvancedStage (int): Whether the Sample should be classified into Stages Finely (IA, IIA, ...)
        Grade (int): Whether the Sample should be classified into Grades (1, 2, ...)
    """

    Unknown = 0
    TripleNegative = 1
    BasicStage = 2
    AdvancedStage = 3
    Grade = 4
    BinaryStage = 5
    BinaryGrade = 6

    def to_str(self: Self) -> str:
        """
        Convert the category type to its corresponding string representation.

        Args:
            self (CategoryType): The enum value to convert

        Returns:
            str: The string representation of the enum value

        Raises:
            AssertionError: If the enum value is not defined
        """
        if self == CategoryType.TripleNegative:
            return "tn"
        elif self == CategoryType.BasicStage:
            return "stage_basic"
        elif self == CategoryType.AdvancedStage:
            return "stage_advanced"
        elif self == CategoryType.Grade:
            return "grade"
        elif self == CategoryType.BinaryStage:
            return "hilostage"
        elif self == CategoryType.BinaryGrade:
            return "hilograde"
        assert False, "Undefined Type!"  # pyright: ignore[reportUnreachable]

    def values(self: Self) -> List[str] | None:
        """
        Get the list of string values corresponding to the category type.

        Args:
            self (CategoryType): The enum value to get string values for

        Returns:
            List[str]: A list of string values
            None: If the enum value is not defined

        Raises:
            AssertionError: If the enum value is not defined
        """
        if self == CategoryType.TripleNegative:
            return ["NTN", "TN"]
        elif self == CategoryType.BasicStage:
            return ["I", "II", "III"]
        elif self == CategoryType.AdvancedStage:
            return ["IA", "IIA", "IIB", "IIIC", "IIIB", "IIIC"]
        elif self == CategoryType.Grade:
            return ["One", "Two", "Three"]
        elif self == CategoryType.BinaryStage:
            return ["Early", "Late"]
        elif self == CategoryType.BinaryGrade:
            return ["Early", "Late"]
        assert False, "Undefined Type!"  # pyright: ignore[reportUnreachable]
