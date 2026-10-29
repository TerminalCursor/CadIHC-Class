from enum import Enum


class DataframeType(Enum):
    """# Dataframe Usage Type"""

    Unknown = 0
    Training = 1
    Validation = 2
    Test = 3
