from ....interfaces.data.types import DataframeType


class Metadata:
    """
    Attributes:
        magnification (int): The level of magnification applied to the sample.
        assignment (DataframeType): The assigned category or type of the sample.
    """

    magnification: int = 20
    assignment: DataframeType = DataframeType.Unknown
