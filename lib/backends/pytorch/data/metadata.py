from ....interfaces.data.types import DataframeType


class Metadata:
    """# Metadata Class for Sample

    Demarcates the Magnification and Assignment of the Sample
    """

    magnification: int = 20
    assignment: DataframeType = DataframeType.Unknown
