from ...tools.call_graph import cg
from .types import DataTypes, CategoryTypes
from typing import List
from ...interfaces.config import CompleteConfig


class Dataset:
    def __init__(self, data_sources: List[str], configuration: CompleteConfig):
        cg.add("Dataset.__init__")
        data_types = configuration.input_data_types
