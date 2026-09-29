import pytest
from histclass.interfaces.config import CompleteConfig
from histclass.interfaces.data.types import DataTypes, CategoryType
from histclass.interfaces.data.dataset import Dataset, Dataframe


def test_dataframe_init():
    """Test initialisation of Dataframe"""
    _df = Dataframe(
        "source",
        DataTypes.Morphology,
        CategoryType.Grade,
        name="df_name",
    )


def test_dataset_init():
    """Test initialisation of Dataset"""
    sources = ["source1", "source2"]
    config = CompleteConfig(
        DataTypes.ImageCNN, CategoryType.TripleNegative, batch_size=10
    )
    dataset = Dataset(sources, config)
    assert dataset.data_types == config.input_data_types
    assert dataset.category_type == config.output_data_type


def test_dataset_get_dataloaders():
    """Test getting dataloaders from dataset"""
    sources = ["source1", "source2"]
    config = CompleteConfig(DataTypes.ImageCNN, CategoryType.BasicStage, batch_size=10)

    # Mock implementation for get_dataloaders
    def mock_get_dataloaders():
        return [{"name": f"dataframe_{i}", "data": {f"a{i}": 1}} for i in range(2)]

    dataset = Dataset(sources, config)
    dataset.get_dataloaders = mock_get_dataloaders
    dataloaders = dataset.get_dataloaders()
    assert len(dataloaders) == 2


def test_dataset_get_dataloaders_not_implemented():
    """Test getting dataloaders from dataset when it's not implemented"""
    sources = ["source1", "source2"]
    config = CompleteConfig(DataTypes.Nuclei, CategoryType.AdvancedStage, batch_size=10)
    with pytest.raises(NotImplementedError):
        Dataset(sources, config).get_dataloaders()
