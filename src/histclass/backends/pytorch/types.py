from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from typing import Callable, Any

ActivationT = Callable[[Any], Any]
DataT = Tuple[Tensor, Tensor, str]
DatasetT = Dataset[DataT]
CDatasetT = ConcatDataset[DataT]
DataloaderT = DataLoader[DataT]
