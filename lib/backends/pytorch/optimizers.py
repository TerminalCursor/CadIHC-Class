from ...tools.call_graph import cg
from ...interfaces.config import CompleteConfig
from typing import Callable
from torch.optim import Adam, AdamW, SGD


def get_optimizer(config: CompleteConfig) -> Callable | None:
    """# Returns the optimizer function corresponding to the name

    Backend-specific function
    """
    cg.add("get_optimizer")
    name = config.optimizer
    if name == "Adam":
        return Adam
    elif name == "AdamW":
        return AdamW
    elif name == "SGD":
        return SGD
    return None
