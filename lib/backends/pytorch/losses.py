from ...tools.call_graph import cg
from typing import Callable
from torch.nn.functional import mse_loss
from torch.nn.functional import cross_entropy


def get_loss(name: str) -> Callable | None:
    """# Returns the loss function corresponding to the name

    Backend-specific function
    """
    cg.add("get_loss")
    if name == "mse_loss":
        return mse_loss
    elif name == "cross_entropy":
        return cross_entropy
    return None
