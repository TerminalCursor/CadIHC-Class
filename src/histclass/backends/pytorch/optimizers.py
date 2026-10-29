from ...interfaces.config import CompleteConfig
from ...interfaces.monad.maybe import Maybe
from ...interfaces.models.types import OptimizerT
from torch.optim import Adam, AdamW, SGD


def get_optimizer(config: CompleteConfig) -> Maybe[OptimizerT]:
    """# Returns the optimizer function corresponding to the name

    Backend-specific function
    """
    name = config.optimizer
    if name == "Adam":
        return Maybe(Adam)
    elif name == "AdamW":
        return Maybe(AdamW)
    elif name == "SGD":
        return Maybe(SGD)
    return Maybe()
