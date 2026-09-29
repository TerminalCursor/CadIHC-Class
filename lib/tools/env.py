from .call_graph import cg


def get_env(env_var: str):
    """# Get Environment variable

    Get an environmental variable stored in a `Maybe`
    """
    cg.add("get_env")
    from ..interfaces.monad.maybe import Maybe
    from os import getenv

    return Maybe(getenv(env_var))
