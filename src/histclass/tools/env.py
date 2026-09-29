from ..interfaces.monad.maybe import Maybe


def get_env(env_var: str) -> Maybe[str]:
    """# Get Environment variable

    Get an environmental variable stored in a `Maybe`
    """
    from os import getenv
    from dotenv import load_dotenv

    load_dotenv()
    return Maybe(getenv(env_var))
