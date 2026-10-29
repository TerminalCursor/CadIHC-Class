from ..interfaces.monad.maybe import Maybe


def get_env(env_var: str) -> Maybe[str]:
    """
    Retrieves an environmental variable stored in a `Maybe`.

    This function loads the `.env` file using the `dotenv` library and then retrieves
    the specified environment variable using the `getenv` function from the `os` library.

    Args:
        env_var (str): The name of the environment variable to retrieve.

    Returns:
        Maybe[str]: A `Maybe` object containing either the retrieved value or an empty string if the variable is not set.
    """
    from os import getenv
    from dotenv import load_dotenv

    load_dotenv()
    return Maybe(getenv(env_var))
