from ..interfaces.monad.maybe import Maybe


def get_sat_id() -> int | None:
    """
    Retrieves the Slurm Array Task ID from the environment variables.

    If no value is found, returns None. Otherwise, converts it to an integer index.

    Returns:
        int | None: The Slurm Array Task ID as an integer (or None if not set)
    """
    from .env import get_env

    env_var_v: Maybe[str] = get_env("SLURM_ARRAY_TASK_ID")

    def transform_to_index(x: str) -> int:
        """
        Converts a string to an integer index.

        This is necessary because the task ID may be stored as a string in the environment variable,
        but we need it as an integer for our use case.

        Args:
            x (str): The input string

        Returns:
            int: The converted integer index
        """
        return int(x) - 1

    return env_var_v(transform_to_index).data
