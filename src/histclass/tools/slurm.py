from ..interfaces.monad.maybe import Maybe


def get_sat_id() -> int | None:
    """# Get Slurm Array Task ID

    If there is none, return None
    Else, return it as an int
    """
    from .env import get_env

    env_var_v: Maybe[str] = get_env("SLURM_ARRAY_TASK_ID")

    def index(x: str) -> int:
        """# Transform data to index"""
        return int(x) - 1

    return env_var_v(index).data
