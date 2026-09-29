from .call_graph import cg


def get_sat_id():
    """# Get Slurm Array Task ID

    If there is none, return None
    Else, return it as an int
    """
    cg.add("get_sat_id")
    from .env import get_env

    env_var_v = get_env("SLURM_ARRAY_TASK_ID")

    def index(x):
        return int(x) - 1

    return env_var_v(index).data
