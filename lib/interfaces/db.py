from ..tools.call_graph import cg


class Database:
    """# Database Class

    Handles a SQLite3 Database
    """

    def __init__(self, filename: str):
        cg.add("Database.__init__")
        self.filename = filename
