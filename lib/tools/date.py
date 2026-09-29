from .call_graph import cg


def get_datestring():
    """# Get Datestring

    Format is YYMMDD-hhmmss-us
    """
    cg.add("get_datestring")
    from datetime import datetime

    now = datetime.now()
    return f"{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-{now.microsecond:06}"
