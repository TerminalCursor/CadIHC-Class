def get_datestring() -> str:
    """
    Generates a date string in the format YYMMDD-hhmmss-us.

    This function uses the current date and time, padding with zeros if necessary,
    to create a standardized date string.

    Returns:
        str: A date string in the format YYMMDD-hhmmss-us.
    """
    from datetime import datetime

    now = datetime.now()
    return f"{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-{now.microsecond:06}"
