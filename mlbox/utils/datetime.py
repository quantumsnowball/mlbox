from datetime import datetime


def localnow() -> datetime:
    return datetime.now()


def localnow_string(fmt: str = '%Y-%m-%dT%H.%M.%S.%f') -> str:
    return localnow().strftime(fmt)
