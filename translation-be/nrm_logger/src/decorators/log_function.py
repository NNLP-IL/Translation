from nrm_logger.src.logger import NRMLogger
from nrm_logger.src.logger.objects import LogLevel

logger = NRMLogger()


def log(func):
    def wrapper(*args, **kwargs):
        logger.log(f"Function {func.__name__!r} started", level=LogLevel.TRACE)
        result = func(*args, **kwargs)
        logger.log(f"Function {func.__name__!r} ended", level=LogLevel.TRACE)
        return result

    return wrapper
