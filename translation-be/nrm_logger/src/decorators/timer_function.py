from time import time
from nrm_logger.src.logger import NRMLogger
from nrm_logger.src.logger.objects import LogLevel

logger = NRMLogger()


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        logger.log(f"Function {func.__name__!r} took {end_time - start_time:.4f} seconds", level=LogLevel.TRACE)
        return result

    return wrapper
