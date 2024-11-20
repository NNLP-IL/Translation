from src import NRMLogger
from src.decorators import timer, log
from src.logger.objects import LogLevel


@log
def check():
    new_log = NRMLogger()
    new_log.log(message="Hello World", level=LogLevel.TRACE)


if __name__ == '__main__':
    check()
