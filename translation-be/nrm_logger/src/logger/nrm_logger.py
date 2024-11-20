import logging
import json
import sys
from loguru import logger as loguru_logger
from .objects import LogLevel, LoggerConfig
from .utils import patching
import os 

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class NRMLogger:
    _config = None
    _instance = None
    CONFIG_PATH: str = DIR_PATH + "/config/logging_config.json"
    DEFAULT_LOGGING_FORMAT: str = "{time:MMMM D, YYYY > HH:mm:ss} | {name} {level} | {message}"

    # DEFAULT_LOGGING_FORMAT: str = "{time} | {level: <8} | {name: ^15} | {function: ^15} | {line: >3} | {message}"

    def __new__(cls, logger_name: str = "NRMLogger", config_path: str = None):
        cls.config_path = config_path if config_path else cls.CONFIG_PATH
        cls._config: LoggerConfig = cls._load_config(config_path=cls.config_path)
        if not cls._config.singleton or cls._instance is None:
            cls._instance = super(NRMLogger, cls).__new__(cls)
            
            # Adding the specified format
            loguru_logger.remove()  # Remove all handlers added so far, including the default one.
            # if json_fmt:
            #     loguru_logger.patch(patching)
            #     logging_format = "{extra[serialized]}"
            loguru_logger.add(cls._config.filename, format=cls._config.formatters.simple, rotation=cls._config.rotation.time, retention=cls._config.rotation.retention,
                              level=cls._config.level)
            loguru_logger.add(sys.stdout, format=cls._config.formatters.simple, level=cls._config.level)

            cls._instance._logger = loguru_logger.bind(name=logger_name)
        return cls._instance

    @staticmethod
    def _load_config(config_path: str) -> LoggerConfig:
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return LoggerConfig(**data['logger_object'])
        except json.JSONDecodeError as e:
            print(f"{str(e)}")

    def log(self, message: str, level: LogLevel = LogLevel.DEBUG):
        if not isinstance(level, LogLevel):
            raise ValueError("Invalid log level; must be a LogLevel enum member")

        log_method = {
            LogLevel.TRACE: self._logger.trace,
            LogLevel.DEBUG: self._logger.debug,
            LogLevel.INFO: self._logger.info,
            LogLevel.WARNING: self._logger.warning,
            LogLevel.ERROR: self._logger.error,
            LogLevel.CRITICAL: self._logger.critical,
            LogLevel.EXCEPTION: self._logger.exception
        }.get(level)

        log_method(message)

    def add(self, sink: logging.Handler, **kwargs):
        self._logger.add(sink=sink, **kwargs)
