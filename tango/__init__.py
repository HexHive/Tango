VERSION='1.0.0'

###

import logging, sys

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    'DEBUG': CYAN,
    'INFO': WHITE,
    'WARNING': YELLOW,
    'CRITICAL': MAGENTA,
    'ERROR': RED
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color, **kwargs):
        super().__init__(msg, **kwargs)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            msg_colored = f"{COLOR_SEQ % (30 + COLORS[levelname])} {record.msg} {RESET_SEQ}"
            record.msg = msg_colored
        return logging.Formatter.format(self, record)

class ColoredLogger(logging.Logger):
    FORMAT = "%(asctime)s [$BOLD%(name)-12s$RESET][%(levelname)-8s] %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    LOGGER_CACHE = {}

    def __new__(cls, name=None):
        if (logger := cls.LOGGER_CACHE.get(name)):
            return logger
        use_color = sys.stdout.isatty()
        color_fmt = cls.formatter_message(cls.FORMAT, use_color=use_color)
        color_formatter = ColoredFormatter(color_fmt, use_color=use_color, datefmt='%m/%d/%Y %I:%M:%S %p')

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        if name is None:
            logger = logging.getLogger()
        else:
            logger = super(logging.Logger, cls).__new__(cls)
            logging.Logger.__init__(logger, name, logging.WARNING)

        logger.addHandler(console)
        cls.LOGGER_CACHE[name] = logger
        return logger

    @staticmethod
    def formatter_message(message, use_color):
        if use_color:
            message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
        else:
            message = message.replace("$RESET", "").replace("$BOLD", "")
        return message

logging.getLogger().handlers.clear()
logging.setLoggerClass(ColoredLogger)
root_logger = ColoredLogger()

logger = logging.getLogger("root")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

###

from . import common
from . import exceptions
from . import unix
from . import cov
from . import havoc
from . import net
from . import raw
from . import reactive
from . import webui
from . import fuzzer
from . import inference