VERSION='1.0.0'

###

import logging, sys
import pyclbr, os

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

class PowerfulLogRecordFactory(logging.LogRecord):
    def lookup_className(self):
        # ref: https://code.activestate.com/lists/python-list/727185
        if self.funcName in ['<module>']:
            return '<no-class>'
        if self.module in ['application', 'shellapp']:
            return '<no-class>'

        try:
            m = pyclbr.readmodule_ex(self.module, path=self.lookup_pathnames)
        except ModuleNotFoundError:
            return '<no-class>'

        for className, cls in m.items():
            # get rid of imported classes
            if cls.file != self.pathname:
                continue
            if self.lineno >= cls.lineno and self.lineno <= cls.end_lineno:
                return className

        return '<no-class>'

    def __init__(self, name, level, pathname, lineno, msg, args, exc_info, func, sinfo):
        super().__init__(name, level, pathname, lineno, msg, args, exc_info, func, sinfo)
        dirname = os.path.dirname(__file__)
        self.lookup_pathnames = [
            dirname,
            os.path.join(dirname, 'core'),
            os.path.join(dirname, 'ptrace/debugger'),
        ]
        self.className = self.lookup_className()
        if self.className in ['Fuzzer']:
            indent = 0
        elif self.className in ['ProcessSignal', 'SignalInfo']:
            indent = 0
        elif self.className not in [
                    'FuzzerSession',
                    'FuzzerConfig',
                    'CoverageExplorer',
                    'UniformStrategy',
                    'ReactiveInputGenerator',
                    'CoverageReplayLoader',
                    'CoverageTracker',
                    'CoverageDriver', 'CoverageForkDriver',
                    'TCPChannelFactory', 'TCPForkChannelFactory',
                    'UDPChannelFactory', 'UDPForkChannelFactory',
                ] and self.funcName in ['initialize', 'finalize']:
            indent = 2
        elif (self.className in ['BaseTracker', 'CoverageTracker'] \
                and self.funcName in ['update_state', 'peek', 'extract_snapshot', '_update_local', 'reset_state']) \
                or (self.className in ['BaseExplorer'] \
                and self.funcName in ['reload_state', 'attempt_load_state', '_arbitrate_load_state']):
            indent = 2
        elif self.className in ['ReplayLoader'] \
                and self.funcName in ['load_state', 'load_path', 'apply_transition']:
            indent = 2
        elif self.className in [
                'TCPChannel', 'TCPForkChannelFactory', 'TCPForkBeforeAcceptChannel',
                'UDPChannel', 'UDPForkChannelFactory',
                'ListenerSocketState', 'UDPSocketState',
                'FileDescriptorChannel',
                'PtraceChannel', 'PtraceForkChannel',
                'ProcessDriver', 'ProcessForkDriver',
                'BaseStateGraph', 'FeatureMap']:
            indent = 2
        else:
            indent = 1
        self.msg = ' ' * 4 * indent + msg

class ColoredLogger(logging.Logger):
    FORMAT = "%(asctime)s [$BOLD%(name)-12s$RESET][%(levelname)-8s] %(message)s " \
             "($BOLD%(className)s$RESET::%(funcName)s()) ($BOLD%(filename)s$RESET:%(lineno)d)"
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
logging.setLogRecordFactory(PowerfulLogRecordFactory)
root_logger = ColoredLogger()

logger = logging.getLogger("root")
for func in ('debug', 'info', 'warning', 'error', 'critical'):
    setattr(sys.modules[__name__], func, getattr(logger, func))

###

from . import common
from . import exceptions
from . import unix
from . import replay
from . import cov
from . import havoc
from . import net
from . import raw
from . import reactive
from . import webui
from . import cli
from . import fuzzer
from . import inference
from . import hotplug