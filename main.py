from refuzz import *

from fuzzer import FuzzerSession, FuzzerConfig
import logging

logging.getLogger().setLevel(logging.WARN)
logging.getLogger("statemanager").setLevel(logging.DEBUG)
logging.getLogger("networkio").setLevel(logging.INFO)
logging.getLogger("fuzzer").setLevel(logging.DEBUG)

config = FuzzerConfig("./fuzz.json")
sess = FuzzerSession(config)
sess.start()
