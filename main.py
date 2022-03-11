from tangofuzz import *

from fuzzer import FuzzerSession, FuzzerConfig
import logging

logging.getLogger().setLevel(logging.WARN)
logging.getLogger("statemanager").setLevel(logging.INFO)
logging.getLogger("networkio").setLevel(logging.INFO)
logging.getLogger("fuzzer").setLevel(logging.CRITICAL)
logging.getLogger("webui").setLevel(logging.INFO)
logging.getLogger("ptrace").setLevel(logging.CRITICAL)

config = FuzzerConfig("./targets/forkexec_server/fuzz.json")
sess = FuzzerSession(config)
sess.start()
