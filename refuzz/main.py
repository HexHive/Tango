from fuzzer import FuzzerSession, FuzzerConfig
import logging

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("statemanager").setLevel(logging.DEBUG)

config = FuzzerConfig("./fuzz.json")
sess = FuzzerSession(config)
sess.start()