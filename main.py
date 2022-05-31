from tangofuzz import *

from fuzzer import FuzzerSession, FuzzerConfig
import logging
import asyncio

logging.getLogger().setLevel(logging.WARN)
logging.getLogger("statemanager").setLevel(logging.DEBUG)
logging.getLogger("networkio").setLevel(logging.DEBUG)
logging.getLogger("fuzzer").setLevel(logging.INFO)
logging.getLogger("webui").setLevel(logging.INFO)
logging.getLogger("ptrace").setLevel(logging.CRITICAL)
logging.getLogger("input").setLevel(logging.INFO)
logging.getLogger("generator").setLevel(logging.DEBUG)
logging.getLogger("mutator").setLevel(logging.DEBUG)
logging.getLogger("interaction").setLevel(logging.DEBUG)
logging.getLogger("loader").setLevel(logging.DEBUG)

async def main():
    config = FuzzerConfig("./targets/zoom/fuzz.json")
    sess = await FuzzerSession.create(config)
    await sess.start()

asyncio.run(main())