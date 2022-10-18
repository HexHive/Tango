from tangofuzz.fuzzer import FuzzerConfig
from tangofuzz.input import PCAPInput
import asyncio
import argparse
import logging
from subprocess import PIPE

def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Replays a PCAP file to a TangoFuzz target."
    ))
    parser.add_argument("config",
        help="The path to the TangoFuzz fuzz.json file.")
    parser.add_argument("pcap",
        help="The path to the .pcap file.")
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help=("Controls the verbosity of messages. "
            "-v prints info. -vv prints debug. Default: warnings and higher.")
        )
    return parser.parse_args()

def configure_verbosity(level):
    mapping = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    # will raise exception when level is invalid
    numeric_level = mapping[level]
    logging.getLogger().setLevel(numeric_level)

async def replay(config, pcap):
    inp = PCAPInput(pcap, protocol=await config.protocol)
    ld = await config.loader
    await ld.load_state(None, None)
    await ld.execute_input(inp)

def main():
    args = parse_args()
    configure_verbosity(args.verbose)

    config = FuzzerConfig(args.config)
    config._config["exec"]["stdout"] = "inherit"
    config._config["exec"]["stderr"] = "inherit"
    asyncio.run(replay(config, args.pcap))

if __name__ == '__main__':
    main()
