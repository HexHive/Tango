from tangofuzz import *

from fuzzer import FuzzerSession, FuzzerConfig
import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Launches a TangoFuzz fuzzing session."
    ))
    parser.add_argument("config",
        help="The path to the TangoFuzz fuzz.json file.")
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

def main():
    args = parse_args()
    configure_verbosity(args.verbose)

    config = FuzzerConfig(args.config)
    sess = FuzzerSession(config)
    sess.run()

if __name__ == '__main__':
    main()
