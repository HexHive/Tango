from tangofuzz.fuzzer import FuzzerConfig
from tangofuzz.common import timeit
import asyncio
import argparse
import logging
from subprocess import PIPE

def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Tests the reset rate of a target."
    ))
    parser.add_argument("config",
        help="The path to the TangoFuzz fuzz.json file.")
    parser.add_argument('-S', "--samples", default=1000, type=int,
        help="The number of samples to measure.")
    parser.add_argument('--override', action='append', nargs=2)
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

async def measure_time(config, samples):
    # load at least once without measuring
    ld = await config.loader
    await ld.load_state(None, None)

    async def load_many():
        for _ in range(samples):
            await ld.load_state(None, None)
    await timeit(load_many)()

def main():
    args = parse_args()
    configure_verbosity(args.verbose)

    overrides = dict()
    if args.override:
        for name, value in args.override:
            keys = name.split('.')
            levels = keys[:-1]
            d = overrides
            for k in levels:
                if not d.get(k):
                    d[k] = dict()
                d = d[k]
            key = keys[-1]
            if value.lower() in ('true', 'false'):
                value = (value == 'true')
            elif value.isnumeric():
                value = literal_eval(value)
            d[key] = value

    config = FuzzerConfig(args.config, overrides)
    config._config["exec"].pop("stdout", None)
    config._config["exec"].pop("stderr", None)
    asyncio.run(measure_time(config, args.samples))

if __name__ == '__main__':
    main()
