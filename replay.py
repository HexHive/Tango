from tangofuzz.fuzzer import FuzzerConfig
import asyncio
import argparse
import logging
from subprocess import PIPE

def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Replays an input file to a TangoFuzz target."
    ))
    parser.add_argument("config",
        help="The path to the TangoFuzz fuzz.json file.")
    parser.add_argument("file",
        help="The path to the input file.")
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

async def replay(config, file):
    gen = await config.input_generator
    inp = gen.load_input(file)
    ld = await config.loader
    await ld.load_state(None, None)
    await ld.execute_input(inp)

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
    config._config["exec"]["stdout"] = "inherit"
    config._config["exec"]["stderr"] = "inherit"
    asyncio.run(replay(config, args.file))

if __name__ == '__main__':
    main()
