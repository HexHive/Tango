from tangofuzz.fuzzer import FuzzerConfig
from tangofuzz.common import timeit

import asyncio
import argparse
import logging
from subprocess import PIPE
from functools import wraps

def parse_args():
    top = argparse.ArgumentParser(description=(
        "Measures the execution rates of different fuzzer operations."
    ))
    top.add_argument('-C', '--config', required=True,
        help="The path to the TangoFuzz fuzz.json file.")
    top.add_argument('-S', '--samples', default=1000, type=int,
        help="The number of samples to measure.")
    top.add_argument('--override', action='append', nargs=2)
    top.add_argument('-v', '--verbose', action='count', default=0,
        help=("Controls the verbosity of messages. "
            "-v prints info. -vv prints debug. Default: warnings and higher.")
        )
    sub_top = top.add_subparsers(dest='command')

    # loader reset rate
    reset = sub_top.add_parser('reset', description=(
        "Restart target to root state"
    ))

    # channel throughput
    throughput = sub_top.add_parser('throughput', description=(
        "Start target and send input file contents over the channel"
    ))
    throughput.add_argument('file', help="The path to the input file.")

    return top.parse_args()

def configure_verbosity(level):
    mapping = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    # will raise exception when level is invalid
    numeric_level = mapping[level]
    logging.getLogger().setLevel(numeric_level)

async def measure_load_time(config, args):
    # load once without measuring
    ld = await config.loader
    await ld.load_state(None, None)

    async def load_many():
        for _ in range(args.samples):
            await ld.load_state(None, None)
    await timeit(load_many)()

async def measure_send_time(config, args):
    gen = await config.input_generator
    inp = gen.load_input(args.file)
    ld = await config.loader
    await ld.load_state(None, None)

    async def do_many():
        for _ in range(args.samples):
            await ld.execute_input(inp)
    await timeit(do_many)()

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

    command = {
        'reset': measure_load_time,
        'throughput': measure_send_time
    }[args.command]

    asyncio.run(command(config, args))

if __name__ == '__main__':
    main()
