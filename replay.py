# we disable profiling before importing tango
import os
os.environ['TANGO_NO_PROFILE'] = 'y'

from tango.core import FuzzerConfig
from tango.raw import RawInput
import asyncio
import argparse
import logging
from subprocess import PIPE
from typing import Iterable

from tango.core.tracker import *
class EmptyTracker(BaseTracker):
    async def finalize(self, owner):
        pass

    @property
    def entry_state(self) -> AbstractState:
        """
        The state of the target when it is first launched (with no inputs sent)

        :returns:   The state object describing the entry state.
        :rtype:     AbstractState
        """
        pass

    @property
    def current_state(self) -> AbstractState:
        pass

    @property
    def state_graph(self) -> AbstractStateGraph:
        pass

    def peek(self, default_source: AbstractState, expected_destination: AbstractState) -> AbstractState:
        pass

    def reset_state(self, state: AbstractState):
        """
        Informs the state tracker that the loader has reset the target into a
        state.
        """
        pass

    def out_edges(self, state: AbstractState) -> Iterable[Transition]:
        pass

    def in_edges(self, state: AbstractState) -> Iterable[Transition]:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Replays an input file to a TangoFuzz target."
    ))
    parser.add_argument("config",
        help="The path to the TangoFuzz fuzz.json file.")
    parser.add_argument("file",
        help="The path to the input file.")
    parser.add_argument('-o', '--override', action='append', nargs=2)
    parser.add_argument('-v', '--verbose', action='count', default=-1,
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

async def replay_load(config, file):
    gen = await config.instantiate('generator')
    inp = gen.load_input(file)
    RawInput(file=f'replay.{gen._fmt.typ}.bin', fmt=gen._fmt).dumpi(inp)
    gen._input_kls(file=f'replay.{gen._fmt.typ}').dumpi(inp)

    ld = await config.instantiate('loader')
    try:
        await ld.load_state(None).asend(None)
    except StopAsyncIteration:
        pass
    drv = await config.instantiate('driver')
    return drv, inp

async def replay(config, file):
    drv, inp = await replay_load(config, file)
    await drv.execute_input(inp)

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

    overrides['tracker'] = {'type': 'empty'}
    config = FuzzerConfig(args.config, overrides)
    config._config["driver"]["exec"]["stdout"] = "inherit"
    config._config["driver"]["exec"]["stderr"] = "inherit"
    asyncio.run(replay(config, args.file))

if __name__ == '__main__':
    main()
