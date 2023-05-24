# we disable profiling before importing tango
import os
os.environ['TANGO_NO_PROFILE'] = 'y'

from tango.fuzzer import Fuzzer
from tango.common import create_session_context
from pathlib import Path
import numpy as np
import asyncio
import argparse
import logging
import json

def create_argparse():
    parser = argparse.ArgumentParser(description=(
        "Runs state inference on a seed corpus and outputs results."
    ))
    parser.add_argument("-C", "--corpus", type=Path, required=True,
        help="The path to the seed corpus directory.")
    return parser

async def run_inference(session):
    # strategy is already instantiated, we only get a reference to it
    strat = await session.owner.instantiate('strategy')
    tracker = await session.owner.instantiate('tracker')
    while (rem := len(tracker.unmapped_states)) >= strat._inference_batch:
        logging.info(f"Remaining snapshots: {rem}")
        await strat.step()
    # flush the remaining nodes
    strat._inference_batch = rem
    await strat.step()
    groupings = {str(k): v for k, v in tracker.equivalence_states.items()}
    print(json.dumps(groupings, cls=NumpyEncoder))
    logging.info("Done!")

async def infer(fuzzer):
    async with asyncio.TaskGroup() as tg:
        context = create_session_context(tg)
        session = await fuzzer.create_session(context)
        await tg.create_task(run_inference(session), context=context)

def main():
    parser = create_argparse()
    argspace, rest = parser.parse_known_args()

    overrides = {
        'generator.seeds': str(argspace.corpus),
        'strategy.type': 'inference',
        'strategy.inference_batch': 50,
        'strategy.recursive_collapse': True,
        'strategy.extend_on_groups': True,
        'strategy.dt_predict': True,
        'strategy.dt_extrapolate': True,
        'tracker.native_lib': False,
        'tracker.skip_counts': True,
    }
    fuzzer = Fuzzer(args=rest, overrides=overrides)
    asyncio.run(infer(fuzzer))

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    main()
