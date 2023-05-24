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
    parser.add_argument("-O", "--out", type=Path, required=False,
        help="The path to the output file (stdout otherwise).")
    return parser

async def run_inference(session, *, outfile=None):
    # strategy is already instantiated, we only get a reference to it
    strat = await session.owner.instantiate('strategy')
    tracker = await session.owner.instantiate('tracker')
    while True:
        while (rem := len(tracker.unmapped_states)) >= strat._inference_batch:
            logging.info(f"Remaining snapshots: {rem}")
            await strat.step()
        if rem == 0:
            break
        # flush the remaining nodes
        strat._inference_batch = rem
    groupings = {str(k): v for k, v in tracker.equivalence_states.items()}
    dump = json.dumps(groupings, cls=NumpyEncoder)
    if outfile:
        outfile.write_text(dump)
    else:
        print(dump)
    logging.info("Done!")

async def infer(fuzzer, **kwargs):
    async with asyncio.TaskGroup() as tg:
        context = create_session_context(tg)
        session = await fuzzer.create_session(context)
        await tg.create_task(run_inference(session, **kwargs), context=context)

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
    asyncio.run(infer(fuzzer, outfile=argspace.out))

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
