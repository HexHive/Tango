import os
os.environ['TANGO_NO_PROFILE'] = 'y'

from tango.fuzzer import Fuzzer
from tango.core import FuzzerConfig
from tango.common import timeit
from tango.net import PCAPInput
import asyncio
import sys
import argparse
import logging

def parse_args():
    top = argparse.ArgumentParser(description=(
        "Dumps the raw data from a Tango input."
    ))
    top.add_argument('config',
        help="The path to the TangoFuzz fuzz.json file.")
    top.add_argument('input',
        help="The path to the Tango input.ext file.")
    # top.add_argument('output',
    #     help="The path where the output will be saved.")
    top.add_argument('-v', '--verbose', action='count', default=0,
        help=("Controls the verbosity of messages. "
            "-v prints info. -vv prints debug. Default: warnings and higher.")
        )
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

async def task(args):
    config = FuzzerConfig(args.config, {'fuzzer': {'resume': True}})
    gen = await config.instantiate('generator')
    inp = gen.load_input(args.input)
    new_inp = PCAPInput(file=f'{args.input}.orig', fmt=inp._fmt)
    new_inp.dumpi(inp)
    new_inp._file.close() 

class FMT(object):
    def __init__(self):
        self.protocol = 'tcp'
        self.port = 2022

async def pcap2tango(args):
    config = FuzzerConfig(args.config, {'fuzzer': {'resume': True}})
    gen = await config.instantiate('generator')
    pcap_inp = PCAPInput(file=args.input, fmt=gen._fmt)
    pcap_inp.dump(pcap_inp.loadi())
    # pcap_inp.dump(None, name="")
    # gen.save_input(new_inp, filepath=args.input + ".raw")
    
    pcap_inp._file.close() 

def main():
    args = parse_args()
    configure_verbosity(args.verbose)
    asyncio.run(task(args))
    # asyncio.run(pcap2tango(args))

if __name__ == '__main__':
    main()