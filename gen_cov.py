import os
os.environ['TANGO_NO_PROFILE'] = 'y'

from tango.core import FuzzerConfig
from tango.core.tracker import *
from abc          import ABC
from tango.common import AsyncComponent, ComponentType
from tango.common import AsyncComponent, timeit
from typing import Iterable
import asyncio
import sys
import argparse
import logging
import signal
import time

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
    top = argparse.ArgumentParser(description=(
        "Dumps the raw data from a Tango input."
    ))
    top.add_argument('config',
        help="The path to the TangoFuzz fuzz.json file.")
    top.add_argument('workdir',
        help="The path to the workdir to be analyzed.")
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

def read_sancov(file_path):
    pc_set = []
    with open(file_path, "rb") as file:
        byte = file.read(1)
        if int.from_bytes(byte, "little") == 0x64:
            pc_length = 8
        else:
            pc_length = 4
        byte = file.read(7)
        while byte:
            byte = file.read(pc_length)
            pc_set.append(hex(int.from_bytes(byte, "little")))
    return set(pc_set[:-1])

async def send_eof(channel):
    await channel.shutdown()
    if not channel._proc.is_stopped:
        channel._proc.kill(signal.SIGSTOP)
        await channel._proc.waitEvent()
        channel._proc.kill(signal.SIGCONT)
    channel._proc.detach()
    await asyncio.sleep(0.1)
    channel._proc.kill(signal.SIGSEGV)
    await channel._proc.waitExit()

tango_folder = os.getcwd()
async def task(config, file):
    gen = await config.instantiate('generator')
    inp = gen.load_input(file)

    drv = await config.instantiate('driver')
    await drv.relaunch()
    await drv.execute_input(inp)
    await send_eof(drv._channel)

def rm_target_dir(file_path):
    folders = []
    for folder in file_path.split("/"):
        if folder != "/":
            folders.append(folder)
    folders = folders[2:]
    return os.path.join(*folders)

import glob, json
# from tqdm import tqdm
from datetime import datetime
def main():
    args = parse_args()
    configure_verbosity(args.verbose)

    workdir = rm_target_dir(args.workdir)
    print(workdir)
    cov_info_dict = {}
    cov_files = []

    for _, _, files in os.walk(os.path.join(args.workdir, "queue")):
        for file in files:
            cov_files.append(os.path.join(args.workdir, "queue", file))
    cov_files.sort(key=os.path.getmtime)
    for file in cov_files:
        file_name = file.split("/")[-1]
        cov_info_dict[file_name] = {
            "creation_time": str(datetime.fromtimestamp(os.path.getmtime(file)))
        }
    cov_files = [os.path.join(workdir, "queue", file.split("/")[-1]) for file in cov_files]

    config = FuzzerConfig(args.config, {
        'fuzzer': {'resume': True, 'work_dir': workdir},
        'driver': {'forkserver': False},
    })
    config._config["driver"]["exec"]["env"]["ASAN_OPTIONS"] += ":coverage=1"
    config._config["driver"]["exec"]["env"]["ASAN_OPTIONS"] += ":coverage_dir=/shared"
    config._config['generator'] = {'type': config._config['generator']['type']}
    config._config['strategy'] = {'type': config._config['strategy']['type']}
    config._config['tracker'] = {'type': 'empty'}

    for file in cov_files:
        print("current file is", file)
        file_name = file.split("/")[-1]
        try:
            asyncio.run(task(config, file))
            # find the generated sancov file
            sancov_file = glob.glob("**/*.sancov", recursive=True)[0]
            print(glob.glob("**/*.sancov", recursive=True))
            print("sancov_file is ", sancov_file)
            pc_set = read_sancov(sancov_file)
            cov_info_dict[file_name]["pc_set"] = list(pc_set)
            os.remove(sancov_file)
        except Exception as ex:
            print("!!!!!!!!!!crashing file is {}!!!!!!!!!!".format(file_name))
            import traceback; print(traceback.format_exc())

    os.chdir(tango_folder)
    cov_info_json = json.dumps(cov_info_dict, indent=4)
    with open(os.path.join(args.workdir, "cov_info.json"), "w") as file:
        file.write(cov_info_json)

if __name__ == '__main__':
    main()