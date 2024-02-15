import os
os.environ['TANGO_NO_PROFILE'] = 'y'
import sys
sys.setrecursionlimit(10000)

from tango.core import FuzzerConfig
from tango.core.tracker import *
from tango.exceptions import LoadedException, ChannelBrokenException, \
    ProcessTerminatedException, ProcessCrashedException
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
import psutil

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
    top.add_argument('-C', '--config', required=True,
        help="The path to the TangoFuzz fuzz.json file.")
    top.add_argument('-W', '--workdir', required=True,
        help="The absolute path to the workdir to be analyzed.")
    top.add_argument('-c', '--cwd', required=True,
        help="The cwd for fuzzer.cwd in the fuzz.json")
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
    await asyncio.sleep(0.1)
    channel._proc.kill(signal.SIGSEGV)
    channel._proc.detach()

    ## ahmad's original version
    ## channel._proc.detach()
    ## await asyncio.sleep(0.1)
    ## channel._proc.kill(signal.SIGSEGV)
    await channel._proc.waitExit()

tango_folder = os.getcwd()
async def task(args, workdir, file):
    try:
        # file name
        file_name = file.split("/")[-1]

        config = FuzzerConfig(args.config, {
            'fuzzer': {'resume': True, 'work_dir': args.workdir, 'cwd': args.cwd},
            'driver': {'forkserver': False, "isolate_fs": False},
        })
        config._config["driver"]["exec"]["env"]["ASAN_OPTIONS"] += \
                        ":coverage=1:handle_segv=2"
        # config._config["driver"]["exec"]["env"]["ASAN_OPTIONS"] += \
        #                 ":coverage_dir=/shared/{}_sancov_dir".format(process_seed_name(file_name))
        config._config["driver"]["exec"]["env"]["ASAN_OPTIONS"] += \
                        ":coverage_dir={}/fs/shared/{}_sancov_dir".format(args.workdir, process_seed_name(file_name))
        # config._config["driver"]["exec"]["env"]["ASAN_OPTIONS"] = ""
        config._config["driver"]["exec"]["stdout"] = "inherit"
        config._config["driver"]["exec"]["stderr"] = "inherit"
        config._config['generator'] = {'type': config._config['generator']['type']}
        config._config['strategy'] = {'type': config._config['strategy']['type']}
        config._config['tracker'] = {'type': 'empty'}

        gen = await config.instantiate('generator')
        inp = gen.load_input(file)

        drv = await config.instantiate('driver')
        proc_crashed = False
        await drv.relaunch()
        try:
            await drv.execute_input(inp)
        except LoadedException as ex:
            if isinstance(ex._ex, ProcessCrashedException):
                proc_crashed = True
            if not isinstance(ex._ex, ProcessTerminatedException) and \
               not isinstance(ex._ex, ChannelBrokenException):
                raise
        # only send eof if process not crashed, otherwise would have "process
        # not exist" error
        if not proc_crashed and psutil.pid_exists(drv._channel._proc.pid):
            await send_eof(drv._channel)
    finally:
        os.chdir(tango_folder)


def rm_target_dir(file_path):
    folders = []
    for folder in file_path.split("/"):
        if folder != "/":
            folders.append(folder)
    folders = folders[2:]
    return os.path.join(*folders)

def check_not_processed(cov_files, analyzed_file_names):
    cov_file_names = [cov_file.split("/")[-1] for cov_file in cov_files]
    print(set(cov_file_names) - set(analyzed_file_names))

# need to replace some characters ("," ":")
def process_seed_name(name):
    new_name = name.replace(",", "_")
    new_name = new_name.replace(":", "_")
    return new_name

# def diff_old_new_sancov(workdir, file_name):
#     old_sancov = os.path.join(workdir, "fs/shared", "{}_sancov".format(file_name))
#     new_sancov_dir_path = os.path.join(workdir, "fs/shared", "{}_sancov_dir".format(process_seed_name(file_name)))
#     if not os.listdir(new_sancov_dir_path):
#         return
#     new_sancov = os.path.join(new_sancov_dir_path, os.listdir(new_sancov_dir_path)[0])
#     os.system("diff {} {}".format(old_sancov, new_sancov))

from datetime import datetime
def main():
    args = parse_args()
    configure_verbosity(args.verbose)

    workdir = rm_target_dir(args.workdir)
    cov_info_dict = {}
    cov_files = []

    for _, _, files in os.walk(os.path.join(args.workdir, "queue")):
        for file in files:
            cov_files.append(os.path.join(args.workdir, "queue", file))
        break
    cov_files.sort(key=os.path.getmtime)
    # for file in cov_files:
    #     file_name = file.split("/")[-1]
    #     cov_info_dict[file_name] = {
    #         "creation_time": str(datetime.fromtimestamp(os.path.getmtime(file)))
    #     }
    cov_files = [os.path.join(args.workdir, "queue", file.split("/")[-1]) for file in cov_files]
    
    # cov_files = ["out-openssl-nyxnet-001/queue/cov_569.tango"]
    for file in cov_files:
        file_name = file.split("/")[-1]
        # check if this file has been analyzed

        sancov_dir = "{}/fs/shared/{}_sancov_dir".format(args.workdir, process_seed_name(file_name))
        if not os.path.exists(sancov_dir):
            os.makedirs(sancov_dir)
        if os.listdir(sancov_dir):
            # already analyzed it
            continue
        # diff_old_new_sancov(args.workdir, file_name)
        # continue
        try:
            print("current file is", file)
            asyncio.run(task(args, workdir, file))
            # find the generated sancov file
            # sancov_file = os.path.join(args.workdir, glob.glob("**/*.sancov", root_dir=args.workdir, recursive=True)[0])
            # print(glob.glob("**/*.sancov", root_dir=args.workdir, recursive=True))
            # print("sancov_file is ", sancov_file)
            # pc_set = read_sancov(sancov_file)
            # cov_info_dict[file_name]["pc_set"] = list(pc_set)
            # os.rename(sancov_file, os.path.join(args.workdir, "fs/shared", file_name + "_sancov"))
        except Exception as ex:
            #import ipdb; ipdb.set_trace()
            print("!!!!!!!!!!crashing file is {}: {}!!!!!!!!!!".format(workdir, file_name))
            # if any sancov file has been generated along with the error, delete it
            if os.listdir(sancov_dir):
                os.system("rm {}/*".format(sancov_dir))
            import traceback; print(traceback.format_exc())
            return
    # # now delete all generated .sancov files for consistency
    # workdir_real_path = os.path.realpath(args.workdir)
    # os.chdir(workdir_real_path)
    # if glob.glob("**/*.sancov", root_dir=args.workdir, recursive=True):
    #     os.system("rm $(find . -name *.sancov)")

    # os.chdir(tango_folder)

if __name__ == '__main__':
    main()