import os
os.environ['TANGO_NO_PROFILE'] = 'y'
import sys
sys.setrecursionlimit(10000)
from datetime import datetime
import pandas as pd

from tango.core import FuzzerConfig
from tango.exceptions import LoadedException, ChannelBrokenException, \
    ProcessTerminatedException, ProcessCrashedException
import asyncio
import sys
import argparse
import logging
import signal
import psutil
from replay import replay_load, EmptyTracker

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
    if not channel.root.is_stopped:
        channel.root.kill(signal.SIGSTOP)
        await channel.root.waitEvent()
        channel.root.kill(signal.SIGCONT)
    await asyncio.sleep(0.1)
    channel.root.kill(signal.SIGSEGV)
    channel.root.detach()

    ## ahmad's original version
    ## channel._pobj.detach()
    ## await asyncio.sleep(0.1)
    ## channel._pobj.kill(signal.SIGSEGV)
    await channel.root.waitExit()

tango_folder = os.getcwd()
async def task(args, file):
    try:
        config = FuzzerConfig(args.config, {
            'fuzzer': {
                'resume': True,
                'work_dir': args.workdir,
                'cwd': args.cwd
            }, 'driver': {
                'forkserver': False,
                'isolate_fs':  False,
                'exec': {
                    'stdout': 'inherit',
                    'stderr': 'inherit',
                }
            }, 'tracker': {
                'type': 'empty'
            }
        })
        config._config["driver"]["exec"]["env"]["ASAN_OPTIONS"] += \
                        ":coverage=1:handle_segv=2"
        filename = file.split("/")[-1]
        config._config["driver"]["exec"]["env"]["ASAN_OPTIONS"] += \
                        ":coverage_dir={}/fs/shared/{}_sancov_dir".format(
                            args.workdir, process_seed_name(filename))

        proc_crashed = False
        drv, inp = await replay_load(config, file)
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
        if not proc_crashed and psutil.pid_exists(drv._channel.root.pid):
            await send_eof(drv._channel)
    finally:
        os.chdir(tango_folder)

# need to replace some characters ("," ":")
def process_seed_name(name):
    new_name = name.replace(",", "_")
    new_name = new_name.replace(":", "_")
    return new_name

def main():
    args = parse_args()
    configure_verbosity(args.verbose)

    # clear sancov_dir
    os.system(f"rm -rf {args.workdir}/fs/shared/*_sancov_dir")

    cov_info = {}
    seed_filenames = []

    # get all pcaps to replay
    for _, _, filenames in os.walk(os.path.join(args.workdir, "queue")):
        for filename in filenames:
            seed_filenames.append(os.path.join(args.workdir, "queue", filename))
        break
    seed_filenames.sort(key=os.path.getmtime)

    # calculate the time elapsed
    start_time = os.path.getmtime(seed_filenames[0])
    for seed_filename in seed_filenames:
        filename = seed_filename.split("/")[-1]
        creation_time = os.path.getmtime(seed_filename)
        cov_info[filename] = {
            'time_elapsed': creation_time - start_time
        }

    # replay one by one
    for seed_filename in seed_filenames:
        filename = seed_filename.split("/")[-1]

        sancov_dir = "{}/fs/shared/{}_sancov_dir".format(args.workdir, process_seed_name(filename))
        if not os.path.exists(sancov_dir):
            os.makedirs(sancov_dir)
        try:
            print("current file is", seed_filename)
            asyncio.run(task(args, seed_filename))
        except Exception as ex:
            # import ipdb; ipdb.set_trace()
            print("!!!!!!!!!!crashing file is: {}!!!!!!!!!!".format(filename))
            import traceback; print(traceback.format_exc())

    # construct edge coverage
    global_pc_set = set()
    time2cov = {}
    for seed_filename in seed_filenames:
        filename = seed_filename.split("/")[-1]
        sancov_folder_pathname = os.path.join(args.workdir, "fs/shared", process_seed_name(filename) + "_sancov_dir")

        local_pc_set = set()
        if os.listdir(sancov_folder_pathname):
            # if we need to consider the shared libraries
            for sancov_filename in os.listdir(sancov_folder_pathname):
                if ".so." in sancov_filename:
                    continue
                local_pc_set = local_pc_set.union(read_sancov(os.path.join(sancov_folder_pathname, sancov_filename)))
        else:
            print("{} has no sancov file".format(filename))
        global_pc_set = global_pc_set.union(local_pc_set)
        cov_info[filename]['pc_cov_cnt'] = len(global_pc_set)
        time2cov[cov_info[filename]['time_elapsed']] = len(global_pc_set)

    # dump the edge coverage
    time_list, pc_cnt_list = [], []
    for time, pc_cnt in time2cov.items():
        time_list.append(time)
        pc_cnt_list.append(pc_cnt)
    df = pd.DataFrame({'time_elapsed': time_list, 'pc_cov_cnt': pc_cnt_list})
    df.to_csv(os.path.join(args.workdir, 'pc_cov_cnts.csv'), index=False)

if __name__ == '__main__':
    main()
