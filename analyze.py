# we disable profiling before importing tango
import os
# os.environ['TANGO_NO_PROFILE'] = 'y'

from tango.inference import InferenceMap
from tango.fuzzer import Fuzzer, FuzzerConfig
from tango.common import create_session_context
from tango.core import (
    initialize as initialize_profiler, is_profiling_active, LambdaProfiler)
from tango.core import PreparedInput, TransmitInstruction
from tango.exceptions import ProcessCrashedException

from pathlib import Path
from collections import defaultdict
from itertools import permutations
from dataclasses import dataclass, InitVar
from mmap import mmap, PROT_READ, PROT_WRITE
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import argparse
import logging
import asyncio
import json
import struct
import time
import sys
import shlex

###

async def cmd_inference(*, fuzzer_args, corpus, **kwargs):
    overrides = {
        'generator.seeds': str(corpus),
        'strategy.type': 'inference',
        'strategy.inference_batch': 50,
        'strategy.recursive_collapse': True,
        'strategy.extend_on_groups': True,
        'strategy.dt_predict': True,
        'strategy.dt_extrapolate': True,
        'tracker.skip_counts': True,
        'explorer.observe_postmortem': False,
    }
    fuzzer = Fuzzer(args=fuzzer_args, overrides=overrides)

    async with asyncio.TaskGroup() as tg:
        context = create_session_context(tg)
        session = await tg.create_task(
            fuzzer.create_session(context), context=context)
        return await tg.create_task(
            run_inference(session, **kwargs), context=context)

async def run_inference(session):
    # strategy is already instantiated, we only get a reference to it
    strat = await session.owner.instantiate('strategy')
    tracker = await session.owner.instantiate('tracker')
    while True:
        while (rem := len(tracker.unmapped_snapshots)) >= strat._inference_batch:
            logging.info(f"Remaining snapshots: {rem}")
            await strat.step()
        if rem == 0:
            break
        # flush the remaining nodes
        strat._inference_batch = rem
    groupings = {str(k): v for k, v in tracker.equivalence_states.items()}
    dump = json.dumps(groupings, cls=NumpyEncoder)
    logging.info("Done!")
    return dump

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

###

async def cmd_cross_inference(*, fuzzer_args, corpus_A, corpus_B):
    overrides = {
        'generator.seeds': None,
        'strategy.type': 'inference',
        'strategy.inference_batch': 50,
        'strategy.dynamic_inference': False,
        'strategy.recursive_collapse': False,
        'strategy.extend_on_groups': False,
        'strategy.dt_predict': False,
        'strategy.dt_extrapolate': False,
        'tracker.skip_counts': True,
        'explorer.observe_postmortem': False,
    }
    fuzzer = Fuzzer(args=fuzzer_args, overrides=overrides)
    cwd = os.getcwd()

    A = B = overlaps = 0
    corpus_A = corpus_A.absolute()
    corpus_B = corpus_B.absolute()
    for sid, corpora in enumerate(permutations((corpus_A, corpus_B))):
        os.chdir(cwd)
        async with asyncio.TaskGroup() as tg:
            context = create_session_context(tg)
            session = await tg.create_task(
                fuzzer.create_session(context, sid=sid), context=context)
            if is_profiling_active('cli'):
                cli = await session.owner.instantiate('cli')
                cli_task = tg.create_task(cli.run(), name=f'cli', context=context)
            if is_profiling_active('webui'):
                webui = await session.owner.instantiate('webui')
                webui_task = tg.create_task(webui.run(), name=f'webui', context=context)
            rv = await tg.create_task(
                run_cross_inference(session, *corpora),
                context=context)

            A += rv[:2][-sid]
            B += rv[:2][1-sid]
            overlaps += rv[2]

            if is_profiling_active('cli'):
                cli_task.cancel()
            if is_profiling_active('webui'):
                webui_task.cancel()
            if (ch := session._explorer._driver.channel):
                await ch._debugger.quit()

    dump = json.dumps(dict(item for item in locals().items() \
        if item[0] in ('A', 'B', 'overlaps')), cls=NumpyEncoder)
    logging.info("Done!")
    return dump

async def run_cross_inference(session, corpus_A, corpus_B):
    strat = await session.owner.instantiate('strategy')
    tracker = await session.owner.instantiate('tracker')
    explorer = await session.owner.instantiate('explorer')
    generator = await session.owner.instantiate('generator')
    equivalence = InferenceMap()

    LambdaProfiler('inferred_snapshots')(
        lambda: len(equivalence.mapped_snapshots))
    LambdaProfiler('states')(
        lambda: len(equivalence.state_labels))

    snapshots_A, features_A = await load_corpus(session, corpus_A)

    LambdaProfiler("snapshots")(lambda: len(target_snapshots))

    target_snapshots = snapshots_A
    target_features = features_A
    while True:
        equivalence = await strat.perform_inference(
            snapshots=target_snapshots,
            features=target_features,
            equivalence=equivalence)
        target_snapshots &= tracker.state_graph.nodes
        target_features &= tracker.state_graph.nodes
        if equivalence.mapped_snapshots == target_snapshots and \
                equivalence.feature_snapshots == target_features:
            break

    snapshots_B, features_B = await load_corpus(session, corpus_B)

    target_snapshots |= snapshots_B
    target_features |= features_B
    while True:
        equivalence = await strat.perform_inference(
            snapshots=target_snapshots,
            features=target_features,
            equivalence=equivalence)
        target_snapshots &= tracker.state_graph.nodes
        target_features &= tracker.state_graph.nodes
        if equivalence.mapped_snapshots == target_snapshots and \
                equivalence.feature_snapshots == target_features:
            break

    A = B = overlaps = 0
    for _, members in equivalence.states.items():
        in_A_set = members & snapshots_A
        in_B_set = members & snapshots_B
        in_A = list(in_A_set)
        in_B = list(in_B_set)

        subsumers_A = equivalence.subsumers(in_A)
        subsumers_B = equivalence.subsumers(in_B)

        B_subsumes_A = {lo for lo, hi in zip(in_A, subsumers_A) if hi & in_B_set}
        A_subsumes_B = {lo for lo, hi in zip(in_B, subsumers_B) if hi & in_A_set}

        if not in_A_set or B_subsumes_A == in_A_set:
            B += 1
        elif not in_B_set or A_subsumes_B == in_B_set:
            A += 1
        else:
            overlaps += 1

    return A, B, overlaps

async def load_corpus(session, corpus):
    tracker = await session.owner.instantiate('tracker')
    explorer = await session.owner.instantiate('explorer')
    generator = await session.owner.instantiate('generator')

    glbl_map = tracker._global
    saved = glbl_map.clone()
    inputs = import_inputs(generator, corpus)
    non_atomics = set()
    snapshots = set()
    features = set()
    for path, inp in inputs.items():
        # reset the global coverage map to that of the root;
        # this ensures that coverage is not masked by previous inputs
        glbl_map.copy_from(
            tracker.entry_state._feature_context)
        glbl_map.commit(
            tracker.entry_state._feature_mask)
        try:
            await explorer.reload_state()
            # feed input to target and populate state machine
            exp_path = await explorer.follow(inp,
                minimize=False, validate=True, atomic=False)
            assert explorer._last_state == exp_path[-1][1]

            snapshots.add(explorer._last_state)
            features.add(explorer._last_state)
            logging.info("Loaded seed file: %s", path)
        except ProcessCrashedException as pc:
            import ipdb; ipdb.set_trace()
            pass
        except Exception as ex:
            logging.warning("Failed to load %s: %s", path, ex)
            non_atomics.add(path)
    glbl_map.copy_from(saved)

    for path, inp in inputs.items():
        if len(inp) > 1:
            try:
                await explorer.reload_state()
                # feed input to target and populate state machine
                await explorer.follow(inp,
                    minimize=False, validate=False, atomic=True)
            except Exception as ex:
                pass
            exp_path = explorer._current_path.copy()
            assert explorer._last_state == exp_path[-1][1]

            logging.info("Loaded atomics from seed file: %s", inp)
            if path in non_atomics:
                snapshots.add(explorer._last_state)
            for _, dst, _ in exp_path:
                features.add(dst)

    return frozenset(snapshots), frozenset(features)

def import_inputs(generator, corpus):
    inputs = {}
    for path in corpus.glob('*'):
        if not path.is_file():
            continue
        input = generator.load_input(str(path))
        inputs[path] = input
    return inputs

###

IN_FORMATS = ('tango', 'nyx_py', 'nyx_bin')
OUT_FORMATS = ('tango', 'nyx_bin')
NYXBIN_CHECKSUMS = {
    'kamailio': 0xfffff98c474e534b,
    'lightftp': 0x2371d768474e534b,
    'bftpd': 0x2371d768474e534b,
    'dnsmasq': 0x1989dd87474e534b,
    'exim': 0xfffff98c474e534b,
    'openssh': 0x07b6cf07474e534b,
    'live555': 0x2371d768474e534b,
    'tinydtls': 0x1989dd87474e534b,
    'openssl': 0x1989dd87474e534b,
    'pure-ftpd': 0x2371d768474e534b,
    'proftpd': 0x2371d768474e534b,
    'dcmtk': 0x5230c19e474e534b,
    'forked-daapd': 0x91129c9b474e534b,
    'expat': 0x1870c1e5474e534b,
}
# HACK
NYXBIN_TARGETS = {
    'kamailio': 'kamailio',
    'dnsmasq': 'dnsmasq',
    'exim': 'exim',
    'sshd': 'openssh',
    'live555': 'live555',
    'openssl': 'openssl',
    'dcmqrscp': 'dcmtk',
    'xmlwf': 'expat',
}
NYXBIN_HDR = '<QQQQQ'
# HACK use speclib to parse and use spec correctly maybe
NYXBIN_SPEC = {
    'openssh': {
        0: lambda p: struct.pack(
            f'>I{max(len(p),4)-4}s', max(len(p),4)-4, p[4:]),
        1: lambda p: struct.pack(
            f'>I{max(len(p),12)-12}s', max(len(p),12)-12, p[4:]),
        2: lambda p: p,
    },
    'dcmtk': {
        # UNUSED
        -1: lambda p: p[:2] + struct.pack(
            f'>I{max(len(p),6)-6}s', len(p) if len(p) >= 6 else 0, p[6:]),
        0: lambda p: p,
    }
}

async def cmd_convert_corpus(*,
        fuzzer_args, in_format, in_dir, out_format, out_dir):
    fuzzer = Fuzzer(args=fuzzer_args)
    # FIXME find a way to avoid having to chdir every time...
    cwd = os.getcwd()
    config = FuzzerConfig(fuzzer._argspace.config, fuzzer._overrides)
    os.chdir(cwd)

    assert in_dir.is_dir()
    out_dir.mkdir(exist_ok=True)

    in_paths = (path for path in in_dir.glob('*') if path.is_file())
    parse_fn = globals()[f'read_{in_format}']
    write_fn = globals()[f'write_{out_format}']
    for path in in_paths:
        inp = await parse_fn(config, path)
        out_path = out_dir / path.stem
        await write_fn(config, out_path, inp)
        logging.info(f"Processed file: {path!s}")

async def read_tango(config, path):
    generator = await config.instantiate('generator')
    return generator._input_kls(file=str(path)).load()

async def write_tango(config, path, inp):
    path = path.with_suffix(path.suffix + '.tango')
    generator = await config.instantiate('generator')
    with path.open('wb') as f:
        generator._input_kls(file=f).dump(inp, name=path.stem)

async def read_nyx_py(config, path):
    inp = PreparedInput()
    exec(path.read_text(), globals() | {
        'packet': lambda data: inp.append(
            TransmitInstruction(bytes(data, 'latin-1')))
    }, locals())
    return inp

async def read_nyx_bin(config, path):
    program = Path(config._config['driver']['exec']['path']).name
    target = NYXBIN_TARGETS[program]
    with NyxBin(target, path) as f:
        return f.read()

async def write_nyx_bin(config, path, inp):
    path = path.with_suffix(path.suffix + '.bin')
    program = Path(config._config['driver']['exec']['path']).name
    target = NYXBIN_TARGETS[program]
    with NyxBin(target, path) as f:
        f.write(inp)

@dataclass
class NyxBin:
    target: str
    path: Path

    def __enter__(self):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        if fd == -1:
            raise FileNotFoundError
        with os.fdopen(fd, 'r+b') as f:
            self.mm = mmap(f.fileno(), 0, prot=PROT_READ | PROT_WRITE)
            self.off = 0
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.mm.close()

    def unpack(self, fmt):
        sz = struct.calcsize(fmt)
        res = struct.unpack_from(fmt, self.mm, self.off)
        self.off += sz
        return res

    def pack(self, fmt, *args):
        sz = struct.calcsize(fmt)
        res = struct.pack_into(fmt, self.mm, self.off, *args)
        self.off += sz
        return res

    def unpack_header(self):
        for name, val in zip(
                ('checksum', 'graph_size', 'data_size',
                 'graph_offset', 'data_offset'), self.unpack(NYXBIN_HDR)):
            setattr(self, name, val)

    def pack_header(self):
        self.pack(NYXBIN_HDR, *(getattr(self, name) for name in
            ('checksum', 'graph_size', 'data_size',
             'graph_offset', 'data_offset')))

    def unpack_graph(self):
        assert self.off == self.graph_offset
        self.graph_data = self.unpack('H' * self.graph_size)
        assert self.off == self.data_offset
        self.node_data = [None] * self.graph_size
        for i in range(self.graph_size):
            data_len, = self.unpack('H')
            self.node_data[i] = self.unpack(f'{data_len}s')[0]
        assert self.off == self.data_offset + self.data_size

    def pack_graph(self):
        assert self.graph_offset == self.off
        self.pack('H' * self.graph_size, *self.graph_data)
        assert self.data_offset == self.off
        for payload in self.node_data:
            self.pack('H', len(payload))
            self.pack(f'{len(payload)}s', payload)
        assert self.off == self.data_offset + self.data_size

    def read(self):
        self.unpack_header()
        self.unpack_graph()
        inp = PreparedInput()
        for node_type, payload in zip(self.graph_data, self.node_data):
            assert node_type == 0 or self.target in NYXBIN_SPEC, \
                f"Unknown node type {node_type} for {self.target}"
            nodes = NYXBIN_SPEC.get(self.target, {0: lambda p: p})
            conv = nodes[node_type]
            inp.append(TransmitInstruction(conv(payload)))
        return inp

    def write(self, inp):
        self.checksum = NYXBIN_CHECKSUMS[selt.target]
        self.graph_offset = struct.calcsize(NYXBIN_HDR)
        self.graph_size = 0
        self.data_size = 0
        self.graph_data = []
        self.node_data = []
        for instr in inp:
            if not isinstance(instr, TransmitInstruction) or not instr._data:
                continue
            # FIXME this is spec-dependent; we'll assume node 0 is always raw
            self.graph_data.append[0]
            self.node_data.append(instr._data)
            self.graph_size += 1
            self.data_size += len(instr._data)
        self.data_offset = \
            self.graph_offset + struct.calcsize('H' * self.graph_size)
        file_sz = self.data_offset + self.data_size
        self.mm.resize(file_sz)
        self.pack_header()
        self.pack_graph()

###

async def cmd_replay_input(*, fuzzer_args, restarts, input=None):
    overrides = {'generator.seeds': None}
    fuzzer = Fuzzer(args=fuzzer_args, overrides=overrides)
    states = defaultdict(int)
    if input:
        input = input.absolute()
    async with asyncio.TaskGroup() as tg:
        context = create_session_context(tg)
        session = await tg.create_task(
            fuzzer.create_session(context), context=context)

        if is_profiling_active('cli'):
            cli = await session.owner.instantiate('cli')
            cli_task = tg.create_task(cli.run(), name=f'cli', context=context)
        if is_profiling_active('webui'):
            webui = await session.owner.instantiate('webui')
            webui_task = tg.create_task(webui.run(), name=f'webui', context=context)

        rv = await tg.create_task(
            run_replay_input(session, restarts, input, states),
            context=context)

        if is_profiling_active('cli'):
            cli_task.cancel()
        if is_profiling_active('webui'):
            webui_task.cancel()

    return dict((repr(state), count) for state, count in states.items())

async def run_replay_input(session, restarts, input, states):
    explorer = await session.owner.instantiate('explorer')
    loader = await session.owner.instantiate('loader')
    driver = await session.owner.instantiate('driver')
    generator = await session.owner.instantiate('generator')
    if input:
        input = generator.load_input(str(input))

    while True:
        # this allows us to set restarts < 0 to specify continuous replay
        restarts -= 1
        try:
            await explorer.reload_state()
            if input:
                exp_path = await explorer.follow(input,
                    minimize=False, validate=True, atomic=True)
        except Exception as ex:
            logging.warning("Failed to replay: ex=%s", ex)
        finally:
            states[explorer._last_state] += 1
        if restarts == 0:
            break

###

async def cmd_batch(*, fuzzer_args, batch_cmd, cmdlines, workers):
    try:
        params = ACTIONS[batch_cmd]
        cmdfn = params['fn']
        if cmdlines:
            f = cmdlines.open("r")
        else:
            f = sys.stdin

        cwd = os.getcwd()
        context = mp.get_context('forkserver')
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=context)
        tasks = []
        i = -1
        for i, cmdline in enumerate(f):
            if not cmdline:
                # skip empty lines, if they exist
                continue
            args = shlex.split(cmdline)
            parser = argparse.ArgumentParser(**params['parser_kw'])
            for kw in params['args_kws']:
                cpy = kw.copy()
                name = cpy.pop('name')
                if isinstance(name, str):
                    name = (name,)
                parser.add_argument(*name, **cpy)
            parser.add_argument('fuzzer_args', nargs=argparse.REMAINDER)
            argspace = parser.parse_args(args)
            worker_args = fuzzer_args + argspace.fuzzer_args[1:] + \
                shlex.split(f'-o fuzzer.work_dir worker_{i}')
            kwargs = vars(argspace) | {'fuzzer_args': worker_args}
            loop = asyncio.get_running_loop()
            tasks.append(loop.run_in_executor(
                executor, run_worker, cwd, cmdfn, kwargs))

        for coro in asyncio.as_completed(tasks):
            kwargs, result = await coro
            print({'args': kwargs, 'result': result})
    finally:
        f.close()

def run_worker(cwd, cmdfn, kwargs):
    os.chdir(cwd)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return (kwargs, loop.run_until_complete(cmdfn(**kwargs)))

###

def create_argparse():
    parser = argparse.ArgumentParser(description=(
        "An AIO tool for running post-processing routines in Tango."
    ))
    parser.add_argument('-v', '--verbose', action='count', default=0,
        help=("Controls the verbosity of messages. "
            "-v prints info. -vv prints debug. Default: warnings and higher.")
        )
    parser.add_argument('-t', '--timeit', action='store_true', default=False,
        help=("Print out a time summary after the command terminates.")
        )

    subparsers = parser.add_subparsers(title="Analyses", required=True, dest='cmd')
    for cmd, params in ACTIONS.items():
        parser_cmd = subparsers.add_parser(cmd, **params['parser_kw'])
        for kw in params['args_kws']:
            cpy = kw.copy()
            name = cpy.pop('name')
            if isinstance(name, str):
                name = (name,)
            parser_cmd.add_argument(*name, **cpy)
        # parser_cmd.add_argument('fn',
        #     action='store_const', const=params['fn'], default='_')
        parser_cmd.add_argument('fuzzer_args', nargs=argparse.REMAINDER)

    return parser

ACTIONS = {}
ACTIONS |= {
    'inference': {
        'parser_kw': {},
        'args_kws': [
            {
                'name': ('-C', '--corpus',),
                'type': Path,
                'required': True,
                'help': (
                    "The path to the seed corpus directory."
                )
            },
        ],
        'fn': cmd_inference,
    },
    'cross_inference': {
        'parser_kw': {},
        'args_kws': [
            {
                'name': ('-A', '--corpus-A',),
                'type': Path,
                'required': True,
                'help': (
                    "The path to the seed corpus of fuzzer A."
                )
            },
            {
                'name': ('-B', '--corpus-B',),
                'type': Path,
                'required': True,
                'help': (
                    "The path to the seed corpus of fuzzer B."
                )
            },
        ],
        'fn': cmd_cross_inference,
    },
    'convert_corpus': {
        'parser_kw': {},
        'args_kws': [
            {
                'name': 'in_format',
                'choices': IN_FORMATS,
                'help': (
                    "The format of the input seeds."
                )
            },
            {
                'name': 'in_dir',
                'type': Path,
                'help': (
                    "The path to the seed corpus to be converted."
                )
            },
            {
                'name': 'out_format',
                'choices': OUT_FORMATS,
                'help': (
                    "The format of the output seeds."
                )
            },
            {
                'name': 'out_dir',
                'type': Path,
                'help': (
                    "The path to the output directory where converted files"
                    " will be stored."
                )
            },
        ],
        'fn': cmd_convert_corpus,
    },
    'replay_input': {
        'parser_kw': {},
        'args_kws': [
            {
                'name': ('-i', '--input'),
                'type': Path,
                'required': False,
                'help': (
                    "Path to the input file to replay after restart."
                )
            },
            {
                'name': ('-r', '--restarts'),
                'type': int,
                'default': 1,
                'help': (
                    "The number of times to repeat the restart+exec loop."
                )
            },
        ],
        'fn': cmd_replay_input,
    },
}
ACTIONS |= {
    'batch_run': {
        'parser_kw': {},
        'args_kws': [
            {
                'name': 'batch_cmd',
                'choices': tuple(ACTIONS),
                'help': (
                    "The command which will be run in batch."
                )
            },
            {
                'name': '--cmdlines',
                'type': Path,
                'required': False,
                'help': (
                    "The file containing the sets of arguments for each task,"
                    " one set per line (or from stdin otherwise)."
                )
            },
            {
                'name': ('-j', '--workers'),
                'type': int,
                'required': False,
                'help': (
                    "The number of worker processes in the pool."
                )
            },
        ],
        'fn': cmd_batch,
    }
}

async def bootstrap(coro):
    await initialize_profiler()
    return await coro

def configure_verbosity(level):
    mapping = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    # will raise exception when level is invalid
    numeric_level = mapping[level]
    logging.getLogger().setLevel(numeric_level)

def main():
    parser = create_argparse()
    argspace = parser.parse_args()
    configure_verbosity(argspace.verbose)

    if '--' in argspace.fuzzer_args:
        argspace.fuzzer_args.remove('--')
    kwargs = vars(argspace).copy()
    kwargs.pop('fn', None)
    kwargs.pop('cmd', None)
    kwargs.pop('verbose', None)
    kwargs.pop('timeit', None)

    start_time = time.time()
    fn = ACTIONS[argspace.cmd]['fn']
    try:
        if (result := asyncio.run(bootstrap(fn(**kwargs)))):
            print(result)
    except Exception as ex:
        logging.warning(ex)
        import ipdb
        ipdb.post_mortem(ex.__traceback__)
    finally:
        if argspace.timeit:
            logging.info("--- {} seconds ---".format(time.time() - start_time))
        return

if __name__ == '__main__':
    main()
