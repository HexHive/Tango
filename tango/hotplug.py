from . import debug, info, warning

from tango.core import (
    AbstractInput, PreparedInput, TransmitInstruction, SeedableStrategy)
from tango.inference import StateInferenceStrategy
from tango.common import ComponentOwner

from asyncinotify import Inotify, Mask, Event as InotifyEvent

from abc import abstractmethod
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from itertools import count
from collections import defaultdict
import numpy as np
import asyncio
import struct
import os
import signal
import json
import re

@dataclass
class InotifyBatchObserver:
    inotify: Inotify
    timeout: int
    batch_size: int
    select: Callable[[InotifyEvent], bool] = lambda _: True
    process: Callable[[InotifyEvent], ...] = None

    def __post_init__(self):
        self._counter = 0

    async def get_event(self):
        while True:
            if self.select(ev := await anext(aiter(self.inotify))):
                return ev

    async def get_batch(self, batch: Optional[list]=None):
        for i in range(self.batch_size):
            ev = await self.get_event()
            info(f"Observed inotify event ({i + 1}/{self.batch_size}) for {ev.path!s}")
            if self.process:
                self.process(ev)
            if batch is not None:
                batch.append(ev)
        return batch

    async def get_batch_or_timeout(self):
        batch = []
        try:
            await asyncio.wait_for(self.get_batch(batch), self.timeout)
        finally:
            return batch

class HotplugInference(StateInferenceStrategy,
        capture_paths=('strategy.batch_timeout',)):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            'hotplug' in config['strategy']

    def __init__(self, *,
            batch_timeout: int,
            **kwargs):
        super().__init__(**kwargs)
        self._batch_timeout = batch_timeout
        self._snapshots = defaultdict(list)
        self._proc = None

    async def initialize(self):
        await super().initialize()

    async def finalize(self, owner: ComponentOwner):
        # we skip seed loading, to allow the fuzzer to map the discovered states
        # to paths
        await super(SeedableStrategy, self).finalize(owner)

    async def step(self, input: Optional[AbstractInput]=None):
        if not self._proc:
            self._proc = await self.start_fuzzer()
            while not self.watch_path.exists():
                await asyncio.sleep(1)
        remaining = self._inference_batch
        batch = []
        # WARN this is racey; the fuzzer could have created more inputs while
        # we iterate over the existing ones
        for p in self.watch_path.iterdir():
            if self.select_file(p):
                batch.append(p)
        remaining -= len(batch)

        if remaining > 0:
            with Inotify() as inotify:
                inotify.add_watch(self.watch_path, Mask.CREATE | Mask.ONLYDIR)
                observer = InotifyBatchObserver(
                    inotify, self._batch_timeout, remaining,
                    select=lambda ev: self.select_file(ev.path))
                events = await observer.get_batch_or_timeout()
                for ev in events:
                    batch.append(ev.path)

        if not batch:
            info("No new inputs after timeout");
            return

        inputs = await self.import_inputs(batch)
        for path, inp in inputs.items():
            try:
                await self._explorer.reload_state()

                # reset the global coverage map to that of the root;
                # this ensures that coverage is not masked by previous inputs
                self._explorer._tracker._global.copy_from(
                    self._explorer._last_state._feature_context)
                self._explorer._tracker._global.commit(
                    self._explorer._last_state._feature_mask)

                # feed input to target and populate state machine
                await self._explorer.follow(inp, minimize=False, atomic=True)
                info("Loaded seed file: %s", inp)
            except Exception as ex:
                warning("Failed to load %s: %s", inp, ex)
            self._snapshots[self._explorer._last_state].append(path)

        start = self._crosstest_timer.value
        await self.perform_inference()
        end = self._crosstest_timer.value
        self._batch_timeout = max(1.1 * self._batch_timeout, 2 * (end - start))
        info(f"Increased batch timeout to {self._batch_timeout}")

        state_to_path_mapping = {
            i[0]: paths
                for i in self._tracker.equivalence.states.items()
                # remove intermediate states for which no input file exists;
                # this occurs when an input traverses multiple snapshots,
                # arriving at a different final snapshot (only happens when
                # atomic==True). Without atomic==True, an input may result in
                # terminating the target and would not generate any states. As a
                # compromise, for such inputs, we consider the last snapshot
                # they arrive at before termination as their final snapshot
                # (which would otherwise be considered intermediate).
                if (paths := [p for s in i[1] for p in self._snapshots[s]])
        }
        await self.export_results(state_to_path_mapping)

    @property
    @abstractmethod
    def watch_path(self):
        pass

    @abstractmethod
    def select_file(self, path):
        pass

    @abstractmethod
    def select_siblings(self, siblings):
        pass

    @abstractmethod
    async def start_fuzzer(self):
        pass

    @abstractmethod
    async def stop_fuzzer(self, process):
        pass

    @abstractmethod
    async def import_inputs(self, paths):
        pass

    @abstractmethod
    async def export_results(self, state_to_path_mapping):
        pass

class NyxNetInference(HotplugInference):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['strategy'].get('hotplug') == 'nyx'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shared = os.environ['SHARED']
        target = os.environ['TARGETNAME']
        self._checksum = {
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
        }[target]
        self._sharedir = Path(f'{shared}/out-{target}-balanced-000')
        self._inputs = {}

    @property
    def watch_path(self):
        return self._sharedir / 'corpus/normal'

    def select_file(self, path):
        return path not in self._inputs and path.match('*.bin')

    def select_siblings(self, siblings):
        choices = set()
        sblgs = tuple(filter(
            lambda s: not self._tracker.equivalence.subsumers(s), siblings))
        choices.add(max(
            sblgs, key=lambda s: np.count_nonzero(np.asarray(s._raw_coverage))))
        choices.add(min(
            reversed(sblgs), key=lambda s: np.count_nonzero(np.asarray(s._raw_coverage))))
        choices.add(self._entropy.choice(tuple(siblings)))
        return choices

    async def start_fuzzer(self):
        await (await asyncio.create_subprocess_shell('set -x;'
            f'rm -rf "{self._sharedir}"')).wait()
        return await asyncio.create_subprocess_shell('set -x;'
            'cd "$FUZZER/targets/profuzzbench-nyx/scripts/nyx-eval";'
            './start.sh -c 0 -i 0 -T $TIMEOUT -p balanced -d "$SHARED"'
            ' -t "$TARGETNAME" $NYX_FUZZARGS',
            start_new_session=True)

    async def stop_fuzzer(self, process):
        os.killpg(process.pid, signal.SIGKILL)
        await process.wait()

    async def import_inputs(self, paths):
        await (await asyncio.create_subprocess_shell('set -x;'
            'cd "$FUZZER/targets/profuzzbench-nyx/scripts/nyx-eval";'
            './reproducible.sh -c 0 -i 0 -p balanced -d "$SHARED"'
            ' -t "$TARGETNAME" $NYX_FUZZARGS')).wait()

        reproducible = self._sharedir / 'reproducible'
        stems = {p.stem: p for p in paths}
        inputs = {}
        for file in reproducible.iterdir():
            if not (orig := stems.get(file.stem)):
                continue
            input = PreparedInput()
            exec(file.read_text(), globals() | {
                'packet': lambda data: input.append(
                    TransmitInstruction(bytes(data, 'latin-1')))
            }, locals())
            inputs[orig] = input
        self._inputs.update(inputs)

        # archive current queue to measure total coverage over time
        await (await asyncio.create_subprocess_shell('set -x;'
            f'cd "{self.watch_path.parent!s}";'
            'mkdir -p "$SHARED/archive";'
            R'tar czf "$SHARED/archive"/`date +%s`.tar.gz .')).wait()

        return inputs

    async def export_results(self, state_to_path_mapping):
        prog = re.compile(r'.*?_(\d+)')
        m = {
            i[0]: [int(prog.match(p.stem).group(1)) for p in i[1]]
                for i in state_to_path_mapping.items()
        }
        outfile = self._sharedir / 'inference.json'
        outfile.write_text(json.dumps(m))

        for snapshot in self._tracker.equivalence.mapped_snapshots:
            inp = self._explorer.get_reproducer(target=snapshot)
            p = self._sharedir / 'imports' / f'{hash(snapshot)}.bin'
            if not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                assert p.parent.is_dir()
            p.write_bytes(self._pack_input(inp))

    def _pack_input(self, input):
        graph_size = 0
        data = b''
        for instruction in input:
            if isinstance(instruction, TransmitInstruction) and instruction._data:
                data += struct.pack('H', len(instruction._data))
                data += instruction._data
                graph_size += 1

        data_size = len(data)
        base = 5 * 8
        graph_offset = base
        data_offset = graph_offset + struct.calcsize('H') * graph_size
        header = struct.pack("<QQQQQ",
            self._checksum, graph_size, data_size, graph_offset, data_offset)
        # FIXME this is spec-dependent; we'll assume node 0 is always raw
        graph_data = struct.pack("<"+ "H" * graph_size, *((0,) * graph_size))
        return header + graph_data + data

class AFLppInference(HotplugInference):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['strategy'].get('hotplug') == 'aflpp'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shared = os.environ['SHARED']
        self._out_dir = Path(f'{shared}/default')
        self._inputs = {}

    @property
    def watch_path(self):
        return self._out_dir / 'queue'

    def select_file(self, path):
        return path not in self._inputs and path.match('id:*')

    def select_siblings(self, siblings):
        choices = set()
        sblgs = tuple(filter(
            lambda s: not self._tracker.equivalence.subsumers(s), siblings))
        choices.add(max(
            sblgs, key=lambda s: np.count_nonzero(np.asarray(s._raw_coverage))))
        choices.add(min(
            reversed(sblgs), key=lambda s: np.count_nonzero(np.asarray(s._raw_coverage))))
        choices.add(self._entropy.choice(tuple(siblings)))
        return choices

    async def start_fuzzer(self):
        await (await asyncio.create_subprocess_shell('set -x;'
            f'rm -rf "{self._out_dir}"')).wait()
        return await asyncio.create_subprocess_shell('set -x;'
            'cd "$FUZZER/fuzzer";'
            'AFL_NO_UI=1 ./afl-fuzz -i "$TARGET/corpus/$PROGRAM" -o "$SHARED" -X'
            ' -F "$SHARED/imports" $AFL_FUZZARGS'
            ' -- "$OUT/nyx/packed/nyx_$TARGETNAME"',
            start_new_session=True)

    async def stop_fuzzer(self, process):
        os.killpg(process.pid, signal.SIGKILL)
        await process.wait()

    async def import_inputs(self, paths):
        reproducible = self._out_dir / 'queue'
        inputs = {}
        for file in paths:
            input = self._generator.load_input(str(file))
            inputs[file] = input
        self._inputs.update(inputs)

        # archive current queue to measure total coverage over time
        await (await asyncio.create_subprocess_shell('set -x;'
            f'cd "{self.watch_path.parent!s}";'
            'mkdir -p "$SHARED/archive";'
            R'tar czf "$SHARED/archive"/`date +%s`.tar.gz .')).wait()

        return inputs

    async def export_results(self, state_to_path_mapping):
        prog = re.compile(r'id.(\d{6}).*?')
        m = {
            state: [int(prog.match(p.stem).group(1)) for p in paths]
                for state, paths in state_to_path_mapping.items()
        }
        outfile = self._out_dir / 'inference.bin'

        with outfile.open('wb') as f:
            f.write(struct.pack('I', len(m)))
            for state, inputs in m.items():
                f.write(struct.pack('I', len(inputs)))
            for state, inputs in m.items():
                f.write(struct.pack('I', state))
                for i in inputs:
                    f.write(struct.pack('I', i))

        for snapshot in self._tracker.equivalence.mapped_snapshots:
            inp = self._explorer.get_reproducer(target=snapshot)
            p = self._out_dir.parent / 'imports' / f'{hash(snapshot)}.bin'
            if not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                assert p.parent.is_dir()
            p.write_bytes(self._pack_input(inp))

    def _pack_input(self, input):
        data = b''
        for instruction in input:
            if isinstance(instruction, TransmitInstruction):
                data += instruction._data
        return data