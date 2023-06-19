from . import debug, info, warning

from tango.core import AbstractInput, PreparedInput, TransmitInstruction
from tango.inference import StateInferenceStrategy

from asyncinotify import Inotify, Mask, Event as InotifyEvent

from abc import abstractmethod
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from itertools import count
import numpy as np
import asyncio
import struct
import os

@dataclass
class InotifyBatchObserver:
    inotify: Inotify
    timeout: int
    batch_size: int
    select: Callable[[InotifyEvent], bool] = lambda _: True

    def __post_init__(self):
        self._counter = 0

    async def get_event(self):
        while True:
            if self.select(ev := await anext(aiter(self.inotify))):
                return ev

    async def get_batch(self, batch: Optional[list]=None):
        for _ in range(self.batch_size):
            ev = await self.get_event()
            info(f"Observed inotify event: {ev}")
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
        capture_paths=('strategy.batch_timeout')):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            'hotplug' in config['strategy']

    def __init__(self, *,
            batch_timeout: int,
            **kwargs):
        super().__init__(**kwargs)
        self._queue = Path(path_to_queue)
        self._batch_timeout = batch_timeout

    async def step(self, input: Optional[AbstractInput]=None):
        fuzzer = await self.start_fuzzer()
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

        await self.stop_fuzzer(fuzzer)

        inputs = await self.import_inputs(batch)
        for inp in inputs:
            try:
                await self._explorer.reload_state()
                # feed input to target and populate state machine
                await self._explorer.follow(inp, minimize=True)
                info("Loaded seed file: %s", inp)
            except LoadedException as ex:
                warning("Failed to load %s: %s", inp, ex.exception)

        await self.perform_inference()

        culled = []
        for sid, sblgs in self._tracker.equivalence.states.items():
            for sblg in self.select_siblings(sblgs):
                inp = self._explorer.get_reproducer(target=sblg)
                inp = self._generator.startup_input + inp
                culled.append(inp)
        await self.export_inputs(culled)

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
    async def export_inputs(self, inputs):
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

    @property
    def watch_path(self):
        return self._sharedir / 'corpus/normal'

    def select_file(self, path):
        return path.match('cov_*.bin')

    def select_siblings(self, siblings):
        sblgs = filter(
            lambda s: not self._tracker.equivalence.subsumers(s), siblings)
        yield (most := max(
            sblgs, key=lambda s: np.count_nonzero(np.asarray(s._raw_coverage))))
        least = min(
            sblgs, key=lambda s: np.count_nonzero(np.asarray(s._raw_coverage)))
        if least is not most:
            yield least

    async def start_fuzzer(self):
        await (await asyncio.create_subprocess_shell('set -x;'
            f'rm -rf "{self._sharedir}"')).wait()
        return await asyncio.create_subprocess_shell('set -x;'
            'cd "$FUZZER/targets/profuzzbench-nyx/scripts/nyx-eval";'
            './start.sh -c 0 -i 0 -T $TIMEOUT -p balanced -d "$SHARED"'
            ' -t "$TARGETNAME" $NYX_FUZZARGS')

    async def stop_fuzzer(self, process):
        process.terminate()
        await process.wait()

    async def import_inputs(self, paths):
        await (await asyncio.create_subprocess_shell('set -x;'
            'cd "$FUZZER/targets/profuzzbench-nyx/scripts/nyx-eval";'
            './reproducible.sh -c 0 -i 0 -p balanced -d "$SHARED"'
            ' -t "$TARGETNAME" $NYX_FUZZARGS')).wait()

        reproducible = self._sharedir / 'reproducible'
        stems = {p.stem for p in paths}
        inputs = []
        for file in reproducible.iterdir():
            if file.stem not in stems:
                continue
            input = PreparedInput()
            exec(file.read_text(), globals=globals() | {
                'packet': lambda b: input.append(
                    TransmitInstruction(bytes(b, 'latin-1')))
            })
            inputs.append(input)
        return inputs

    async def export_inputs(self, inputs):
        await (await asyncio.create_subprocess_shell('set -x;'
            f'rm -f "$OUT/nyx/packed/nyx_$TARGETNAME/seeds"/*')).wait()
        for i, inp in enumerate(inputs):
            packed = self._pack_input(inp)
            Path(os.environ['OUT'], 'packed', f'nyx_{os.environ["TARGETNAME"]}',
                'seeds', f'seed_{i}.bin').write_bytes(packed)

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