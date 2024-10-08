from . import debug, info, warning, critical
from tango.core.profiler import FrequencyProfiler
from tango.core.tracker  import AbstractState, IUpdateCallback
from tango.core.dataio   import FormatDescriptor, AbstractChannelFactory
from tango.core.input import AbstractInput, Serializer, EmptyInput
from tango.common import AsyncComponent, ComponentType, ComponentOwner

from abc import ABC, abstractmethod
from random import Random
from typing import Sequence, Iterable, Optional
from functools import reduce
import os
import unicodedata
import re

__all__ = ['AbstractInputGenerator', 'BaseInputGenerator']

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s\.-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

class AbstractInputGenerator(AsyncComponent, IUpdateCallback, ABC,
        component_type=ComponentType.generator):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if (target := cls.__dict__.get('generate')):
            setattr(cls, 'generate', FrequencyProfiler('gens')(target))

    @abstractmethod
    def generate(self, state: AbstractState, entropy: Random) -> AbstractInput:
        pass

class BaseInputGenerator(AbstractInputGenerator,
        capture_components={ComponentType.channel_factory},
        capture_paths=['generator.startup', 'generator.seeds', 'fuzzer.work_dir']):
    def __init__(self, *, work_dir: str, channel_factory: AbstractChannelFactory,
            startup: Optional[str]=None, seeds: Optional[str]=None, **kwargs):
        super().__init__(**kwargs)
        self._seed_dir = seeds
        self._work_dir = work_dir
        self._startup_path = startup
        self._fmt: FormatDescriptor = channel_factory.fmt

        self._startup = EmptyInput()
        self._seeds = []

    async def initialize(self):
        debug("TBD")
        self._input_kls = Serializer.get(self._fmt)
        if self._input_kls is None:
            warning("No serializer available for `%s`!", self._fmt.typ)
            return

        if self._startup_path and os.path.isfile(self._startup_path):
            self._startup = self._input_kls(file=self._startup_path, load=True)

        if self._seed_dir is None or not os.path.isdir(self._seed_dir):
            return

        seed_files = []
        for root, _, files in os.walk(self._seed_dir):
            seed_files.extend(os.path.join(root, file) for file in files)

        for seed in seed_files:
            input = self._input_kls(file=seed, load=True)
            self._seeds.append(input)

    async def finalize(self, owner: ComponentOwner):
        debug("Done nothing")
        return await super().finalize(owner)

    def select_candidate(self, state: AbstractState):
        out_edges = (inp for _,_,inp in state.out_edges)
        in_edges = (inp for _,_,inp in state.in_edges)
        candidates = (*out_edges, *in_edges, *self.seeds, EmptyInput())
        return self._entropy.choice(candidates)

    def save_input(self,
            input: AbstractInput, category: str, label: str, filepath: str=None):
        if self._input_kls is None:
            return

        full_input = self.startup_input + input
        long_name = f'[{label}] {repr(input)}'

        if filepath is None:
            filename = slugify(f'0x{input.id:08X}.{self._fmt.typ}.tango')
            filepath = os.path.join(self._work_dir, category, filename)
        self._input_kls(file=filepath).dump(full_input, name=long_name)

    def load_input(self, filepath: str) -> AbstractInput:
        if self._input_kls is None:
            raise RuntimeError(f"Cannot deserialize `{self._fmt.typ}.`")
        return self._input_kls(file=filepath).load()

    @property
    def seeds(self) -> Iterable[AbstractInput]:
        # the default behavior is to just expose the list of loaded inputs
        return self._seeds

    @property
    def startup_input(self) -> AbstractInput:
        return self._startup

    def update_state(self, state: AbstractState, /, *, input: AbstractInput,
            exc: Exception=None, **kwargs):
        pass

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, /, *,
            state_changed: bool, exc: Exception=None, **kwargs):
        pass
