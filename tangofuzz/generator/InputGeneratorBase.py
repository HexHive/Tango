from . import warning
from abc import ABC, abstractmethod
from collections.abc import Iterable
from tracker import StateBase
from dataio   import ChannelFactoryBase
from input import InputBase, Serializer, PreparedInput, FormatDescriptor
from profiler import ProfileFrequency
from typing import Sequence
from functools import reduce
import os
import unicodedata
import re
import operator

class InputGeneratorBase(ABC):
    def __init__(self, startup: str, seed_dir: str, work_dir: str, fmt: FormatDescriptor):
        self._seed_dir = seed_dir
        self._work_dir = work_dir

        # FIXME maybe add an EmptyInput class
        self._startup = PreparedInput()
        self._seeds = []

        self._fmt = fmt
        self._input_kls = Serializer.get(fmt)
        if self._input_kls is None:
            warning(f"No serializer available for `{self._fmt.typ}`!")
            return

        if startup and os.path.isfile(startup):
            self._startup = self._input_kls(file=startup, load=True)

        if self._seed_dir is None or not os.path.isdir(self._seed_dir):
            return

        seed_files = []
        for root, _, files in os.walk(self._seed_dir):
            seed_files.extend(os.path.join(root, file) for file in files)

        for seed in seed_files:
            input = self._input_kls(file=seed, load=True)
            self._seeds.append(input)

    def save_input(self, input: InputBase,
            prefix_path: Sequence[tuple[StateBase, StateBase, InputBase]],
            category: str, label: str, filepath: str=None):
        if self._input_kls is None:
            return

        prefix = reduce(operator.add, (x[2] for x in prefix_path))
        full_input = self.startup_input + prefix + input
        long_name = f'[{label}] {repr(input)}'

        if filepath is None:
            filename = slugify(f'0x{input.id:08X}.{self._fmt.typ}')
            filepath = os.path.join(self._work_dir, category, filename)
        self._input_kls(file=filepath).dump(full_input, name=long_name)

    def load_input(self, filepath: str) -> InputBase:
        if self._input_kls is None:
            raise RuntimeError(f"Cannot deserialize `{self._fmt.typ}.`")
        return self._input_kls(file=filepath).load()

    def update_state(self, state: StateBase, input: InputBase, *, exc: Exception=None, **kwargs):
        pass

    def update_transition(self, source: StateBase, destination: StateBase, input: InputBase, *, state_changed: bool, exc: Exception=None, **kwargs):
        pass

    @ProfileFrequency('gens')
    def generate(self, *args, **kwargs) -> InputBase:
        return self.generate_internal(*args, **kwargs)

    @abstractmethod
    def generate_internal(self, state: StateBase, entropy) -> InputBase:
        pass

    @property
    def seeds(self) -> Iterable[InputBase]:
        # the default behavior is to just expose the list of loaded inputs
        return self._seeds

    @property
    def startup_input(self) -> InputBase:
        return self._startup

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
