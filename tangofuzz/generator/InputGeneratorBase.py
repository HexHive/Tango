from abc import ABC, abstractmethod
from collections.abc import Iterable
from tracker import StateBase
from dataio   import ChannelFactoryBase
from input import InputBase, PCAPInput, PreparedInput
import os

class InputGeneratorBase(ABC):
    def __init__(self, startup: str, seed_dir: str, protocol: str):
        self._seed_dir = seed_dir
        self._protocol = protocol

        if startup and os.path.isfile(startup):
            self._startup = PCAPInput(startup, protocol=self._protocol)
        else:
            # FIXME maybe add an EmptyInput class
            self._startup = PreparedInput()

        self._pcaps = []
        if self._seed_dir is None or not os.path.isdir(self._seed_dir):
            return

        seeds = []
        for root, _, files in os.walk(self._seed_dir):
            seeds.extend(os.path.join(root, file) for file in files)

        for seed in seeds:
            # parse seed to PreparedInput
            input = PCAPInput(seed, protocol=self._protocol)
            self._pcaps.append(input)

    def update_state(self, state: StateBase, input: InputBase, *, exc: Exception=None, **kwargs):
        pass

    def update_transition(self, source: StateBase, destination: StateBase, input: InputBase, *, state_changed: bool, exc: Exception=None, **kwargs):
        pass

    @abstractmethod
    def generate(self, state: StateBase, entropy) -> InputBase:
        pass

    @property
    def seeds(self) -> Iterable[InputBase]:
        # the default behavior is to just expose the raw PCAPInputs
        return self._pcaps

    @property
    def startup_input(self) -> InputBase:
        return self._startup