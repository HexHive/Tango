from abc import ABC, abstractmethod
from collections.abc import Iterable
from statemanager import StateBase
from networkio   import ChannelFactoryBase
from input import InputBase, PCAPInput, PreparedInput
import os

class InputGeneratorBase(ABC):
    def __init__(self, startup: str, seed_dir: str, ch_env: ChannelFactoryBase):
        self._seed_dir = seed_dir
        self._ch_env = ch_env

        if startup and os.path.isfile(startup):
            self._startup = PCAPInput(startup, self._ch_env)
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
            input = PCAPInput(seed, self._ch_env)
            self._pcaps.append(input)

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