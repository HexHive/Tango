from generator import InputGeneratorBase
from networkio   import ChannelFactoryBase
from statemanager import StateBase
from input import InputBase, PreparedInput, CachingDecorator
from random import Random
from mutator import HavocMutator

class RandomInputGenerator(InputGeneratorBase):
    def __init__(self, initial: str, seed_dir: str, ch_env: ChannelFactoryBase):
        super().__init__(initial, seed_dir, ch_env)

    def generate(self, state: StateBase, entropy: Random) -> InputBase:
        # TODO move escapers from state to state tracker/generator?
        candidate = state.get_escaper()
        if candidate is None:
            if self.seeds:
                candidate = entropy.choice(self.seeds)
            else:
                candidate = PreparedInput()

        return CachingDecorator()(HavocMutator(entropy)(candidate), copy=False)