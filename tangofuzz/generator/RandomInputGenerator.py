from generator import InputGeneratorBase
from dataio   import ChannelFactoryBase
from tracker import StateBase
from input import InputBase, PreparedInput
from random import Random
from mutator import HavocMutator

class RandomInputGenerator(InputGeneratorBase):
    def __init__(self, startup: str, seed_dir: str, protocol: str):
        super().__init__(startup, seed_dir, protocol)

    def generate_internal(self, state: StateBase, entropy: Random) -> InputBase:
        out_edges = list(state.out_edges)
        if out_edges:
            _, dst, data = entropy.choice(out_edges)
            candidate = data['minimized']
        else:
            in_edges = list(state.in_edges)
            if in_edges:
                _, dst, data = entropy.choice(in_edges)
                candidate = data['minimized']
            elif self.seeds:
                candidate = entropy.choice(self.seeds)
            else:
                candidate = PreparedInput()

        return HavocMutator(entropy)(candidate)