from generator import BaseInputGenerator
from tracker import AbstractState
from input import AbstractInput, PreparedInput
from random import Random
from mutator import HavocMutator

class RandomInputGenerator(BaseInputGenerator):
    def generate(self, state: AbstractState, entropy: Random) -> AbstractInput:
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
