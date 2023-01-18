from . import debug, critical
from typing import Sequence, Iterable
from input import InputBase
from mutator import MutatorBase
from interaction import (InteractionBase, TransmitInteraction,
                        ReceiveInteraction, DelayInteraction)
from random import Random
from itertools import tee
from enum import Enum
from copy import deepcopy

class ReactiveHavocMutator(MutatorBase):
    class RandomOperation(Enum):
        DELETE = 0
        PUSHORDER = 1
        POPORDER = 2
        REPEAT = 3
        CREATE = 4
        MUTATE = 5

    class RandomInteraction(Enum):
        TRANSMIT = 0
        RECEIVE = 1
        DELAY = 2

    def __init__(self, entropy: Random, havoc_actions: Iterable):
        super().__init__(entropy)
        self._actions = havoc_actions
        self._actions_taken = False

    def __enter__(self):
        # When two mutators are being sequenced simultaneously, the shared
        # entropy object is accessed by both, and depending on the access order,
        # it may change the outcome of each mutator. To solve this, we
        # essentially clone the entropy object, and on exit, we set it to the
        # state of one of the two cloned entropies (the last one to exit)
        entropy = Random()
        entropy.setstate(self._state0)
        self._temp = entropy
        return entropy

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._entropy.setstate(self._temp.getstate())

    def _iter_helper(self, orig):
        with self as entropy:
            i = -1
            reorder_buffer = []
            for i, interaction in enumerate(orig()):
                new_interaction = deepcopy(interaction)
                seq = self._mutate(new_interaction, reorder_buffer, entropy)
                yield from seq
            if i == -1:
                yield from self._mutate(None, reorder_buffer, entropy)

            # finally, we flush the reorder buffer
            yield from entropy.sample(reorder_buffer, k=len(reorder_buffer))
            reorder_buffer.clear()

    def ___iter___(self, orig):
        self._actions_taken = False
        for interaction in self._iter_helper(orig):
            yield interaction
            self._actions_taken = False

    def ___repr___(self, orig):
        return f'HavocMutatedInput:0x{self._input.id:08X} (0x{self._input_id:08X})'

    def _apply_actions(self, data, entropy):
        for func in self._actions:
            data = bytearray(data)
            data = func(data, entropy)
        self._actions_taken = True
        return data

    def _mutate(self, interaction: InteractionBase, reorder_buffer: Sequence, entropy: Random) -> Sequence[InteractionBase]:
        if interaction is not None:
            low = 0
            for _ in range(entropy.randint(3, 7)):
                if low > 5:
                    return
                oper = self.RandomOperation(entropy.randint(low, 5))
                low = oper.value + 1
                if oper == self.RandomOperation.DELETE:
                    return
                elif oper == self.RandomOperation.PUSHORDER:
                    reorder_buffer.append(interaction)
                    return
                elif oper == self.RandomOperation.POPORDER:
                    yield interaction
                    if reorder_buffer:
                        yield from self._mutate(reorder_buffer.pop(), reorder_buffer, entropy)
                elif oper == self.RandomOperation.REPEAT:
                    yield from (interaction for _ in range(2))
                elif oper == self.RandomOperation.CREATE:
                    buffer = entropy.randbytes(entropy.randint(1, 256))
                    yield TransmitInteraction(buffer)
                elif oper == self.RandomOperation.MUTATE:
                    if isinstance(interaction, TransmitInteraction):
                        interaction._data = self._apply_actions(interaction._data, entropy)
                    else:
                        # no mutations on other interaction types for now
                        pass
                    yield interaction
        else:
            buffer = entropy.randbytes(entropy.randint(1, 256))
            yield TransmitInteraction(buffer)
