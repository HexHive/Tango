from typing import Sequence
from input import InputBase
from mutator import MutatorBase
from interaction import (InteractionBase, TransmitInteraction,
                        ReceiveInteraction, DelayInteraction)
from random import Random
from itertools import tee
from enum import Enum
from copy import deepcopy

class HavocMutator(MutatorBase):
    class RandomOperation(Enum):
        DELETE = 0
        PUSHORDER = 1
        POPORDER = 2
        REPEAT = 3
        CREATE = 4

    class RandomInteraction(Enum):
        TRANSMIT = 0
        RECEIVE = 1
        DELAY = 2

    def __init__(self, entropy: Random):
        super().__init__(entropy)

    def __enter__(self):
        entropy = Random()
        entropy.setstate(self._state0)
        self._temp = entropy
        return entropy

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._entropy.setstate(self._temp.getstate())

    def ___iter___(self, orig):
        with self as entropy:
            i = -1
            reorder_buffer = []
            for i, interaction in enumerate(orig()):
                new_interaction = deepcopy(interaction)
                seq = self._mutate(new_interaction, reorder_buffer, entropy)
                yield from seq
            else:
                if i == -1:
                    yield from self._mutate(None, reorder_buffer, entropy)

    def ___repr___(self, orig):
        return f'HavocMutatedInput:0x{self._input.id:08X} (0x{self._input_id:08X})'

    def _mutate(self, interaction: InteractionBase, reorder_buffer: Sequence, entropy: Random) -> Sequence[InteractionBase]:
        if interaction is not None:
            raise NotImplementedError("InteractionBase.mutate was removed!")
            interaction.mutate(self, entropy)
            # TODO perform random operation
            low = 0
            for _ in range(entropy.randint(1, 4)):
                if low > 4:
                    return
                oper = self.RandomOperation(entropy.randint(low, 4))
                low = oper.value + 1
                if oper == self.RandomOperation.DELETE:
                    return
                elif oper == self.RandomOperation.PUSHORDER:
                    reorder_buffer.append(interaction)
                    return
                elif oper == self.RandomOperation.POPORDER:
                    if reorder_buffer:
                        yield reorder_buffer.pop()
                elif oper == self.RandomOperation.REPEAT:
                    yield from (interaction for _ in range(2))
                elif oper == self.RandomOperation.CREATE:
                    # FIXME use range(3) to enable delays, but they affect throughput
                    inter = self.RandomInteraction(entropy.choices(
                            range(2),
                            cum_weights=range(998, 1000)
                        )[0])
                    if inter == self.RandomInteraction.TRANSMIT:
                        buffer = entropy.randbytes(entropy.randint(1, 256))
                        new = TransmitInteraction(buffer)
                    elif inter == self.RandomInteraction.RECEIVE:
                        new = ReceiveInteraction()
                    elif inter == self.RandomInteraction.DELAY:
                        delay = entropy.random() * 5
                        new = DelayInteraction(delay)
                    yield new
        else:
            oper = self.RandomOperation(entropy.randint(4, 4))
            if oper == self.RandomOperation.CREATE:
                # FIXME use range(3) to enable delays, but they affect throughput
                inter = self.RandomInteraction(entropy.choices(
                        range(2),
                        cum_weights=range(998, 1000)
                    )[0])
                if inter == self.RandomInteraction.TRANSMIT:
                    buffer = entropy.randbytes(entropy.randint(1, 256))
                    new = TransmitInteraction(buffer)
                elif inter == self.RandomInteraction.RECEIVE:
                    new = ReceiveInteraction()
                elif inter == self.RandomInteraction.DELAY:
                    delay = entropy.random() * 5
                    new = DelayInteraction(delay)
                yield new

            # when interaction is None, we should flush the reorder buffer
            yield from entropy.sample(reorder_buffer, k=len(reorder_buffer))
            reorder_buffer.clear()
