from typing import Sequence
from input import InputBase
from mutator import MutatorBase
from interaction import (InteractionBase, TransmitInteraction,
                        ReceiveInteraction, DelayInteraction)
from random import Random
from itertools import tee, chain
from enum import Enum

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
        self._reorder = []

    def ___iter___(self, orig):
        old_state = self._entropy.getstate()
        self._entropy.setstate(self._state0)

        for i, interaction in enumerate(chain(orig(), (None,))):
            seq = self._mutate(interaction)
            yield from seq

        # self._entropy.setstate(old_state)

    def ___repr___(self, orig):
        return f'HavocMutatedInput:{self._input.id} ({self._input_id})'

    def mutate_int(self, n: int):
        def to_bytes(n, order='little'):
            nbytes = (n.bit_length() - 1) // 8 + 1
            return n.to_bytes(nbytes, order)
        def from_bytes(buf, order='little'):
            return int.from_bytes(buf, order)
        buffer_n = bytearray(to_bytes(n))
        self.mutate_buffer(buffer_n)
        return from_bytes(buffer_n)

    def mutate_buffer(self, buffer: bytearray):
        # optionally extend buffer, but highest weight is for no-extend
        ext = b'\0' * self._entropy.choices(
                range(8),
                cum_weights=range(92, 100)
            )[0]
        buffer.extend(ext)

        # ultra efficient mutator, courtesy of Brandon Falk
        for _ in range(self._entropy.randint(1, 8)):
            buffer[self._entropy.randint(0, len(buffer) - 1)] = \
                self._entropy.randint(0, 255)

    def _mutate(self, interaction: InteractionBase) -> Sequence[InteractionBase]:
        if interaction is not None:
            interaction.mutate(self)
            # TODO perform random operation
            low = 0
            for _ in range(self._entropy.randint(1, 4)):
                if low > 4:
                    return
                oper = self.RandomOperation(self._entropy.randint(low, 4))
                low = oper.value + 1
                if oper == self.RandomOperation.DELETE:
                    return
                elif oper == self.RandomOperation.PUSHORDER:
                    self._reorder.append(interaction)
                    return
                elif oper == self.RandomOperation.POPORDER:
                    if self._reorder:
                        yield self._reorder.pop()
                elif oper == self.RandomOperation.REPEAT:
                    yield from (interaction for _ in range(2))
                elif oper == self.RandomOperation.CREATE:
                    # FIXME use range(3) to enable delays, but they affect throughput
                    inter = self.RandomInteraction(self._entropy.choices(
                            range(2),
                            cum_weights=range(998, 1000)
                        )[0])
                    if inter == self.RandomInteraction.TRANSMIT:
                        buffer = self._entropy.randbytes(self._entropy.randint(1, 256))
                        new = TransmitInteraction(buffer)
                    elif inter == self.RandomInteraction.RECEIVE:
                        new = ReceiveInteraction()
                    elif inter == self.RandomInteraction.DELAY:
                        delay = self._entropy.random() * 5
                        new = DelayInteraction(delay)
                    yield new
        else:
            oper = self.RandomOperation(self._entropy.randint(4, 4))
            if oper == self.RandomOperation.CREATE:
                # FIXME use range(3) to enable delays, but they affect throughput
                inter = self.RandomInteraction(self._entropy.choices(
                        range(2),
                        cum_weights=range(998, 1000)
                    )[0])
                if inter == self.RandomInteraction.TRANSMIT:
                    buffer = self._entropy.randbytes(self._entropy.randint(1, 256))
                    new = TransmitInteraction(buffer)
                elif inter == self.RandomInteraction.RECEIVE:
                    new = ReceiveInteraction()
                elif inter == self.RandomInteraction.DELAY:
                    delay = self._entropy.random() * 5
                    new = DelayInteraction(delay)
                yield new

            # when interaction is None, we should flush the reorder buffer
            yield from self._entropy.sample(self._reorder, k=len(self._reorder))
            self._reorder.clear()
