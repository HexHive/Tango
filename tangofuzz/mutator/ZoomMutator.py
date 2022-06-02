from . import debug

from typing import Sequence
from statemanager import StateBase
from input import InputBase
from mutator import MutatorBase
from interaction import (InteractionBase, MoveInteraction, ShootInteraction,
                        ActivateInteraction, DelayInteraction,
                        RotateInteraction, ReachInteraction, KillInteraction)
from random import Random
from itertools import tee
from enum import Enum
from copy import deepcopy

class ZoomMutator(MutatorBase):
    class RandomOperation(Enum):
        DELETE = 0
        PUSHORDER = 1
        POPORDER = 2
        REPEAT = 3
        CREATE = 4

    class RandomInteraction(Enum):
        MOVE = 0
        ACTIVATE = 1
        SHOOT = 2
        DELAY = 3

    def __init__(self, entropy: Random, state: StateBase):
        super().__init__(entropy)
        self._clobbers = (
                ("forward", "backward"),
                ("strafe_left", "strafe_right", "rotate_left", "rotate_right")
            )
        self._state = state

    def __enter__(self):
        entropy = Random()
        entropy.setstate(self._state0)
        self._temp = entropy
        return entropy

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._entropy.setstate(self._temp.getstate())

    def ___iter___(self, orig):
        last = None
        for mut in self.___iter_helper___(orig):
            if isinstance(mut, (RotateInteraction, ReachInteraction)):
                # deliberately ignore interactions that the strategy added to the
                # SM because they're not entirely useful for random exploration
                continue

            if isinstance(last, ActivateInteraction) and isinstance(mut, ActivateInteraction):
                continue
            yield mut
            last = mut
            # # we always try to kill
            # yield KillInteraction(self._state)

    def ___iter_helper___(self, orig):
        with self as entropy:
            i = -1
            reorder_buffer = []
            for i, interaction in enumerate(orig()):
                if isinstance(interaction, DelayInteraction):
                    yield interaction
                    continue
                new_interaction = deepcopy(interaction)
                seq = self._mutate(new_interaction, reorder_buffer, entropy)
                yield from seq
            else:
                if i == -1:
                    yield from self._mutate(None, reorder_buffer, entropy)

    def ___repr___(self, orig):
        return f'ZoomMutatedInput:0x{self._input.id:08X} (0x{self._input_id:08X})'

    def mutate_int(self, n: int, entropy: Random):
        def to_bytes(n, order='little'):
            nbytes = (n.bit_length() - 1) // 8 + 1
            return n.to_bytes(nbytes, order)
        def from_bytes(buf, order='little'):
            return int.from_bytes(buf, order)
        buffer_n = bytearray(to_bytes(n))
        self.mutate_buffer(buffer_n, entropy)
        return from_bytes(buffer_n)

    def mutate_buffer(self, buffer: bytearray, entropy: Random):
        # optionally extend buffer, but highest weight is for no-extend
        ext = b'\0' * entropy.choices(
                range(8),
                cum_weights=range(92, 100)
            )[0]
        buffer.extend(ext)

        # ultra efficient mutator, courtesy of Brandon Falk
        if buffer:
            if len(buffer) == 1:
                buffer[0] = entropy.randint(0, 255)
            else:
                for _ in range(entropy.randint(1, 8)):
                    buffer[entropy.randint(0, len(buffer) - 1)] = \
                        entropy.randint(0, 255)

    def _mutate(self, interaction: InteractionBase, reorder_buffer: Sequence, entropy: Random) -> Sequence[InteractionBase]:
        if interaction is not None:
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
                            range(1),
                            cum_weights=(4,)#, 7, 9, 10)
                        )[0])
                    if inter == self.RandomInteraction.MOVE:
                        duration = entropy.random() * 2
                        new = MoveInteraction(None, duration=duration)
                        new.mutate(self, entropy)
                        yield new
                    elif inter == self.RandomInteraction.ACTIVATE:
                        yield ActivateInteraction()
                    elif inter == self.RandomInteraction.SHOOT:
                        yield ShootInteraction()
                    elif inter == self.RandomInteraction.DELAY:
                        delay = entropy.random() * 2
                        yield DelayInteraction(delay)
        else:
            oper = self.RandomOperation(entropy.randint(4, 4))
            if oper == self.RandomOperation.CREATE:
                # FIXME use range(3) to enable delays, but they affect throughput
                inter = self.RandomInteraction(entropy.choices(
                        range(1),
                        cum_weights=(4,)#, 7, 9, 10)
                    )[0])
                if inter == self.RandomInteraction.MOVE:
                    duration = entropy.random() * 2
                    new = MoveInteraction(None, duration=duration)
                    new.mutate(self, entropy)
                    yield new
                elif inter == self.RandomInteraction.ACTIVATE:
                    yield ActivateInteraction()
                elif inter == self.RandomInteraction.SHOOT:
                    yield ShootInteraction()
                elif inter == self.RandomInteraction.DELAY:
                    delay = entropy.random() * 2
                    yield DelayInteraction(delay)

            # when interaction is None, we should flush the reorder buffer
            yield from entropy.sample(reorder_buffer, k=len(reorder_buffer))
            reorder_buffer.clear()