from tango.core.input import BaseDecorator

from random import Random
from contextlib import contextmanager

__all__ = ['BaseMutator']

class BaseMutator(BaseDecorator):
    def __init__(self, entropy: Random):
        self._entropy = entropy
        self._state0 = self._entropy.getstate()

    @property
    @contextmanager
    def entropy_ctx(self):
        # When two mutators are being sequenced simultaneously, the shared
        # entropy object is accessed by both, and depending on the access order,
        # it may change the outcome of each mutator. To solve this, we
        # essentially clone the entropy object, and on exit, we set it to the
        # state of one of the two cloned entropies (the last one to exit)
        entropy = SeedlessRandom(self._state0)
        self._temp = entropy

        try:
            yield entropy
        finally:
            self._entropy.setstate(self._temp.getstate())

class SeedlessRandom(Random):
    """
    A child class of Random that does not call super().__init__(). This skips
    the seed generation phase which can be time-consuming when a seed is not
    needed.
    """
    def __init__(self, state):
        self.gauss_next = None
        self.setstate(state)