from abc import abstractmethod
from input import DecoratorBase
from random import Random
from copy import deepcopy

class MutatorBase(DecoratorBase):
    def __init__(self, entropy: Random):
        self._entropy = entropy
        self._state0 = self._entropy.getstate()

    def __deepcopy__(self, memo):
        # make sure that the entropy object is shared across mutators;
        # otherwise, the state of the PRNG would not change
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != '_entropy':
                setattr(result, k, deepcopy(v, memo))
        setattr(result, '_entropy', self._entropy)
        return result