from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
from input import JoiningDecorator, SlicingDecorator

class InputBase(ABC):
    @abstractmethod
    def ___iter___(self):
        pass

    def ___repr___(self):
        return f"{self.__class__.__name__}"

    def __eq__(self, other):
        diff = False

        def zip_strict(*iterators):
            for x in zip(*iterators):
                yield x
            failed = False
            for x in iterators:
                try:
                    next(x)
                    failed = True
                    break
                except:
                    pass
            return failed

        def eq(*iterators):
            nonlocal diff
            diff = yield from zip_strict(*iterators)

        return all(type(x) == type(y) and x == y
                    for x, y in eq(iter(self), iter(other))) and \
               not diff

    def __add__(self, other: InputBase):
        return JoiningDecorator(other)(self)

    def __iadd__(self, other: InputBase):
        return JoiningDecorator(other)(self, copy=False)

    def __getitem__(self, idx: Union[int, slice]):
        return SlicingDecorator(idx)(self)

    ## Decoratable functions ##
    # These definitions are needed so that a decorator can override the behavior
    # of "special methods". The python interpreter does not resolve special
    # methods as it would other class attributes, so a level of indirection is
    # needed to bypass this limitation.
    def __iter__(self):
        yield from self.___iter___()

    def __repr__(self):
        return self.___repr___()