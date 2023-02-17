from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union

class AbstractInput(ABC):
    _COUNTER = 0

    def __init__(self):
        self.id = self.uniq_id

    @classmethod
    @property
    def uniq_id(cls):
        cls._COUNTER += 1
        return cls._COUNTER

    ## Abstract methods ##
    @property
    @abstractmethod
    def decorated(self):
        pass

    @abstractmethod
    def ___iter___(self):
        pass

    @abstractmethod
    async def ___aiter___(self):
        pass

    @abstractmethod
    def ___repr___(self):
        pass

    @abstractmethod
    def ___len___(self):
        pass

    @abstractmethod
    def ___eq___(self, other: AbstractInput):
        pass

    @abstractmethod
    def ___getitem___(self, idx: Union[int, slice]):
        pass

    @abstractmethod
    def ___add___(self, other: AbstractInput):
        pass

    ## Decoratable functions ##
    # These definitions are needed so that a decorator can override the behavior
    # of "special methods". The python interpreter does not resolve special
    # methods as it would other class attributes, so a level of indirection is
    # needed to bypass this limitation.
    def __iter__(self):
        return self.___iter___()

    def __aiter__(self):
        return self.___aiter___()

    def __repr__(self):
        return self.___repr___()

    def __len__(self):
        return self.___len___()

    def __eq__(self, other):
        return self.___eq___(other)

    def __getitem__(self, idx):
        return self.___getitem___(idx)

    def __add__(self, other):
        return self.___add___(other)