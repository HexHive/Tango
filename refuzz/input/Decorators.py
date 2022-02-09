from abc import ABC
from functools import partial, reduce
from itertools import islice, tee, chain
from copy import deepcopy, copy
import inspect
import unicodedata
import re
import os
import io
import operator

class DecoratorBase(ABC):
    def __call__(self, input, copy=True): # -> InputBase:
        self._handle_copy(input, copy)

        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name not in ('___iter___', '___eq___', '___add___', '___getitem___', '___repr___', '___len___'):
                continue
            oldfunc = getattr(self._input, name, None)
            newfunc = partial(func, oldfunc)
            setattr(self._input, name, newfunc)

        setattr(self._input, '___decorator___', self)
        return self._input

    def _handle_copy(self, input, copy):
        self._input_id = input.id
        if copy:
            self._input = deepcopy(input)
            self._input.id = self._input.uniq_id
        else:
            self._input = input

    def ___repr___(self, orig):
        return f"{self.__class__.__name__}({orig()})"

    def undecorate(self):
        assert getattr(self, '_input', None) is not None, \
                "Decorator has not been called before!"
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if name not in ('___iter___', '___eq___', '___add___', '___getitem___', '___repr___', '___len___'):
                continue
            newfunc = getattr(self._input, name, None)
            oldfunc = newfunc.args[0]
            setattr(self._input, name, oldfunc)
        del self._input.___decorator___

    @classmethod
    def get_parent_class(cls, method):
        """
        Gets the class that defined the method.
        Courtesy of:
        https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545
        """
        if isinstance(method, partial):
            return cls.get_parent_class(method.func)
        if inspect.ismethod(method) or (inspect.isbuiltin(method) and getattr(method, '__self__', None) is not None and getattr(method.__self__, '__class__', None)):
            for cls in inspect.getmro(method.__self__.__class__):
                if method.__name__ in cls.__dict__:
                    return cls
            method = getattr(method, '__func__', method)  # fallback to __qualname__ parsing
        return getattr(method, '__objclass__', None)  # handle special descriptor objects

class MemoryCachingDecorator(DecoratorBase):
    def __init__(self):
        self._cached_iter = None
        self._cached_repr = None

    def __call__(self, input, copy=True): # -> InputBase:
        self._handle_copy(input, copy)

        self._cached_repr = repr(input)
        self._cached_iter = tuple(input)
        self._cached_len = len(self._cached_iter)
        self._input.___repr___ = self.___cached_repr___
        self._input.___iter___ = self.___cached_iter___
        self._input.___len___ = self.___cached_len___

        # we delete self._input because it is no longer needed, and a dangling
        # reference to it will result in redundant deepcopy-ing later
        inp = self._input
        del self._input
        return inp

    def ___cached_iter___(self):
        yield from self._cached_iter

    def ___cached_repr___(self):
        return self._cached_repr

    def ___cached_len___(self):
        return self._cached_len

class FileCachingDecorator(MemoryCachingDecorator):
    IO_BUFFER_SIZE = 0

    def __init__(self, workdir, subdir, protocol):
        super().__init__()
        self._dir = os.path.join(workdir, subdir)
        self._protocol = protocol

    def __call__(self, input, sman, copy=True): # -> InputBase:
        path = reduce(operator.add, (x[2] for x in next(sman.state_machine.get_paths(sman._last_state))))
        self._prefix_len = len(tuple(path))
        joined = path + input
        inp = super().__call__(input, copy=copy)

        input_typ = self.get_parent_class(inp.___iter___)
        filename = self.slugify(f'0x{inp.id:08X}_{input_typ.__name__}')
        self._path = os.path.join(self._dir, f'{filename}.pcap')

        with open(self._path, "wb", buffering=self.IO_BUFFER_SIZE) as file:
            # FIXME remove this ugly hack; had to be here due to circular dependency
            from input import PCAPInput
            pcap = PCAPInput(file, interactions=tuple(joined), protocol=self._protocol)

        return inp

    @staticmethod
    def slugify(value, allow_unicode=False):
        """
        Taken from https://github.com/django/django/blob/master/django/utils/text.py
        Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
        dashes to single dashes. Remove characters that aren't alphanumerics,
        underscores, or hyphens. Convert to lowercase. Also strip leading and
        trailing whitespace, dashes, and underscores.
        """
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize('NFKC', value)
        else:
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')

    # def ___cached_iter___(self):
    #     with open(self._path, "rb", buffering=self.IO_BUFFER_SIZE) as file:
    #         # FIXME remove this ugly hack; had to be here due to circular dependency
    #         from input import PCAPInput
    #         pcap = PCAPInput(file, protocol=self._protocol)
    #         yield from pcap._interactions[self._prefix_len:]

class SlicingDecorator(DecoratorBase):
    def __init__(self, idx):
        self._idx = idx
        if isinstance(self._idx, slice):
            self._start = self._idx.start or 0
            self._stop = self._idx.stop
            self._step = self._idx.step or 1
        else:
            self._start = self._idx
            self._stop = self._idx + 1
            self._step = 1

    def __call__(self, input, copy=True): # -> InputBase:
        if self.get_parent_class(input.___iter___) is self.__class__ and \
                input.___decorator___._step == self._step:
            self._handle_copy(input, copy)
            self._input.___decorator___._start += self._start
            if self._stop is not None:
                stop = self._input.___decorator___._start + (self._stop - self._start)
                assert self._input.___decorator___._stop is None or \
                    self._input.___decorator___._stop >= stop
                self._input.___decorator___._stop = stop
            return self._input
        elif self._start == 0 and self._stop == None:
            return deepcopy(input) if copy else input
        else:
            return super().__call__(input)

    def ___iter___(self, orig):
        yield from islice(orig(), self._start, self._stop, self._step)

    def ___repr___(self, orig):
        if self._stop == self._start + 1 and self._step == 1:
            fmt = f'{self._start}'
        elif self._stop is None:
            if self._step == 1:
                fmt = f'{self._start}:'
            else:
                fmt = f'{self._start}::{self._step}'
        else:
            fmt = f'{self._start}:{self._stop}:{self._step}'
        return f'SlicedInput:0x{self._input.id:08X} (0x{self._input_id:08X}[{fmt}])'

    def ___len___(self, orig):
        raise NotImplemented()

class JoiningDecorator(DecoratorBase):
    def __init__(self, *others):
        self._others = list(others)

    def __call__(self, input, copy=True): # -> InputBase:
        if self.get_parent_class(input.___iter___) is self.__class__:
            self._handle_copy(input, copy)
            self._input.___decorator___._others.extend(self._others)
            return self._input
        else:
            return super().__call__(input)

    def ___iter___(self, orig):
        yield from chain(orig(), *self._others)

    def ___repr___(self, orig):
        id = f'0x{self._input_id:08X}'
        ids = (f'0x{x.id:08X}' for x in self._others)
        return f'JoinedInput:0x{self._input.id:08X} ({" || ".join((id, *ids))})'

    def ___len___(self, orig):
        raise NotImplemented()
