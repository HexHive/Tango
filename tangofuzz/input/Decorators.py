from input import DecoratorBase, MemoryCachingDecorator
from functools import reduce
import unicodedata
import re
import os
import operator

class FileCachingDecorator(MemoryCachingDecorator):
    IO_BUFFER_SIZE = 0

    def __init__(self, workdir, subdir, protocol):
        super().__init__()
        self._dir = os.path.join(workdir, subdir)
        self._protocol = protocol

    def __call__(self, input, sman, copy=True): # -> InputBase:
        path = reduce(operator.add, (x[2] for x in next(sman.state_machine.get_min_paths(sman._last_state))))
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
