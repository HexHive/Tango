from input       import SerializedInputMeta
from typing      import Iterable
from interaction import AbstractInteraction, TransmitInteraction
import struct

class RawInput(metaclass=SerializedInputMeta, typ='raw'):
    CHUNKSIZE = 4

    def loadi(self) -> Iterable[AbstractInteraction]:
        data = self._file.read()
        unpack_len = len(data) - (len(data) % self.CHUNKSIZE)
        for s, in struct.iter_unpack(f'{self.CHUNKSIZE}s', data[:unpack_len]):
            interaction = TransmitInteraction(data=s)
            yield interaction
        if unpack_len < len(data):
            interaction = TransmitInteraction(data=data[unpack_len:])
            yield interaction

    def dumpi(self, itr: Iterable[AbstractInteraction], /):
        for interaction in itr:
            if isinstance(interaction, TransmitInteraction):
                self._file.write(interaction._data)

    def __repr__(self):
        return f"RawInput({self._file})"
