from input       import SerializedMetaInput 
from typing      import Iterable
from interaction import InteractionBase, TransmitInteraction
import struct

class RawInput(metaclass=SerializedMetaInput, typ='raw'):
    def loadi(self) -> Iterable[InteractionBase]:
        data = self._file.read()
        for c in struct.unpack(f'{len(data)}c', data):
            interaction = TransmitInteraction(data=c)
            yield interaction

    def dumpi(self, itr: Iterable[InteractionBase], /):
        for interaction in itr:
            if isinstance(interaction, TransmitInteraction):
                self._file.write(interaction._data)

    def __repr__(self):
        return f"RawInput({self._file})"
