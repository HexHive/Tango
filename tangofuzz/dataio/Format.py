from dataclasses import dataclass

@dataclass(frozen=True)
class FormatDescriptor:
    """
    This is just a helper class to be used when passing around
    information describing the format of the serialized data.
    Additional fields could be added by inheritance, in case a
    format can be further specialized, depending on the channel
    being used.
    """
    typ: str