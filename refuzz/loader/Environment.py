from abc         import ABC, abstractmethod
from dataclasses import dataclass, field
from typing      import IO

@dataclass
class Environment:
    """
    This class describes a process execution environment.
    """
    path: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = None # field(default_factory=dict)
    cwd: str = None
    stdin:  IO = None
    stdout: IO = None
    stderr: IO = None