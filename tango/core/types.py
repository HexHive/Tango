from tango.core.tracker import AbstractState, Transition, Path, PathGenerator
from tango.core.input import AbstractInput

from typing import TypeVar, Iterable, Callable, Awaitable, Optional, Any

__all__ = [
    'Transition', 'Path', 'PathGenerator', 'LoadableTarget',
    'ExplorerStateUpdateCallback', 'ExplorerTransitionUpdateCallback',
    'ExplorerStateReloadCallback'
]

S = TypeVar('State', bound=AbstractState)
I = TypeVar('Input', bound=AbstractInput)
LoadableTarget = S | Path
ExplorerStateReloadCallback = Callable[
    [
        LoadableTarget, # target state or path to load
        ## KW_ONLY
        Optional[Exception], # exc= any exceptions encountered on the way
    ],
    Awaitable[Any]
]
ExplorerStateUpdateCallback = Callable[
    [
        S, # current state
        ## KW_ONLY
        LoadableTarget, # breadcrumbs= how to get there again
        I, # input= last input executed
        I, # orig_input= consumed input (as provided by the generator)
        Optional[Exception], # exc= any exceptions encountered on the way
    ],
    Awaitable[Any]
]
ExplorerTransitionUpdateCallback = Callable[
    [
        S, # previous state
        S, # current state
        I, # last input executed
        ## KW_ONLY
        LoadableTarget, # breadcrumbs= how to get there again
        I, # input= last input executed
        I, # orig_input= consumed input (as provided by the generator)
        bool, # state_changed=
        bool, # new_transition= (also implies new state)
        Optional[Exception], # exc= any exceptions encountered on the way
    ],
    Awaitable[Any]
]