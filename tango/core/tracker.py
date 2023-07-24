from __future__ import annotations

from . import debug, warning, error

from tango.core.input import AbstractInput, BaseInput, EmptyInput
from tango.core.profiler import (LambdaProfiler, EventProfiler, CountProfiler,
    NumericalProfiler, AbstractProfilerMeta as create_profiler)
from tango.common import AsyncComponent, ComponentType, ComponentOwner
from tango.exceptions import StateNotReproducibleException

from abc          import ABC, ABCMeta, abstractmethod
from typing import TypeVar, Iterable, Optional, Any
from nptyping import NDArray, Shape
from itertools import product as xproduct, tee
from functools import partial
from statistics import mean
from datetime import datetime
import networkx as nx
import numpy as np
import collections

__all__ = [
    'AbstractState', 'BaseState', 'AbstractTracker', 'BaseTracker',
    'AbstractStateGraph', 'BaseStateGraph', 'IUpdateCallback',
    'Transition', 'Path', 'PathGenerator'
]

def shortest_simple_edge_paths(*args, **kwargs):
    for path in nx.shortest_simple_paths(*args, **kwargs):
        a, b = tee(path)
        next(b) # skip one item in b
        yield list(zip(a, b))

nx.shortest_simple_edge_paths = shortest_simple_edge_paths

class AbstractStateMeta(ABCMeta):
    _cache = {}
    _id_counter = 0

    @classmethod
    def __prepare__(metacls, name, bases):
        namespace = super().__prepare__(metacls, name, bases)
        namespace['__slots__'] = ()
        return namespace

    def __new__(metacls, name, bases, namespace):
        # see: https://stackoverflow.com/a/53519136
        if {'__eq__', '__hash__'} & namespace.keys() == {'__eq__'}:
            namespace['__hash__'] = AbstractState.__hash__
        cls = super().__new__(metacls, name, bases, namespace)
        return cls

    def __call__(cls, *args, state_hash: int, update_cache: bool=True,
            fetch_cache: bool=True, **kwargs):
        if fetch_cache and (cached := cls._cache.get(state_hash)):
            return cached
        new = cls.__new__(cls)
        new._hash = state_hash
        if update_cache:
            cls._cache[state_hash] = new
            new._id = cls._id_counter
            cls._id_counter += 1
        else:
            new._id = 'local'
        new.__init__(*args, **kwargs)
        return new

    @classmethod
    def invalidate(cls, state: AbstractState):
        cls._cache.pop(state._hash, None)

class AbstractState(ABC, metaclass=AbstractStateMeta):
    __slots__ = '_id', '_hash', '_tracker'

    def __init__(self, *, tracker: AbstractTracker):
        self._tracker = tracker

    def __hash__(self) -> int:
        return self._hash

    @property
    def tracker(self) -> AbstractTracker:
        return self._tracker

    @property
    @abstractmethod
    def out_edges(self) -> Iterable[Transition]:
        pass

    @property
    @abstractmethod
    def in_edges(self) -> Iterable[Transition]:
        pass

    @property
    @abstractmethod
    def predecessor_transition(self) -> Transition:
        pass

    @abstractmethod
    def __eq__(self, other: AbstractState) -> bool:
        pass

class BaseState(AbstractState):
    __slots__ = '_pred',

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pred = None

    @property
    def out_edges(self) -> Iterable[Transition]:
        return self._tracker.out_edges(self)

    @property
    def in_edges(self) -> Iterable[Transition]:
        return self._tracker.in_edges(self)

    @property
    def predecessor_transition(self) -> Transition:
        return self._pred

    @predecessor_transition.setter
    def predecessor_transition(self, transition: Transition):
        self._pred = transition

class AbstractStateGraphMeta(ABCMeta):
    def __new__(metacls, name, bases, namespace, *, graph_cls=None):
        inherit = False
        if not graph_cls:
            for base in bases:
                if hasattr(base, 'graph_cls'):
                    inherit = True
                    break
            else:
                graph_cls = nx.Graph
        if not inherit:
            namespace['graph_cls'] = graph_cls
            # networkx relies on G.__class__ for creating subviews and other
            # operations; this is a workaround to allow it to instantiate such
            # classes without the arguments typically required by subclasses of
            # AbstractStateGraph
            namespace['__class__'] = graph_cls
            bases = (graph_cls,) + bases
        return super().__new__(metacls, name, bases, namespace)

class AbstractStateGraph(ABC, metaclass=AbstractStateGraphMeta):
    @abstractmethod
    def copy(self, *, fresh=False) -> AbstractStateGraph:
        G = self.__class__.__new__(self.__class__)
        self.graph_cls.__init__(G)
        if fresh:
            G.add_nodes_from(self)
            G.add_edges_from(self.edges)
        else:
            G.graph.update(self.graph)
            G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
            G.add_edges_from(
                (u, v, datadict.copy())
                for u, nbrs in self._adj.items()
                for v, datadict in nbrs.items()
            )
        return G

    @abstractmethod
    def update_state(self, state: AbstractState) -> tuple[AbstractState, bool]:
        pass

    @abstractmethod
    def update_transition(self, source: AbstractState, destination: AbstractState,
            input: AbstractInput):
        pass

    @abstractmethod
    def delete_transition(self, source: AbstractState, destination: AbstractState):
        pass

    @abstractmethod
    def get_paths(self, destination: AbstractState, source: AbstractState=None) \
            -> PathGenerator:
        pass

    @abstractmethod
    def get_inputs(self, edge: Edge) -> Iterable[AbstractInput]:
        pass

    @property
    @abstractmethod
    def adjacency_matrix(self) \
            -> NDArray[Shape["Nodes, Nodes"],
                Iterable[Optional[AbstractInput]] | Optional[AbstractInput]]:
        pass

class BaseStateGraph(AbstractStateGraph, graph_cls=nx.DiGraph):
    """
    A graph-based representation of the explored states. States are derived from
    the hashable and equatable AbstractState base class. Transitions are
    circular buffers of inputs derived from the BaseInput base class.
    """

    def __init__(self, *, entry_state: AbstractState, **kwargs):
        super().__init__(**kwargs)
        self._entry_state = entry_state
        self._queue_maxlen = 10
        self.update_state(self._entry_state)

        NumericalLambdaProfiler = create_profiler('NumericalLambdaProfiler',
            (NumericalProfiler, LambdaProfiler),
            {'value': property(lambda self: float(self._value()))}
        )
        NumericalLambdaProfiler("transition_length")(lambda: \
            mean(                                       # mean value of
                map(                                    # all
                    lambda q: mean(                     # mean values of
                        map(                            # all
                            lambda i: len(i.flatten()), # len(input)
                            q)                          # in each deque of
                    ),
                    map(                                # all
                        lambda x: x[2],                 # transitions
                        edges)                          # in each edge
                )
            ) if len(edges := self.edges(data='transition')) > 0 else None
        )
        NumericalLambdaProfiler("minimized_length")(lambda: \
            mean(                                       # mean value of
                map(                                    # all
                    lambda i: len(i.flatten()),         # len(input) of
                    map(                                # all
                        lambda x: x[2],                 # minimized inputs
                        edges)                          # in each edge
                )
            ) if len(edges := self.edges(data='minimized')) > 0 else None
        )
        LambdaProfiler("snapshots")(lambda: len(self.nodes))

    def copy(self, **kwargs) -> AbstractStateGraph:
        G = super().copy(**kwargs)
        G._entry_state = self._entry_state
        G._queue_maxlen = self._queue_maxlen
        return G

    def adjacency_matrix(self, **kwargs) \
            -> NDArray[Shape["Nodes, Nodes"], Optional[Iterable[BaseInput]]]:
        if kwargs.setdefault('weight', 'transition') == 'transition':
            kwargs.setdefault('nonedge', None)
            to_input_sets = np.vectorize(lambda x: x and frozenset(x))
            return to_input_sets(nx.to_numpy_array(self, **kwargs))
        else:
            return nx.to_numpy_array(self, **kwargs)

    @EventProfiler('update_state')
    def update_state(self, state: AbstractState) -> tuple[AbstractState, bool]:
        new = False
        if state not in self.nodes:
            self.add_node(state, node_obj=state)
            new = True
        else:
            # we retrieve the original state object, in case the tracker
            # returned a new state object with the same hash
            state = self.nodes[state]['node_obj']
        return state, new

    @EventProfiler('update_transition')
    def update_transition(self, source: AbstractState, destination: AbstractState,
            input: BaseInput):
        """
        Adds or updates the transition between two states.

        :param      source:       The source state. Must exist in the state machine.
        :type       source:       AbstractState
        :param      destination:  The destination state. Can be a new state.
        :type       destination:  AbstractState
        :param      input:        The input that causes the transition.
        :type       input:        BaseInput
        """
        new = False

        if source not in self.nodes and source != destination:
            raise KeyError("Source state not present in state machine.")

        if destination not in self.nodes:
            self.update_state(destination)
            new = True

        try:
            data = self.edges[source, destination]
            transition = data['transition']
        except KeyError:
            debug("New transition discovered from %s to %s",
                source, destination)
            transition = collections.deque(maxlen=self._queue_maxlen)
            self.add_edge(source, destination, transition=transition,
                minimized=input)
            new = True

        exists = not new and any(inp == input for inp in transition)
        if not exists:
            transition.append(input)

        return source, destination, new

    def dissolve_state(self, state: AbstractState, stitch: bool=True):
        """
        Deletes a state from the state machine, stitching incoming and outgoing
        transitions to maintain graph connectivity.

        :param      state:  The state to be removed.
        :type       state:  AbstractState
        """
        def flatten(edges):
            for src, dst, data in edges:
                minimized = data['minimized']
                for input in data['transition']:
                    yield src, dst, minimized, input

        if state not in self.nodes:
            raise KeyError("State not present in state machine.")

        # TODO minimize transitions before stitching?

        if stitch:
            t_product = xproduct(
                flatten(self.in_edges(state, data=True)),
                flatten(self.out_edges(state, data=True))
            )
            for (src_in, _, min_in, input_in), (_, dst_out, min_out, input_out) in t_product:
                stitched = (input_in + input_out)
                minimized = (min_in + min_out)

                self.update_transition(
                    source=src_in,
                    destination=dst_out,
                    input=minimized
                )
        self.remove_node(state)

    def delete_transition(self, source: AbstractState, destination: AbstractState):
        if source not in self or destination not in self \
                or not destination in nx.neighbors(self, source):
            raise KeyError("Transition not valid")
        self.remove_edge(source, destination)

    def get_any_path(self, destination: AbstractState, source: AbstractState=None) \
            -> Path:
        """
        Returns an arbitrary path to the destination by reconstructing it from
        each state's cached predecessor transition (i.e. the transition that
        first led to that state from some predecessor). If source is not on the
        reconstructed path, we search for it as usual with get_min_paths.

        :param      destination:  The destination state
        :type       destination:  AbstractState
        :param      source:       The source state
        :type       source:       AbstractState

        :returns:   a list of consecutive edge tuples on the same path.
        :rtype:     list[src, dst, input]
        """
        path = []
        current_state = destination
        while current_state.predecessor_transition is not None \
                and current_state != source:
            pred, inp = current_state.predecessor_transition
            path.append((pred, current_state, inp))
            current_state = pred

        if source not in (None, self._entry_state) and current_state is None:
            # the source state was not on the reconstructed path, so we try to
            # find it through a graph search
            return next(self.get_min_paths(destination, source))

        if not path:
            return [(destination, destination, EmptyInput())]
        else:
            path.reverse()
            return path

    def get_min_paths(self, destination: AbstractState, source: AbstractState=None) \
            -> PathGenerator:
        """
        Generates all minimized paths to destination from source. If source is
        None, the entry point of the state machine is used.

        :param      destination:  The destination state.
        :type       destination:  AbstractState
        :param      source:       The source state.
        :type       source:       AbstractState

        :returns:   Generator object, each item is a list of consecutive edge
                    tuples on the same path.
        :rtype:     generator[list[src, dst, input]]
        """
        return self.get_paths(destination, source, minimized_only=True)

    def get_paths(self, destination: AbstractState, source: AbstractState=None,
            minimized_only=False) \
            -> PathGenerator:
        """
        Generates all paths to destination from source. If source is None, the
        entry point of the state machine is used.

        :param      destination:  The destination state.
        :type       destination:  AbstractState
        :param      source:       The source state.
        :type       source:       AbstractState

        :returns:   Generator object, each item is a list of consecutive edge
                    tuples on the same path.
        :rtype:     generator[list[src, dst, input]]
        """
        source = source or self._entry_state
        if destination == source:
            yield [(source, destination, EmptyInput())]
        else:
            if minimized_only:
                paths = nx.shortest_simple_edge_paths(self, source, destination)
            else:
                paths = nx.all_simple_edge_paths(self, source, destination)
            for path in paths:
                xpaths = xproduct(*(self._get_edge_with_inputs(*edge, minimized_only)
                                        for edge in path))
                for xpath in xpaths:
                    tuples = []
                    for _source, _destination, _input in xpath:
                        tuples.append((_source, _destination, _input))
                    yield tuples

    def get_inputs(self, edge: Edge) -> Iterable[AbstractInput]:
        _,_, data = edge
        return data['transition']

    def _get_edge_with_inputs(self, src, dst, minimized_only):
        data = self.get_edge_data(src, dst)
        yield src, dst, data['minimized']
        if not minimized_only:
            for input in data['transition']:
                yield src, dst, input

class IUpdateCallback(ABC):
    @abstractmethod
    def update_state(self, state: AbstractState, *,
            input: AbstractInput, exc: Exception=None, **kwargs) -> Any:
        pass

    @abstractmethod
    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, *,
            state_changed: bool, exc: Exception=None, **kwargs) -> Any:
        pass

class AbstractTracker(AsyncComponent, IUpdateCallback, ABC,
        component_type=ComponentType.tracker):
    @property
    @abstractmethod
    def entry_state(self) -> AbstractState:
        """
        The state of the target when it is first launched (with no inputs sent)

        :returns:   The state object describing the entry state.
        :rtype:     AbstractState
        """
        pass

    @property
    @abstractmethod
    def current_state(self) -> AbstractState:
        pass

    @property
    @abstractmethod
    def state_graph(self) -> AbstractStateGraph:
        pass

    @abstractmethod
    def peek(self, default_source: AbstractState, expected_destination: AbstractState) -> AbstractState:
        pass

    @abstractmethod
    def reset_state(self, state: AbstractState):
        """
        Informs the state tracker that the loader has reset the target into a
        state.
        """
        pass

    @abstractmethod
    def out_edges(self, state: AbstractState) -> Iterable[Transition]:
        pass

    @abstractmethod
    def in_edges(self, state: AbstractState) -> Iterable[Transition]:
        pass


class BaseTracker(AbstractTracker):
    async def finalize(self, owner: ComponentOwner):
        self._state_graph = BaseStateGraph(entry_state=self.entry_state)

    @property
    def state_graph(self):
        return self._state_graph

    def reset_state(self, state: AbstractState):
        if state not in self._state_graph:
            error("Requested state is not in state graph! Attempting to stitch")
            if state.predecessor_transition:
                self.update_transition(*state.predecessor_transition,
                    state_changed=True)
            else:
                raise StateNotReproducibleException("Orphaned state", state)

    def update_state(self, state: AbstractState, /, *, input: AbstractInput,
            exc: Exception=None, peek_result: Optional[AbstractState]=None) \
            -> Optional[AbstractState]:
        if state:
            if not exc:
                state, is_new = self._state_graph.update_state(state)
            elif state != self.entry_state:
                if isinstance(exc, StateNotReproducibleException):
                    debug("Dissolving irreproducible state=%s", state)
                try:
                    # WARN if stitch==False, this may create disconnected
                    # subgraphs that the strategy is unaware of. Conversely,
                    # stitching may consume too much time and may bring the
                    # fuzzer to a halt (example: states = DOOM map locations)
                    self._state_graph.dissolve_state(state, stitch=True)
                    state.__class__.invalidate(state)
                    CountProfiler("dissolved_states")(1)
                except KeyError as ex:
                    warning("Faulty state was not even valid (ex=%r)", ex)
            return state

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, *,
            state_changed: bool, exc: Exception=None, **kwargs):
        if not exc:
            if state_changed:
                new = self._state_graph.update_transition(source, destination,
                    input)
                # we maintain _a_ path to each new state we encounter so that
                # reproducing the state is more path-aware and does not invoke a
                # graph search every time
                if input is not None and new and \
                        destination.predecessor_transition is None:
                    destination.predecessor_transition = (source, input)
        else:
            try:
                self._state_graph.delete_transition(source, destination)
            except KeyError:
                warning("Faulty transition was not even valid (ex=%r)", ex)

    def out_edges(self, state: AbstractState) -> Iterable[Transition]:
        if state in self.state_graph:
            for edge in self.state_graph.out_edges(state, data=True):
                for inp in self.state_graph.get_inputs(edge):
                    yield (*edge[:2], inp)
        else:
            return ()

    def in_edges(self, state: AbstractState) -> Iterable[Transition]:
        if state in self.state_graph:
            for edge in self.state_graph.in_edges(state, data=True):
                for inp in self.state_graph.get_inputs(edge):
                    yield (*edge[:2], inp)
        else:
            return ()

S = TypeVar('State', bound=AbstractState)
I = TypeVar('Input', bound=AbstractInput)
Edge = tuple[S, S, Any]
Transition = tuple[S, S, I]
Path = list[Transition]
PathGenerator = Iterable[Path]