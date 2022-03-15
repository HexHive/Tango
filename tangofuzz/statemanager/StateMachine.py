from . import debug

from   itertools    import product as xproduct
from   functools    import partial
from   statemanager import StateBase
from   input        import InputBase, PreparedInput, MemoryCachingDecorator
from   typing       import Generator, Tuple, List
import networkx     as     nx
import collections

from profiler import ProfileValue, ProfileValueMean, ProfileEvent
from statistics import mean
from datetime import datetime
now = datetime.now

class StateMachine:
    """
    A graph-based representation of the protocol state machine.
    States are derived from the hashable and equatable StateBase base class.
    Transitions are circular buffers of inputs derived from the InputBase base
    class.
    """
    def __init__(self, entry_state: StateBase):
        self._graph = nx.DiGraph()
        self.update_state(entry_state)
        self._entry_state = entry_state
        self._queue_maxlen = 10

    @property
    def entry_state(self):
        return self._entry_state

    @ProfileEvent('update_state')
    def update_state(self, state: StateBase):
        time = now()
        new = False
        if state not in self._graph.nodes:
            self._graph.add_node(state, added=time, node_obj=state)
            state.out_edges = lambda **kwargs: partial(self._graph.out_edges, state)(**kwargs) if state in self._graph.nodes else ()
            state.in_edges = lambda **kwargs: partial(self._graph.in_edges, state)(**kwargs) if state in self._graph.nodes else ()
            new = True
        else:
            # we retrieve the original state object
            state = self._graph.nodes[state]['node_obj']
        self._graph.add_node(state, last_visit=time)

        ProfileValue("coverage")(len(self._graph.nodes))
        return new, state

    @ProfileEvent('update_transition')
    def update_transition(self, source: StateBase, destination: StateBase,
            input: InputBase):
        """
        Adds or updates the transition between two states.

        :param      source:       The source state. Must exist in the state machine.
        :type       source:       StateBase
        :param      destination:  The destination state. Can be a new state.
        :type       destination:  StateBase
        :param      input:        The input that causes the transition.
        :type       input:        InputBase
        """
        time = now()
        new = False
        if destination not in self._graph.nodes:
            self.update_state(destination)
            new = True

        if source not in self._graph.nodes:
            raise KeyError("Source state not present in state machine.")

        try:
            data = self._graph.edges[source, destination]
            data['last_visit'] = time
            transition = data['transition']
        except KeyError:
            debug(f'New transition discovered from {source} to {destination}')
            transition = collections.deque(maxlen=self._queue_maxlen)
            self._graph.add_edge(source, destination, transition=transition,
                added=time, last_visit=time, minimized=input)
            ProfileValueMean("minimized_length", samples=0)(len(input))
            new = True

        exists = not new and any(inp == input for inp in transition)
        if not exists:
            transition.append(input)
            ProfileValueMean("transition_length", samples=0)(len(input))


    def dissolve_state(self, state: StateBase):
        """
        Deletes a state from the state machine, stitching incoming and outgoing
        transitions to maintain graph connectivity.

        :param      state:  The state to be removed.
        :type       state:  StateBase
        """
        def flatten(edges):
            for src, dst, data in edges:
                minimized = data['minimized']
                for input in data['transition']:
                    yield src, dst, minimized, input

        if state not in self._graph.nodes:
            raise KeyError("State not present in state machine.")

        # TODO minimize transitions before stitching?

        t_product = xproduct(
            flatten(self._graph.in_edges(state, data=True)),
            flatten(self._graph.out_edges(state, data=True))
        )
        for (src_in, _, min_in, input_in), (_, dst_out, min_out, input_out) in t_product:
            # FIXME inputs are cached to add the __len__() functionality so that
            # the WebUI graph can be properly displayed
            # FIXME this should be remedied by introducing length propagation
            # of inputs, where all inputs support the __len__() function
            stitched = MemoryCachingDecorator()(input_in + input_out, copy=False)
            minimized = MemoryCachingDecorator()(min_in + min_out, copy=False)

            self.update_transition(
                source=src_in,
                destination=dst_out,
                input=minimized
            )
        self._graph.remove_node(state)

    def get_any_path(self, destination: StateBase, source: StateBase=None) \
        -> List[Tuple[StateBase, StateBase, InputBase]]:
        """
        Returns an arbitrary path to the destination by reconstructing it from
        each state's cached predecessor transition (i.e. the transition that
        first led to that state from some predecessor). If source is not on the
        reconstructed path, we search for it as usual with get_min_paths.

        :param      destination:  The destination state
        :type       destination:  StateBase
        :param      source:       The source state
        :type       source:       StateBase

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
            return [(destination, destination, PreparedInput())]
        else:
            path.reverse()
            return path

    def get_min_paths(self, destination: StateBase, source: StateBase=None) \
        -> Generator[List[Tuple[StateBase, StateBase, InputBase]], None, None]:
        """
        Generates all minimized paths to destination from source. If source is
        None, the entry point of the state machine is used.

        :param      destination:  The destination state.
        :type       destination:  StateBase
        :param      source:       The source state.
        :type       source:       StateBase

        :returns:   Generator object, each item is a list of consecutive edge
                    tuples on the same path.
        :rtype:     generator[list[src, dst, input]]
        """
        return self.get_paths(destination, source, minimized_only=True)

    def get_paths(self, destination: StateBase, source: StateBase=None,
            minimized_only=False) \
        -> Generator[List[Tuple[StateBase, StateBase, InputBase]], None, None]:
        """
        Generates all paths to destination from source. If source is None, the
        entry point of the state machine is used.

        :param      destination:  The destination state.
        :type       destination:  StateBase
        :param      source:       The source state.
        :type       source:       StateBase

        :returns:   Generator object, each item is a list of consecutive edge
                    tuples on the same path.
        :rtype:     generator[list[src, dst, input]]
        """
        source = source or self._entry_state
        if destination == source:
            yield [(source, destination, PreparedInput())]
        else:
            paths = nx.all_simple_edge_paths(self._graph, source, destination)
            for path in paths:
                xpaths = xproduct(*(self.get_edge_with_inputs(*edge, minimized_only)
                                        for edge in path))
                for xpath in xpaths:
                    tuples = []
                    for _source, _destination, _input in xpath:
                        tuples.append((_source, _destination, _input))
                    yield tuples

    def get_edge_with_inputs(self, src, dst, minimized_only):
        data = self._graph.get_edge_data(src, dst)
        yield src, dst, data['minimized']
        if not minimized_only:
            for input in data['transition']:
                yield src, dst, input