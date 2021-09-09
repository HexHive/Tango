from . import debug

from   itertools    import product as xproduct
from   statemanager import StateBase
from   input        import InputBase, PreparedInput
from   typing       import Generator, Tuple, List
import networkx     as     nx
import collections

from profiler import ProfileLambda, ProfileEvent
from statistics import mean

class StateMachine:
    """
    A graph-based representation of the protocol state machine.
    States are derived from the hashable and equatable StateBase base class.
    Transitions are circular buffers of inputs derived from the InputBase base
    class.
    """
    def __init__(self, entry_state: StateBase):
        self._graph = nx.DiGraph()
        self._graph.add_node(entry_state)
        self._entry_state = entry_state
        self._queue_maxlen = 10

        ProfileLambda("coverage")(lambda: len(self._graph.nodes))

        def unsafe_mean(data):
            try:
                return mean(data)
            except Exception as ex:
                print(ex)
                return 0
        ProfileLambda("transition_length")(lambda: unsafe_mean(len(y)
                                        for _, _, data in self._graph.edges(data=True)
                                        for y in data['transition']))

    @property
    def entry_state(self):
        return self._entry_state

    @ProfileEvent("update_transition")
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
        new = False
        if destination not in self._graph.nodes:
            self._graph.add_node(destination)
            new = True

        if source not in self._graph.nodes:
            raise KeyError("Source state not present in state machine.")

        try:
            find = filter(lambda edge: edge[1] == destination,
                                self._graph.out_edges(source, data=True))
            _, _, data = next(find)
            transition = data['transition']
        except StopIteration:
            debug(f'New transition discovered from {source} to {destination}')
            transition = collections.deque(maxlen=self._queue_maxlen)
            self._graph.add_edge(source, destination, transition=transition)
            new = True

        exists = not new and any(inp == input for inp in transition)
        if not exists:
            transition.append(input)

    def dissolve_state(self, state: StateBase):
        """
        Deletes a state from the state machine, stitching incoming and outgoing
        transitions to maintain graph connectivity.

        :param      state:  The state to be removed.
        :type       state:  StateBase
        """
        def flatten(edges):
            for src, dst, data in edges:
                for input in data['transition']:
                    yield src, dst, input

        if state not in self._graph.nodes:
            raise KeyError("State not present in state machine.")

        # TODO minimize transitions before stitching?

        t_product = xproduct(
            flatten(self._graph.in_edges(state, data=True)),
            flatten(self._graph.out_edges(state, data=True))
        )
        for (src_in, _, input_in), (_, dst_out, input_out) in t_product:
            stitched = input_in + input_out
            self.update_transition(
                source=src_in,
                destination=dst_out,
                transition=stitched
            )
        self._graph.remove_node(state)

    def get_paths(self, destination: StateBase, source: StateBase=None) \
        -> Generator[List[Tuple[StateBase, StateBase, InputBase]], None, None]:
        """
        Generates all paths to destination from source. If source is None, the entry
        point of the state machine is used.

        :param      destination:  The destination state.
        :type       destination:  StateBase
        :param      source:       The source state.
        :type       source:       StateBase

        :returns:   Generator object, each item is a list of consecutive edge tuples
                    on the same path.
        :rtype:     generator
        """
        def get_edge_with_inputs(src, dst):
            data = self._graph.get_edge_data(src, dst)
            for input in data['transition']:
                yield src, dst, input

        source = source or self._entry_state
        if destination == source:
            yield [(source, destination, PreparedInput())]
        else:
            paths = nx.all_simple_edge_paths(self._graph, source, destination)
            for path in paths:
                xpaths = xproduct(*(get_edge_with_inputs(*edge) for edge in path))
                for xpath in xpaths:
                    tuples = []
                    for _source, _destination, _input in xpath:
                        tuples.append((_source, _destination, _input))
                    yield tuples