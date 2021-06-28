from   itertools    import product as xproduct
from   statemanager import (StateBase,
                           TransitionBase)
from   typing       import Generator, Tuple, List
import networkx     as     nx

class StateMachine:
    """
    A graph-based representation of the protocol state machine.
    States are derived from the hashable and equatable StateBase base class.
    Transitions are derived from the addable TransitionBase base class.
    """
    def __init__(self, entry: StateBase, initial: TransitionBase):
        self._graph = nx.MultiDiGraph()
        self._graph.add_node(entry)
        self._entry = entry
        self._initial = initial

    @property
    def entry_state(self):
        return self._entry

    def add_transition(self, source: StateBase, destination: StateBase,
            transition: TransitionBase):
        """
        Adds a transition.

        :param      source:       The source state. Must exist in the state machine.
        :type       source:       StateBase
        :param      destination:  The destination state. Can be a new state.
        :type       destination:  StateBase
        :param      transition:   The transition.
        :type       transition:   TransitionBase
        """
        if source not in self._graph.nodes:
            raise KeyError("Source state not present in state machine.")

        if destination not in self._graph.nodes:
            self._graph.add_node(destination)

        self._graph.add_edge(source, destination, transition=transition)

    def dissolve_state(self, state: StateBase):
        """
        Deletes a state from the state machine, stitching incoming and outgoing
        transitions to maintain graph connectivity.

        :param      state:  The state to be removed.
        :type       state:  StateBase
        """
        if state not in self._graph.nodes:
            raise KeyError("State not present in state machine.")

        # TODO minimize transitions before stitching?

        t_product = xproduct(
            self._graph.in_edges(state, data=True),
            self._graph.out_edges(state, data=True)
        )
        for t_in, t_out in t_product:
            stitched = t_in[2]["transition"] + t_out[2]["transition"]
            self.add_transition(
                source=t_in[0],
                destination=t_out[1],
                transition=stitched
            )
        self._graph.remove_node(state)

    def get_paths(self, destination: StateBase, source: StateBase=None) \
        -> Generator[List[Tuple[StateBase, StateBase, TransitionBase]], None, None]:
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
        if source is None:
            source = self._entry
        if source == destination:
            yield [(source, destination, self._initial),]
        else:
            paths = nx.all_simple_edge_paths(self._graph, source, destination)
            for path in paths:
                tuples = []
                for edge in path:
                    _source = edge[0]
                    _destination = edge[1]
                    _idx = edge[2]
                    _transition = G.get_edge_data(_source, _destination)[_idx]
                    tuples.append((_source, _destination, _transition))
                yield tuples
