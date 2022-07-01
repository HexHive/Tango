from . import debug, warning, critical

from generator import InputGeneratorBase
from statemanager import StateBase
from input import InputBase, ZoomInput
from interaction import KillInteraction, ReachInteraction
from random import Random
from mutator import ZoomMutator
from typing import Sequence, Tuple
import networkx as nx

class ZoomInputGenerator(InputGeneratorBase):
    def generate(self, state: StateBase, entropy: Random) -> InputBase:
        candidate = state.get_escaper()
        if candidate is None:
            out_edges = list(state.out_edges)
            if out_edges:
                _, dst, data = entropy.choice(out_edges)
                candidate = data['minimized']
            else:
                in_edges = list(state.in_edges)
                if in_edges:
                    _, dst, data = entropy.choice(in_edges)
                    candidate = data['minimized']
                elif self.seeds:
                    candidate = entropy.choice(self.seeds)
                else:
                    candidate = ZoomInput()

        return ZoomMutator(entropy, state)(candidate)

    def generate_follow_path(self, \
            path: Sequence[Tuple[StateBase, StateBase, InputBase]]):
        for src, dst, inp in path:
            dst_x, dst_y, dst_z = map(lambda c: getattr(dst._struct.player_location, c), ('x', 'y', 'z'))
            condition = lambda: dst._sman._tracker.current_state == dst
            yield ReachInteraction(src, (dst_x, dst_y, dst_z),
                sufficient_condition=condition,
                necessary_condition=condition)

    def generate_kill_sequence(self, state, from_location=None):
        yield KillInteraction(state)
        if from_location:
            destination_state = min(
                    filter(lambda x: nx.has_path(
                            state._sman.state_machine._graph,
                            state._sman._tracker._entry_state,
                            x),
                        state._sman.state_machine._graph.nodes), \
                    key=lambda s: ReachInteraction.l2_distance(
                            (from_location[0], from_location[1]),
                            (s._struct.player_location.x, s._struct.player_location.y)))
            path = next(state._sman._loader.live_path_gen(destination_state, state._sman))
            yield from self.generate_follow_path(path)