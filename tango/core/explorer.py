from __future__ import annotations
from . import debug, info, warning, critical

from tango.exceptions import (StabilityException, StateNotReproducibleException,
    StatePrecisionException)
from tango.core.tracker import AbstractState, AbstractStateTracker
from tango.core.input import (AbstractInput, BaseInput, PreparedInput,
    BaseDecorator)
from tango.core.loader import AbstractStateLoader
from tango.core.profiler import (ValueProfiler, FrequencyProfiler,
    CountProfiler, LambdaProfiler, EventProfiler, NumericalProfiler,
    AbstractProfilerMeta as create_profiler)
from tango.core.types import (Path, PathGenerator, LoadableTarget,
    ExplorerStateUpdateCallback, ExplorerTransitionUpdateCallback,
    ExplorerStateReloadCallback)
from tango.common import Configurable, ComponentType

from abc import ABC, abstractmethod
from typing import Callable
from itertools import product as xproduct, tee, chain
from functools import partial
from statistics import mean
from datetime import datetime
import networkx as nx
import asyncio
import collections

__all__ = ['StateGraph', 'AbstractExplorer', 'BaseExplorer']

def shortest_simple_edge_paths(*args, **kwargs):
    for path in nx.shortest_simple_paths(*args, **kwargs):
        a, b = tee(path)
        next(b) # skip one item in b
        yield list(zip(a, b))

nx.shortest_simple_edge_paths = shortest_simple_edge_paths

class StateGraph:
    """
    A graph-based representation of the explored states. States are derived from
    the hashable and equatable AbstractState base class. Transitions are
    circular buffers of inputs derived from the BaseInput base class.
    """

    def __init__(self, entry_state: AbstractState):
        self._graph = nx.DiGraph()
        self.update_state(entry_state)
        self._entry_state = entry_state
        self._queue_maxlen = 10

        NumericalLambdaProfiler = create_profiler('NumericalLambdaProfiler',
            (NumericalProfiler, LambdaProfiler),
            {'numerical_value': property(lambda self: float(self._value()))}
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
            ) if len(edges := self._graph.edges(data='transition')) > 0 else None
        )
        NumericalLambdaProfiler("minimized_length")(lambda: \
            mean(                                       # mean value of
                map(                                    # all
                    lambda i: len(i.flatten()),         # len(input) of
                    map(                                # all
                        lambda x: x[2],                 # minimized inputs
                        edges)                          # in each edge
                )
            ) if len(edges := self._graph.edges(data='minimized')) > 0 else None
        )
        LambdaProfiler("coverage")(lambda: len(self._graph.nodes))

    @property
    def entry_state(self):
        return self._entry_state

    @EventProfiler('update_state')
    def update_state(self, state: AbstractState) -> tuple[AbstractState, bool]:
        time = datetime.now()
        new = False
        if state not in self._graph.nodes:
            self._graph.add_node(state, added=time, node_obj=state)
            state.out_edges = lambda **kwargs: partial(self._graph.out_edges, state)(**kwargs) if state in self._graph.nodes else ()
            state.in_edges = lambda **kwargs: partial(self._graph.in_edges, state)(**kwargs) if state in self._graph.nodes else ()
            new = True
        else:
            # we retrieve the original state object, in case the tracker
            # returned a new state object with the same hash
            state = self._graph.nodes[state]['node_obj']
        self._graph.add_node(state, last_visit=time)

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
        time = datetime.now()
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
            new = True

        exists = not new and any(inp == input for inp in transition)
        if not exists:
            transition.append(input)

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

        if state not in self._graph.nodes:
            raise KeyError("State not present in state machine.")

        # TODO minimize transitions before stitching?

        if stitch:
            t_product = xproduct(
                flatten(self._graph.in_edges(state, data=True)),
                flatten(self._graph.out_edges(state, data=True))
            )
            for (src_in, _, min_in, input_in), (_, dst_out, min_out, input_out) in t_product:
                stitched = (input_in + input_out)
                minimized = (min_in + min_out)

                self.update_transition(
                    source=src_in,
                    destination=dst_out,
                    input=minimized
                )
        self._graph.remove_node(state)

    def delete_transition(self, source: AbstractState, destination: AbstractState):
        if source not in self._graph or destination not in self._graph \
                or not destination in nx.neighbors(self._graph, source):
            raise KeyError("Transition not valid")
        self._graph.remove_edge(source, destination)

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
            return [(destination, destination, PreparedInput())]
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
            yield [(source, destination, PreparedInput())]
        else:
            if minimized_only:
                paths = nx.shortest_simple_edge_paths(self._graph, source, destination)
            else:
                paths = nx.all_simple_edge_paths(self._graph, source, destination)
            for path in paths:
                xpaths = xproduct(*(self._get_edge_with_inputs(*edge, minimized_only)
                                        for edge in path))
                for xpath in xpaths:
                    tuples = []
                    for _source, _destination, _input in xpath:
                        tuples.append((_source, _destination, _input))
                    yield tuples

    def _get_edge_with_inputs(self, src, dst, minimized_only):
        data = self._graph.get_edge_data(src, dst)
        yield src, dst, data['minimized']
        if not minimized_only:
            for input in data['transition']:
                yield src, dst, input

class AbstractExplorer(Configurable, ABC,
        component_type=ComponentType.explorer):
    def __init__(self):
        self._state_reload_cb = self._nop_cb
        self._state_update_cb = self._nop_cb
        self._transition_update_cb = self._nop_cb

    @staticmethod
    async def _nop_cb(*args, **kwargs):
        pass

    def register_state_reload_callback(self,
            callback: ExplorerStateReloadCallback):
        self._state_reload_cb = callback

    def register_state_update_callback(self,
            callback: ExplorerStateUpdateCallback):
        self._state_update_cb = callback

    def register_transition_update_callback(self,
            callback: ExplorerTransitionUpdateCallback):
        self._transition_update_cb = callback

    @abstractmethod
    async def reload_state(self, state_or_path=None) -> AbstractState:
        pass

    @abstractmethod
    async def follow(self, input: AbstractInput):
        pass

class BaseExplorer(AbstractExplorer,
        capture_components={ComponentType.loader, ComponentType.tracker},
        capture_paths=['explorer.reload_attempts']):
    def __init__(self,
            loader: AbstractStateLoader, tracker: AbstractStateTracker,
            *, reload_attempts: str='50', **kwargs):
        super().__init__(**kwargs)
        self._loader = loader
        self._tracker = tracker
        self._reload_attempts = int(reload_attempts)

        self._last_state = self._current_state = self._tracker.entry_state
        self._sg = StateGraph(self._last_state)
        self._current_path = []
        self._last_path = []

        LambdaProfiler('global_cov')(lambda: sum(map(lambda x: x._set_count, filter(lambda x: x != self._tracker.entry_state, self._sg._graph.nodes))))

    @property
    def state_graph(self) -> StateGraph:
        return self._sg

    @property
    def tracker(self) -> AbstractStateTracker:
        return self._tracker

    @property
    def loader(self) -> AbstractStateLoader:
        return self._loader

    def _reset_current_path(self):
        self._current_path.clear()

    async def attempt_load_state(self, loadable: LoadableTarget):
        if isinstance(state := loadable, AbstractState):
            # loop over possible paths until retry threshold
            paths = chain(self._sg.get_min_paths(state),
                          self._sg.get_paths(state))
            for i, path in zip(range(self._reload_attempts), paths):
                try:
                    return await self._arbitrate_load_state(path)
                except StabilityException as ex:
                    warning("Failed to follow unstable path"
                        f" (reason = {ex.args[0]})!"
                        f" Retrying... ({i+1}/{self._reload_attempts})")
                    CountProfiler('unstable')(1)
            raise StateNotReproducibleException("destination state not reproducible",
                state)
        else:
            try:
                return await self._arbitrate_load_state(loadable)
            except StabilityException as ex:
                warning("Failed to follow unstable preselected path"
                    f" (reason = {ex.args[0]})!")
                CountProfiler('unstable')(1)
                raise StateNotReproducibleException("destination state not reproducible",
                    None) from ex

    async def _arbitrate_load_state(self, path: Path):
        # FIXME interactions between the loader, tracker, explorer, and possibly
        # higher components (generator, mutator, fuzzer, etc...) could be more
        # conventiently specified by an arbitration protocol which all concerned
        # components implement.

        # clear the current path to stay consistent with the target
        self._reset_current_path()
        # deliberately not `await`ed; we exchange messages with the generator
        load = self._loader.load_state(path)
        try:
            nxt = None
            while True:
                previous, current, inp = await load.asend(nxt)
                if current is None:
                    # special case where the loader is requesting support from
                    # the fuzzer and generator overlords;
                    # we inform them that a reload will happen and forward
                    # their response
                    nxt = await self._state_reload_cb(path)
                else:
                    if inp:
                        self._current_path.append((previous, current, inp))
                    nxt = self._tracker.peek(previous, current)
        except StopAsyncIteration:
            return current

    @FrequencyProfiler('resets')
    async def reload_state(self, state_or_path: LoadableTarget=None, *,
            dryrun=False) -> AbstractState:
        if not dryrun:
            ValueProfiler('status')('reset_state')
            self._reset_current_path()
            # must clear last state's input whenever a new state is
            # loaded
            #
            # WARN if using SnapshotStateLoader, the snapshot must be
            # taken when the state is first reached, not when a
            # transition is about to happen. Otherwise, last_input may
            # be None, but the snapshot may have residual effect from
            # previous inputs before the state change.
            if self._last_state is not None:
                self._last_state.last_input = None

        loadable = state_or_path or self._tracker.entry_state
        try:
            current_state = await self.attempt_load_state(loadable)
            if not dryrun:
                self._tracker.reset_state(current_state)
                if (current_state := self._tracker.current_state) not in self._sg._graph:
                    critical("Current state is not in state graph! Launching interactive debugger...")
                    import ipdb; ipdb.set_trace()
                self._last_state = current_state
        except CoroInterrupt:
            # if an interrupt is received while loading a state (e.g. death),
            # self._last_state is not set to the current state because of the
            # exception. Instead, at the next input step, a transition is
            # "discovered" between the last state and the new state, which is
            # wrong
            # FIXME this should be handled by whomever interrupted the coroutine
            # in the first place
            if not dryrun:
                if (current_state := self._tracker.current_state) not in self._sg._graph:
                    critical("Current state is not in state graph! Launching interactive debugger...")
                    import ipdb; ipdb.set_trace()
                self._last_state = current_state
            raise
        except StateNotReproducibleException as ex:
            if not dryrun:
                faulty_state = ex._faulty_state
                if faulty_state and faulty_state != self._tracker.entry_state:
                    try:
                        debug(f"Dissolving irreproducible {faulty_state = }")
                        # WARN if stitch==False, this may create disconnected
                        # subgraphs that the strategy is unaware of. Conversely,
                        # stitching may consume too much time and may bring the
                        # fuzzer to a halt (example: states = DOOM map locations)
                        self._sg.dissolve_state(faulty_state, stitch=True)
                        CountProfiler("dissolved_states")(1)
                    except KeyError as ex:
                        warning(f"Faulty state was not even valid")
            await self._state_reload_cb(loadable, exc=ex)
            raise
        return current_state

    def get_context_input(self, input: BaseInput, **kwargs) -> BaseExplorerContext:
        return BaseExplorerContext(self, **kwargs)(input)

    async def follow(self, input: BaseInput, **kwargs):
        """
        Executes the input and updates the state queues according to the
        scheduler. May need to receive information about the current state to
        update it.
        """
        context_input = self.get_context_input(input, **kwargs)
        await self._loader.execute_input(context_input)

    async def update(self, input_gen: Callable[..., BaseInput],
            minimize: bool=True, validate: bool=True) \
            -> tuple(bool, bool, BaseInput):
        """
        Updates the state machine in case of a state change.

        :param      input_gen:  A function that returns the input that may have
                      resulted in the state change
        :type       input_gen:  Callable[..., BaseInput]

        :returns:   (state_changed?, unseen?, accumulated input)
        :rtype:     tuple(bool, bool, BaseInput)
        """

        # WARN the AbstractState object returned by the state tracker may have the
        # same hash(), but may be a different object. This means, any
        # modifications made to the state (or new attributes stored) may not
        # persist.
        current_state = self._tracker.current_state

        # the tracker may return None as current_state, in case it has not yet
        # finished the training phase (preprocessing seeds)
        if current_state is None:
            return False, False, None

        # we obtain a persistent reference to the current_state
        self._current_state, is_new_state = self._sg.update_state(current_state)

        debug(f"Reached {'new ' if is_new_state else ''}{self._current_state = }")

        if self._current_state == self._last_state:
            return False, False, None

        is_new_edge = self._current_state not in self._sg._graph.successors(self._last_state)
        debug(f"Possible transition from {self._last_state} to {self._current_state}")

        unseen = (is_new_state or is_new_edge)
        last_input = input_gen()
        if unseen:
            if minimize:
                # During minimization, the input is sliced/joined/cached, which
                # are facilities provided by BaseInput.
                # An input type that does not support these capabilities must
                # inherit from AbstractInput and disable minimization (or use a
                # different Explorer).
                assert isinstance(last_input, BaseInput), \
                    "Minimizing requires inputs to be BaseInput!"
                # call the transition pruning routine to shorten the last input
                debug("Attempting to minimize transition")
                last_input = last_input.flatten(inplace=True)
                # we clone the current path because minimization may corrupt it
                self._last_path = self._current_path.copy()
                try:
                    last_input = await self._minimize_transition(
                        self._last_path, self._current_state, last_input)
                except Exception as ex:
                    # Minimization failed, again probably due to an
                    # indeterministic target
                    warning(f"Minimization failed {ex=}")
                    raise
                last_input = last_input.flatten(inplace=True)
            elif validate:
                try:
                    debug("Attempting to reproduce transition")
                    # we make a copy of _current_path because the loader may
                    # modify it, but at this stage, we may want to restore state
                    # using that path
                    self._last_path = self._current_path.copy()
                    src = await self.reload_state(self._last_state, dryrun=True)
                    assert self._last_state == src
                    await self._loader.execute_input(input)
                    actual_state = self._tracker.peek(
                        src, self._current_state)
                    if self._current_state != actual_state:
                        raise StatePrecisionException(
                        f"The path to {self._current_state} is imprecise")
                except (StabilityException,
                        StateNotReproducibleException,
                        StatePrecisionException):
                    # * StatePrecisionException:
                    #   This occurs when the predecessor state was reached
                    #   through a different path than that used by
                    #   reload_state(). The incremental input thus applied to
                    #   the predecessor state may have built on top of internal
                    #   program state that was not reproduced by reload_state()
                    #   to arrive at the current successor state.
                    #
                    # * StateNotReproducibleException, StatePrecisionException:
                    #   This occurs when the reload_state() encountered an error
                    #   trying to reproduce a state, most likely due to an
                    #   indeterministic target
                    debug(f"Encountered imprecise state ({is_new_state = })")
                    if is_new_state:
                        debug(f"Dissolving {self._current_state = }")
                        self._sg.dissolve_state(self._current_state)
                    raise
                except Exception as ex:
                    warning(f'{ex}')
                    if is_new_state:
                        debug(f"Dissolving {self._current_state = }")
                        self._sg.dissolve_state(self._current_state)
                    raise
            if minimize or validate:
                debug(f"{self._current_state} is reproducible!")

        # FIXME doesn't work as intended, but could save from recalculating cov map diffs if it works
        # if self._current_state != self._tracker.peek(self._last_state):
        #     raise StabilityException("Failed to obtain consistent behavior")
        # self._tracker.update(self._last_state, last_input, peek_result=self._current_state)
        if self._current_state != self._tracker.update(self._last_state, last_input):
            raise StabilityException("Failed to obtain consistent behavior")

        self._sg.update_transition(self._last_state, self._current_state,
            last_input)
        self._last_state = self._current_state
        info(f'Transitioned to {self._current_state}')

        return True, unseen, last_input

    async def _minimize_transition(self,
            state_or_path: LoadableTarget[AbstractState, BaseInput],
            dst: AbstractState, input: BaseInput):
        reduced = False

        # Phase 1: perform exponential back-off to find effective tail of input
        ValueProfiler('status')('minimize_exp_backoff')
        end = len(input)
        begin = end - 1
        # we do not perform exponential backoff past the midpoint, because that
        # is done in phase 2 anyway
        while begin > end // 2:
            success = True
            src = await self.reload_state(state_or_path, dryrun=True)
            exp_input = input[begin:]
            try:
                await self._loader.execute_input(exp_input)
            except Exception:
                success = False

            success &= dst == self._tracker.peek(src, dst)
            if success:
                reduced = True
                break
            begin = 0 if (diff := end - (end - begin) * 2) < 0 else diff

        # Phase 2: prune out dead interactions
        if reduced:
            lin_input = input[begin:]
        else:
            lin_input = input
        end = len(input) - begin
        step = (end - 1) // 2
        i = 0
        while step > 0:
            cur = 0
            while cur + step < end:
                ValueProfiler('status')(f'minimize_bin_search ({100*i/(2 * (len(input) - begin)):.1f}%)')
                success = True
                src = await self.reload_state(state_or_path, dryrun=True)
                tmp_lin_input = lin_input[:cur] + lin_input[cur + step:]
                try:
                    await self._loader.execute_input(tmp_lin_input)
                except Exception:
                    success = False

                success &= dst == self._tracker.peek(src, dst)
                if success:
                    reduced = True
                    lin_input = tmp_lin_input.flatten(inplace=True)
                    end -= step
                else:
                    cur += step
                i += 1
            step //= 2

        # Phase 3: make sure the reduced transition is correct
        src = await self.reload_state(state_or_path, dryrun=True)
        await self._loader.execute_input(lin_input)
        success = (dst == self._tracker.peek(src, dst))
        if not success and reduced:
            # finally, we fall back to validating the original input
            lin_input = input
            src = await self.reload_state(state_or_path, dryrun=True)
            await self._loader.execute_input(lin_input)
            success = (dst == self._tracker.peek(src, dst))
        if not success:
            raise StatePrecisionException("destination state did not match current state")

        if reduced:
            return lin_input
        else:
            return input

class BaseExplorerContext(BaseDecorator):
    """
    This context object provides a wrapper around the input to be sent to the
    target. By wrapping the iterator method, the explorer is updated after
    every interaction in the input. If a state change happens within an input
    sequence, the input is split, where the first part is used as the transition
    and the second part continues to build up state for any following
    transition.
    """
    def __init__(self, sman: BaseExplorer, **kwargs):
        self._sman = sman
        self._start = self._stop = None
        self._update_kwargs = kwargs

    def input_gen(self):
        # we delay the call to the slicing decorator until needed
        head = self._sman._last_state.last_input
        tail = self._input[self._start:self._stop]
        if head is None:
            if self._start >= self._stop:
                return None
            else:
                return tail
        else:
            return head + tail

    async def update_state(self, *args, **kwargs):
        await self._sman._state_update_cb(*args, **kwargs)

    async def update_transition(self, *args, **kwargs):
        await self._sman._transition_update_cb(*args, **kwargs)

    @property
    def orig_input(self):
        # we pop the BaseExplorerContext decorator itself to obtain a reference
        # to the original input, typically the output of the input generator
        return self._input.pop_decorator()[0]

    async def ___aiter___(self, input, orig):
        self._start = self._stop = 0
        idx = -1
        for idx, interaction in enumerate(input):
            ValueProfiler('status')('fuzz')
            self._stop = idx + 1
            yield interaction
            # the generator execution is suspended until next() is called so
            # the BaseExplorer update is only called after the interaction
            # is executed by the loader

            try:
                last_state = self._sman._last_state
                updated, new, last_input = await self._sman.update(self.input_gen,
                    **self._update_kwargs)
                if updated:
                    self._start = idx + 1

                    # if an update has happened, we've transitioned out of the
                    # last_state and as such, it's no longer necessary to keep
                    # track of the last input
                    last_state.last_input = None
            except Exception as ex:
                self._stop = idx + 1
                # we also clear the last input on an exception, because the next
                # course of action will involve a reload_state
                last_state.last_input = None
                await self.update_state(self._sman._current_state,
                    breadcrumbs=self._sman._last_path,
                    input=self.input_gen(), orig_input=self.orig_input, exc=ex)
                raise
            else:
                if not updated:
                    # we force a state tracker update (e.g. update implicit state)
                    # FIXME this may not work well with all state trackers
                    ns = self._sman.tracker.update(last_state, last_input)
                    assert ns == last_state, "State tracker is inconsistent!"
                else:
                    self._sman._current_path.append(
                        (last_state, self._sman._current_state, last_input))

                await self.update_state(self._sman._current_state, input=last_input,
                    orig_input=self.orig_input, breadcrumbs=self._sman._last_path)
                await self.update_transition(
                    last_state, self._sman._current_state, last_input,
                    orig_input=self.orig_input, breadcrumbs=self._sman._last_path,
                    state_changed=updated, new_transition=new)

            # FIXME The explorer interrupts the target to verify individual
            # transitions. To verify a transition, the last state is loaded, and
            # the last input which observed a new state is replayed. However,
            # due to imprecision of states, the chosen path to the last state
            # may result in a different substate, and the replayed input thus no
            # longer reaches the new state. When such an imprecision is met, the
            # target is kept running, but the path has been "corrupted".
            #
            # It may be better to let the input update the explorer freely
            # all throughout the sequence, and perform the validation step at
            # the end to make sure no interesting states are lost.

        if idx >= 0:
            # commit the rest of the input
            self._sman._last_state.last_input = self.input_gen()

    def ___iter___(self, input, orig):
        return orig()
