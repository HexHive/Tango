from . import debug, info, warning, critical

from tango.core.tracker import AbstractState, IUpdateCallback
from tango.core.input     import AbstractInput
from tango.core.types import LoadableTarget
from tango.core.explorer import BaseExplorer
from tango.core.generator import AbstractInputGenerator, BaseInputGenerator
from tango.core.profiler import LambdaProfiler, EventProfiler
from tango.common import AsyncComponent, ComponentType, ComponentOwner
from tango.exceptions import LoadedException, StateNotReproducibleException

from abc import ABC, abstractmethod
from random import Random
from typing import Optional, Iterable
from collections import defaultdict
import asyncio

__all__ = [
    'AbstractStrategy', 'BaseStrategy', 'SeedableStrategy',
    'RolloverCounterStrategy', 'RandomStrategy', 'UniformStrategy'
]

class AbstractStrategy(AsyncComponent, IUpdateCallback, ABC,
        component_type=ComponentType.strategy,
        capture_components={ComponentType.generator}):
    def __init__(self, *, generator: AbstractInputGenerator, **kwargs):
        super().__init__(**kwargs)
        self._generator = generator

    @abstractmethod
    async def step(self, input: Optional[AbstractInput]=None):
        """
        Selects the target state according to the strategy parameters and
        returns whether or not a reset is needed.

        :returns:   Whether or not a state reset is needed.
        :rtype:     bool
        """
        pass

    @abstractmethod
    async def reload_target(self) -> AbstractState:
        pass

    @property
    @abstractmethod
    def target_state(self) -> AbstractState:
        """
        The selected target state. This is mainly used for reporting purposes.
        """
        pass

    @property
    @abstractmethod
    def target(self) -> LoadableTarget:
        """
        The last selected target state or path.
        """
        pass

class BaseStrategy(AbstractStrategy,
        capture_components={ComponentType.explorer},
        capture_paths=[
            'strategy.minimize_transitions', 'strategy.validate_transitions',
            'strategy.invalidate_on_exception']):
    def __init__(self, *,
            explorer: BaseExplorer,
            minimize_transitions: bool=True,
            validate_transitions: bool=True,
            invalidate_on_exception: bool=False,
            **kwargs):
        super().__init__(**kwargs)
        self._explorer = explorer
        self._minimize_transitions = minimize_transitions
        self._validate_transitions = validate_transitions
        self._invalidate = invalidate_on_exception
        self._step_interrupted = True
        if invalidate_on_exception:
            self._invalid_states = set()

    async def initialize(self):
        debug("Done nothing")
        return await super().initialize()

    async def step(self, input: Optional[AbstractInput]=None):
        info(f"Stepping in {self}")
        if self._step_interrupted:
            info(f"Reloading target ...")
            current_state = await self.reload_target()
            debug(f"Reloaded target")
        elif not input:
            current_state = self._explorer.tracker.current_state
        if not input:
            info(f"Generating input ...")
            input = self._generator.generate(current_state)
            debug(f"Generated input")

        self._step_interrupted = True
        await self._explorer.follow(input,
            minimize=self._minimize_transitions,
            validate=self._validate_transitions)
        self._step_interrupted = False
        debug(f"Stepped in {self}")

    @EventProfiler('reload_target')
    async def _reload_target_once(self) -> AbstractState:
        state = await self._explorer.reload_state(self.target)
        self.update_state(state, input=None)
        return state

    async def reload_target(self) -> AbstractState:
        while True:
            try:
                return await self._reload_target_once()
            except StateNotReproducibleException as ex:
                warning("Target state %s not reachable anymore!",
                    ex.faulty_state)
            except (asyncio.CancelledError, LoadedException):
                # LoadedExceptions can bubble up the exception handlers because
                # they only concern the loader/channel
                raise
            except Exception as ex:
                critical("Encountered exception while reloading target! ex=%r", ex)
                # import ipdb; ipdb.set_trace()
                pass

    def update_state(self, state: AbstractState, /, *args, exc: Exception=None,
            **kwargs):
        info(f"Updating states in {self}")
        if not self._invalidate:
            return
        elif exc:
            self._invalid_states.add(state)
        else:
            self._invalid_states.discard(state)

    @property
    def valid_targets(self) -> Iterable[LoadableTarget]:
        if self._invalidate:
            filtered = [x for x in self._explorer.tracker.state_graph.nodes
                if x not in self._invalid_states]
        else:
            filtered = list(self._explorer.tracker.state_graph.nodes)
        return filtered

class SeedableStrategy(BaseStrategy,
        capture_paths=['strategy.minimize_seeds', 'strategy.validate_seeds']):
    def __init__(self, *,
            generator: BaseInputGenerator, # must have attr `seeds`
            minimize_seeds: bool=False,
            validate_seeds: bool=False,
            **kwargs):
        super().__init__(generator=generator, **kwargs)
        self._minimize_seeds = minimize_seeds
        self._validate_seeds = validate_seeds

    async def initialize(self):
        # while loading seeds, new states may be discovered; until the session
        # takes over, we will be informing the input generator of this
        self._explorer.register_state_update_callback(self._state_update_cb)
        debug("Registered state update callback to the explorer")
        self._explorer.register_transition_update_callback(self._transition_update_cb)
        debug("Registered transition update callback to the explorer")
        await super().initialize()

    async def finalize(self, owner: ComponentOwner):
        """
        Loops over the initial set of seeds to populate the state machine with
        known states.
        """
        # TODO also load in existing queue if config.resume is True
        for input in self._generator.seeds:
            try:
                info(f"Reloading states by {self._explorer}")
                await self._explorer.reload_state()
                info(f"Seeding {input} and populating state machine by {self._explorer}")
                await self._explorer.follow(input,
                    minimize=self._minimize_seeds, validate=self._validate_seeds)
                debug(f"Loaded seed file {input}")
            except LoadedException as ex:
                warning("Failed to load %s: %s", input, ex.exception)
        info(f"Reloading state by {self._explorer}")
        await self._explorer.reload_state()
        await super().finalize(owner)

    async def _state_update_cb(self,
            state: AbstractState, /, *, breadcrumbs: LoadableTarget,
            input: AbstractInput, orig_input: AbstractInput,
            exc: Optional[Exception]=None, **kwargs):

        self._generator.update_state(state, breadcrumbs=breadcrumbs,
            input=input, orig_input=orig_input, exc=exc, **kwargs)
        self.update_state(state, breadcrumbs=breadcrumbs,
            input=input, orig_input=orig_input, exc=exc, **kwargs)

    async def _transition_update_cb(self,
            source: AbstractState, destination: AbstractState,
            input: AbstractInput, /, *, breadcrumbs: LoadableTarget,
            orig_input: AbstractInput, state_changed: bool, new_transition: bool,
            exc: Optional[Exception]=None, **kwargs):

        if new_transition:
            reproducer = await self._explorer.get_reproducer(
                input, target=breadcrumbs, validate=False)
            self._generator.save_input(reproducer, 'queue', repr(destination))

        self._generator.update_transition(source, destination, input,
            breadcrumbs=breadcrumbs, orig_input=orig_input,
            state_changed=state_changed, new_transition=new_transition, exc=exc,
            **kwargs)
        self.update_transition(source, destination, input,
            breadcrumbs=breadcrumbs, orig_input=orig_input,
            state_changed=state_changed, new_transition=new_transition, exc=exc,
            **kwargs)

class RolloverCounterStrategy(AbstractStrategy,
        capture_paths=['strategy.rollover']):
    def __init__(self, *, rollover: int=100, **kwargs):
        super().__init__(**kwargs)
        self._rollover = rollover
        self._counter = 0
        self._target = None
        LambdaProfiler('strat_counter')(lambda: self._counter)
        LambdaProfiler('strat_target')(lambda: self._target)

    async def initialize(self):
        debug("Done nothing")
        return await super().initialize()

    async def step(self, input: Optional[AbstractInput]=None):
        info(f"Stepping in {self}")
        should_reset = False
        if self._counter == 0:
            old_target = self._target
            info(f"Recalculating target ...")
            self._target = self.recalculate_target()
            debug(f"Recalculated target")
            should_reset = (old_target != self._target)
        self._counter += 1
        self._counter %= self._rollover

        if should_reset:
            info(f"Reloading target ...")
            await self.reload_target()
            debug(f"Reloaded target")
            self._step_interrupted = False
        await super().step(input)
        debug(f"Stepped in {self}")

    def update_state(self, state: AbstractState, /, *args, exc: Exception=None,
            **kwargs):
        info(f"Updating states in {self}")
        super().update_state(state, *args, exc=exc, **kwargs)
        if exc and self._target == state:
            self._counter = 0
            self._target = self.recalculate_target()

    @abstractmethod
    def recalculate_target(self) -> LoadableTarget:
        pass

    @property
    def target(self) -> LoadableTarget:
        return self._target

    @property
    def target_state(self) -> AbstractState:
        return self._target

# FIXME these strategies are not strictly required to inherit SeedableStrategy;
# a better approach may be to dynamically construct a strategy class/type that
# inherits multiple strategies to acquire the desired features where needed
class RandomStrategy(RolloverCounterStrategy, SeedableStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def initialize(self):
        debug("Done nothing")
        return await super().initialize()

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['strategy'].get('type') == 'random'

    def recalculate_target(self) -> AbstractState:
        return self._entropy.choice((*self.valid_targets, None))

    def update_transition(self, *args, **kwargs):
        pass

class UniformStrategy(RolloverCounterStrategy, SeedableStrategy):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['strategy'].get('type') == 'uniform'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._energy_map = defaultdict(lambda: 1)

    async def initialize(self):
        info(f"Initializing {self}")
        await super().initialize()
        debug(f"Initialized {self}")

    async def finalize(self, owner: ComponentOwner):
        info(f"Finalizing {self}")
        await super().finalize(owner)
        debug(f"Finalized {self}")

    async def step(self, input: Optional[AbstractInput]=None):
        info(f"Stepping in {self}")
        try:
            await super().step(input)
        finally:
            self._energy_map[self._target] += 1
            debug(f"Increased the energy of {self._target} by 1")
        debug(f"Stepped in {self}")

    def recalculate_target(self) -> AbstractState:
        filtered = self.valid_targets
        if not filtered:
            return None
        else:
            weights = [1 / self._energy_map[s] for s in filtered]
            return self._entropy.choices(filtered, weights=weights, k=1)[0]

    def update_state(self, state: AbstractState, /, *args, exc: Exception=None,
            **kwargs):
        info(f"Updating states in {self}")
        super().update_state(state, *args, exc=exc, **kwargs)
        if exc:
            self._energy_map[state] = max(*self._energy_map.values(), 1)
        else:
            self._energy_map[state] += 1

    def update_transition(self, *args, **kwargs):
        pass