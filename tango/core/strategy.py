from . import info, warning

from tango.core.tracker import AbstractState
from tango.core.input     import AbstractInput
from tango.core.types import LoadableTarget
from tango.core.explorer import BaseExplorer
from tango.core.generator import AbstractInputGenerator, BaseInputGenerator
from tango.core.profiler import LambdaProfiler, EventProfiler
from tango.common import Configurable, ComponentType
from tango.exceptions import LoadedException

from abc import ABC, abstractmethod
from random import Random
from typing import Optional

__all__ = [
    'AbstractStrategy', 'BaseStrategy', 'SeedableStrategy',
    'RolloverCounterStrategy', 'RandomStrategy', 'UniformStrategy'
]

class AbstractStrategy(Configurable, ABC,
        component_type=ComponentType.strategy,
        capture_components={ComponentType.input_generator}):
    def __init__(self, *, input_generator: AbstractInputGenerator):
        self._generator = input_generator

    @abstractmethod
    def update_state(self, state: AbstractState, *, input: AbstractInput, exc: Exception=None, **kwargs):
        """
        Updates the internal strategy parameters related to the state. In case a
        state is invalidated, it should remain so until it is revalidated in a
        following call to update_state(). Otherwise, invalidated states must not
        be selected at the target state.

        :param      state:  The current state of the target.
        :type       state:  AbstractState
        :param      exc:    An exception that occured while processing the input.
        :type       exc:    Exception
        """
        pass

    @abstractmethod
    def update_transition(self, source: AbstractState, destination: AbstractState, input: AbstractInput, *, state_changed: bool, exc: Exception=None, **kwargs):
        """
        Similar to update_state(), but for transitions.

        :param      source:       The source state of the transition.
        :type       source:       AbstractState
        :param      destination:  The destination state. This can be assumed to
                                  be the current state of the target too.
        :type       destination:  AbstractState
        :param      input:        The input associated with the transition.
        :type       input:        AbstractInput
        :param      exc:          An exception that occured while processing the
                                  input.
        :type       exc:          Exception
        """
        pass

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
        capture_paths=['strategy.minimize_transitions', 'strategy.validate_transitions']):
    def __init__(self, *,
            explorer: BaseExplorer,
            minimize_transitions: bool=True,
            validate_transitions: bool=True,
            **kwargs):
        super().__init__(**kwargs)
        self._explorer = explorer
        self._minimize_transitions = minimize_transitions
        self._validate_transitions = validate_transitions

    @EventProfiler('strategy_step')
    async def step(self, input: Optional[AbstractInput]=None):
        if not input:
            current_state = self._explorer.tracker.current_state
            input = self._generator.generate(current_state)
        await self._explorer.follow(input,
            minimize=self._minimize_transitions,
            validate=self._validate_transitions)

    async def reload_target(self) -> AbstractState:
        state = await self._explorer.reload_state(self.target)
        self.update_state(state, input=None)

class SeedableStrategy(BaseStrategy,
        capture_paths=['strategy.minimize_seeds', 'strategy.validate_seeds']):
    def __init__(self, *,
            input_generator: BaseInputGenerator, # must have attr `seeds`
            minimize_seeds: bool=False,
            validate_seeds: bool=False,
            **kwargs):
        super().__init__(input_generator=input_generator, **kwargs)
        self._minimize_seeds = minimize_seeds
        self._validate_seeds = validate_seeds

    async def initialize(self):
        """
        Loops over the initial set of seeds to populate the state machine with
        known states.
        """
        # TODO also load in existing queue if config.resume is True
        await super().initialize()

        # while loading seeds, new states may be discovered; until the session
        # takes over, we will be informing the input generator of this
        self._explorer.register_state_update_callback(self._state_update_cb)
        self._explorer.register_transition_update_callback(self._transition_update_cb)

        for input in self._generator.seeds:
            try:
                await self._explorer.reload_state()
                # feed input to target and populate state machine
                await self._explorer.follow(input,
                    minimize=self._minimize_seeds, validate=self._validate_seeds)
                info(f"Loaded seed file: {input}")
            except LoadedException as ex:
                warning(f"Failed to load {input}: {ex.exception}")
        await self._explorer.reload_state()

    async def _state_update_cb(self,
            state: AbstractState, *, breadcrumbs: LoadableTarget,
            input: AbstractInput, orig_input: AbstractInput,
            exc: Optional[Exception]=None, **kwargs):

        self._generator.update_state(state, breadcrumbs=breadcrumbs,
            input=input, orig_input=orig_input, exc=exc, **kwargs)

    async def _transition_update_cb(self,
            source: AbstractState, destination: AbstractState,
            input: AbstractInput, *, breadcrumbs: LoadableTarget,
            orig_input: AbstractInput, state_changed: bool, new_transition: bool,
            exc: Optional[Exception]=None, **kwargs):

        if new_transition:
            self._generator.save_input(input, breadcrumbs, 'queue', repr(destination))

        self._generator.update_transition(source, destination, input,
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

    async def step(self, input: Optional[AbstractInput]=None):
        should_reset = False
        if self._counter == 0:
            old_target = self._target
            self._target = self.recalculate_target()
            should_reset = (old_target != self._target)
        self._counter += 1
        self._counter %= self._rollover

        if should_reset:
            await self._explorer.reload_state(self._target)
        await super().step(input)

    @abstractmethod
    def recalculate_target(self) -> LoadableTarget:
        pass

    @property
    def target(self) -> LoadableTarget:
        return self._target

# FIXME these strategies are not strictly required to inherit SeedableStrategy;
# a better approach may be to dynamically construct a strategy class/type that
# inherits multiple strategies to acquire the desired features where needed
class RandomStrategy(RolloverCounterStrategy, SeedableStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._invalid_states = set()

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['strategy'].get('type') == 'random'

    def recalculate_target(self) -> AbstractState:
        self._counter = 0
        filtered = [x for x in self._explorer.state_graph._graph.nodes if x not in self._invalid_states]
        if not filtered:
            return None
        else:
            return self._entropy.choice(filtered)

    def update_state(self, state: AbstractState, *args, exc: Exception=None, **kwargs):
        super().update_state(state, *args, exc=exc, **kwargs)
        if exc:
            self._invalid_states.add(state)
            if self._target == state:
                self._target = self.recalculate_target()
        else:
            self._invalid_states.discard(state)

    def update_transition(self, *args, **kwargs):
        pass

    @property
    def target_state(self) -> AbstractState:
        return self._target

class UniformStrategy(RolloverCounterStrategy, SeedableStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._invalid_states = set()

        self._exp_weights = (0.64, 0.23, 0.09, 0.03, 0.01)
        self._calc_weights = lambda n: (
            self._exp_weights[0] + sum(self._exp_weights[n:]),
            *self._exp_weights[1:n],
            *((0.0,) * (n - len(self._exp_weights))))[:n]

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['strategy'].get('type') == 'uniform'

    def recalculate_target(self) -> AbstractState:
        filtered = [x for x in self._explorer.state_graph._graph.nodes if x not in self._invalid_states]
        if not filtered:
            return None
        else:
            filtered.sort(key=lambda x: getattr(x, '_energy', 0))
            return self._entropy.choices(filtered,
                        weights=self._calc_weights(len(filtered)),
                        k=1)[0]

    def update_state(self, state: AbstractState, *args, exc: Exception=None, **kwargs):
        super().update_state(state, *args, exc=exc, **kwargs)
        if state is None:
            return
        if exc:
            self._invalid_states.add(state)
            if self._target == state:
                self._recalculate_target()
        else:
            self._invalid_states.discard(state)
            if not hasattr(state, '_energy'):
                state._energy = 1
            else:
                state._energy += 1

    def update_transition(self, *args, **kwargs):
        pass

    @property
    def target_state(self) -> AbstractState:
        return self._target