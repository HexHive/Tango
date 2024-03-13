from . import debug, info, warning, critical
from tango.core import (BaseLoader, AbstractDriver,
    LoadableTarget, Path, AbstractState, Transition)
from tango.unix import ProcessDriver
from tango.common import ComponentOwner
from tango.exceptions import StabilityException
from typing       import AsyncGenerator, Any

__all__ = ['ReplayLoader', 'ReplayForkLoader']

class ReplayLoader(BaseLoader,
        capture_components={'driver'}):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['loader'].get('type') == 'replay'

    def __init__(self, *, driver: AbstractDriver, **kwargs):
        super().__init__(**kwargs)
        self._driver = driver

    async def initialize(self):
        debug("Done nothing")
        return await super().initialize()

    async def finalize(self, owner: ComponentOwner):
        generator = owner['generator']
        self._startup = getattr(generator, 'startup_input', None)
        debug("Registered generator.startup_input")
        await super().finalize(owner)

    async def apply_transition(self, transition: Transition,
            current_state: AbstractState, **kwargs) -> AbstractState:
        source, destination, input = transition
        current_state = current_state or source
        # check if source matches the current state
        if source != current_state:
            raise StabilityException(
                f"source state ({source}) did not match current state ({current_state})",
                current_state, source
            )
        info(f"Executing the input {input}")
        await self._driver.execute_input(input)

        info(f"Peeking ({source}, {destination}) by {self._tracker}")
        current_state = self._tracker.peek(source, destination, **kwargs)
        debug(f"Got current state {current_state} by peeking {destination} by {self._tracker}")
        # check if destination matches the current state
        if destination != current_state:
            raise StabilityException(
                f"destination state ({destination}) did not match current state ({current_state})",
                current_state, destination
            )
        return current_state

    async def load_path(self, path: Path, **kwargs):
        src, _, _ = path[0]
        info(f"Peeking (None, {src}) by {self._tracker}")
        last_state = self._tracker.peek(expected_destination=src, **kwargs)

        for transition in path:
            debug(f"Applying transition {transition} with {last_state}")
            last_state = await self.apply_transition(transition, last_state,
                **kwargs)

    async def load_state(self, state_or_path: LoadableTarget) \
            -> AsyncGenerator[Transition, Any]:
        if isinstance(path := state_or_path, AbstractState):
            raise TypeError(f"{self.__class__.__name__} can only load paths!")

        # It seems that relaunching the driver should go through
        # ReplayLoader.load_state(). CoverageTracker has its own implementation
        # in CoverageTracker.finalize() to relaunch the driver to set up
        # coverage maps/fork server (if required). The implementation has some
        # overlap with ReplayLoader.load_state(), which could be optimized.
        info(f"Relaunching {self._driver}")
        await self._driver.relaunch()
        debug(f"Relaunched {self._driver}")

        if self._startup:
            info(f"Executing startup input {self._startup} with {self._driver}")
            await self._driver.execute_input(self._startup)

        if not path:
            return

        full_path = list(path)
        info(f"Loading path {path}")
        await self.load_path(full_path, update_cache=False)
        debug(f"Loaded path {path}")
        for transition in full_path:
            debug(f"Got transition {transition}")
            yield transition