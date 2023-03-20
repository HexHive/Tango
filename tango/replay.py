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

    async def finalize(self, owner: ComponentOwner):
        generator = owner['generator']
        self._startup = getattr(generator, 'startup_input', None)
        await super().finalize(owner)

    async def apply_transition(self, transition: Transition,
            current_state: AbstractState, **kwargs) -> AbstractState:
        source, destination, input = transition
        current_state = current_state or source
        # check if source matches the current state
        if source != current_state:
            raise StabilityException(
                f"source state ({source}) did not match current state ({current_state})",
                current_state
            )
        # execute the input
        await self._driver.execute_input(input)

        current_state = self._tracker.peek(source, destination, **kwargs)
        # check if destination matches the current state
        if destination != current_state:
            raise StabilityException(
                f"destination state ({destination}) did not match current state ({current_state})",
                current_state
            )
        return current_state

    async def load_path(self, path: Path, **kwargs):
        src, _, _ = path[0]
        last_state = self._tracker.peek(expected_destination=src, **kwargs)
        for transition in path:
            last_state = await self.apply_transition(transition, last_state,
                **kwargs)

    async def load_state(self, state_or_path: LoadableTarget) \
            -> AsyncGenerator[Transition, Any]:
        if isinstance(path := state_or_path, AbstractState):
            raise TypeError(f"{self.__class__.__name__} can only load paths!")

        # relaunch the target and establish channel
        await self._driver.relaunch()

        if self._startup:
            # Send startup input
            await self._driver.execute_input(self._startup)

        if not path:
            return

        full_path = list(path)
        await self.load_path(full_path, update_cache=False)
        for transition in full_path:
            yield transition