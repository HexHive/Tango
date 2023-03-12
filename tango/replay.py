from tango.core import (BaseLoader, AbstractDriver,
    LoadableTarget, AbstractState, Transition)
from tango.unix import ProcessDriver
from tango.common import ComponentOwner
from typing       import AsyncGenerator

__all__ = ['ReplayStateLoader', 'ReplayForkStateLoader']

class ReplayStateLoader(BaseLoader,
        capture_components={'driver'}):
    def __init__(self, *, driver: AbstractDriver, **kwargs):
        super().__init__(**kwargs)
        self._driver = driver

    async def finalize(self, owner: ComponentOwner):
        generator = owner['generator']
        self._startup = getattr(generator, 'startup_input', None)
        await super().finalize(owner)

    async def load_state(self, state_or_path: LoadableTarget) \
            -> AsyncGenerator[Transition, AbstractState]:
        if isinstance(path := state_or_path, AbstractState):
            raise TypeError(f"{self.__class__.__name__} can only load paths!")

        # relaunch the target and establish channel
        await self._driver.relaunch()

        if self._startup:
            # Send startup input
            await self._driver.execute_input(self._startup)

        if not path:
            return

        last_state = None
        for source, destination, input in path:
            current_state = self._tracker.peek(last_state, source)
            yield (last_state, source, None)
            # check if source matches the current state
            if source != current_state:
                raise StabilityException(
                    f"source state ({source}) did not match current state ({current_state})"
                )
            # execute the input
            await self._driver.execute_input(input)

            current_state = self._tracker.peek(source, destination)
            yield (source, destination, input)
            # check if destination matches the current state
            if destination != current_state:
                faulty_state = destination
                raise StabilityException(
                    f"destination state ({destination}) did not match current state ({current_state})"
                )
            last_state = current_state

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['loader'].get('type') == 'replay'