from __future__ import annotations

from . import debug, info, warning, critical, error

from tango.core.tracker import AbstractState
from tango.core.input     import AbstractInput
from tango.core.types import LoadableTarget
from tango.core.profiler import CountProfiler
from tango.core.explorer import BaseExplorer
from tango.core.generator import BaseInputGenerator
from tango.core.strategy import BaseStrategy
from tango.common import (AsyncComponent, ComponentType, Suspendable,
    ComponentOwner, get_session_context)
from tango.exceptions import (StabilityException,
                          StatePrecisionException,
                          LoadedException,
                          ChannelTimeoutException,
                          ChannelBrokenException,
                          ChannelSetupException,
                          ProcessCrashedException,
                          ProcessTerminatedException,
                          StateNotReproducibleException)
from contextvars import ContextVar

import asyncio

__all__ = ['FuzzerSession', 'get_current_session', 'set_current_session']

current_session: ContextVar[FuzzerSession] = ContextVar('session')

def get_current_session():
    try:
        return get_session_context()[current_session]
    except KeyError as ex:
        raise RuntimeError("No session in current session context!") from ex

def set_current_session(session: FuzzerSession):
    get_session_context().run(current_session.set, session)

class FuzzerSession(AsyncComponent, component_type=ComponentType.session,
        capture_components={ComponentType.explorer,
            ComponentType.generator, ComponentType.strategy}):
    """
    This class initializes and tracks the global state of the fuzzer.
    """
    def __init__(self, sid, *, explorer: BaseExplorer,
            generator: BaseInputGenerator, strategy: BaseStrategy, **kwargs):
        """
        FuzzerSession initializer.
        Takes the fuzzer configuration, initial seeds, and other parameters as
        arguments.
        Initializes the target and establishes a communication channel.
        """
        super().__init__(**kwargs)
        self.id = sid
        self.loop = asyncio.get_running_loop()
        self._explorer = explorer
        self._generator = generator
        self._strategy = strategy

    async def initialize(self):
        await super().initialize()
        # needed for accessing session-specific variables while debugging
        self._context = get_session_context()
        self._clear = current_session.set(self)

    async def finalize(self, owner: ComponentOwner):
        self._explorer.register_state_reload_callback(self._state_reload_cb)
        self._explorer.register_state_update_callback(self._state_update_cb)
        self._explorer.register_transition_update_callback(self._transition_update_cb)
        await super().finalize(owner)

    async def _loop_forever(self):
        while True:
            try:
                await self._strategy.step()
            except LoadedException as ex:
                # only raised by the loader, after the strategy target is
                # already reached, i.e., in reaction to a generated input
                try:
                    raise ex.exception
                except StabilityException:
                    debug("Encountered unstable path")
                    CountProfiler('unstable')(1)
                except StatePrecisionException:
                    debug("Encountered imprecise tail state transition")
                    CountProfiler('imprecise')(1)
                except ProcessCrashedException as pc:
                    error(f"Process crashed: {pc = }")
                    CountProfiler('crash')(1)
                    # TODO augment loader to dump stdout and stderr too
                    self._generator.save_input(ex.payload,
                        self._explorer._current_path, 'crash',
                        repr(self._explorer._last_state))
                except ProcessTerminatedException as pt:
                    debug(f"Process terminated unexpectedtly? ({pt = })")
                except ChannelTimeoutException:
                    # TODO save timeout input
                    warning("Received channel timeout exception")
                    CountProfiler('timeout')(1)
                except ChannelBrokenException as ex:
                    # TODO save crashing/breaking input
                    debug(f"Received channel broken exception ({ex = })")
                except ChannelSetupException:
                    # TODO save broken setup input
                    warning("Received channel setup exception")
                except StateNotReproducibleException as ex:
                    warning(f"Failed to reach loadable target")

                # other loaded exceptions will bubble up into the outer handler
                # for inspection
            except asyncio.CancelledError:
                return
            except Exception as ex:
                # everything else, we probably need to debug
                critical(f"Encountered weird exception {ex = }")
                import ipdb; ipdb.set_trace()
                raise

    async def _state_reload_cb(self,
            loadable: LoadableTarget, /, *, exc: Optional[Exception]=None):
        if not exc:
            # called when the loader needs a startup input to reload the target
            # FIXME not very clean, consider using arbitration protocols
            return self._generator.startup_input
        elif isinstance(exc, StateNotReproducibleException):
            faulty_state = exc.faulty_state
            self._strategy.update_state(faulty_state, input=None, exc=exc)
            if hasattr(loadable, '__iter__'):
                # find and invalidate the transition along the path that
                # leads to the faulty state
                try:
                    transition = next(x for x in loadable if x[1] == faulty_state)
                    self._strategy.update_transition(*transition,
                        state_changed=transition[0]==transition[1],
                        exc=exc)
                except StopIteration:
                    pass
        elif isinstance(exc, StabilityException):
            faulty_state = exc.expected_state
            self._strategy.update_state(faulty_state, input=None, exc=exc)
        else:
            # In this case, we need to force the strategy to yield a new
            # target, because we're not entirely sure what went wrong. We
            # invalidate the target state and hope for the best.
            warning(f'Failed to follow path to state; invalidating it {exc=}')
            self._strategy.update_state(self._strategy.target,
                input=None, exc=exc)

    async def _state_update_cb(self,
            state: AbstractState, /, *, breadcrumbs: LoadableTarget,
            input: AbstractInput, orig_input: AbstractInput,
            exc: Optional[Exception]=None, **kwargs):
        if exc:
            if isinstance(exc, StatePrecisionException):
                label = 'unstable'
            elif isinstance(exc, ProcessCrashedException):
                label = 'crash'
                error(f"Process crashed: {exc = }")
            else:
                label = None
            if label:
                self._generator.save_input(input, breadcrumbs, label, repr(state))

            if isinstance(exc, ProcessCrashedException):
                # FIXME kinda ugly
                return

        self._generator.update_state(state, breadcrumbs=breadcrumbs,
            input=input, orig_input=orig_input, exc=exc, **kwargs)
        self._strategy.update_state(state, breadcrumbs=breadcrumbs,
            input=input, orig_input=orig_input, exc=exc, **kwargs)

    async def _transition_update_cb(self,
            source: AbstractState, destination: AbstractState,
            input: AbstractInput, /, *, breadcrumbs: LoadableTarget,
            orig_input: AbstractInput, state_changed: bool, new_transition: bool,
            exc: Optional[Exception]=None, **kwargs):

        if new_transition:
            self._generator.save_input(input, breadcrumbs, 'queue', repr(destination))

        self._generator.update_transition(source, destination, input,
            breadcrumbs=breadcrumbs, orig_input=orig_input,
            state_changed=state_changed, new_transition=new_transition, exc=exc,
            **kwargs)
        self._strategy.update_transition(source, destination, input,
            breadcrumbs=breadcrumbs, orig_input=orig_input,
            state_changed=state_changed, new_transition=new_transition, exc=exc,
            **kwargs)

    async def run(self):
        try:
            # launch fuzzing loop
            await self._loop_forever()
        finally:
            # remove the persistent reference in the session Context
            get_session_context().run(current_session.reset, self._clear)
