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
                          StateNotReproducibleException,
                          PathNotReproducibleException)
from contextvars import ContextVar
from functools import reduce

import asyncio
import operator

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
    def __init__(self, context, sid=0, *, explorer: BaseExplorer,
            generator: BaseInputGenerator, strategy: BaseStrategy, **kwargs):
        """
        FuzzerSession initializer.
        Takes the fuzzer configuration, initial seeds, and other parameters as
        arguments.
        Initializes the target and establishes a communication channel.
        """
        super().__init__(**kwargs)
        self.context = context
        self.id = sid
        self.loop = asyncio.get_running_loop()
        self._explorer = explorer
        self._generator = generator
        self._strategy = strategy

    async def initialize(self):
        info(f"Initializing {self}")
        await super().initialize()
        self._clear = current_session.set(self)
        debug(f"Initialized {self}")

    async def finalize(self, owner: ComponentOwner):
        info(f"Finalizing {self}")
        self._explorer.register_state_reload_callback(self._state_reload_cb)
        self._explorer.register_state_update_callback(self._state_update_cb)
        self._explorer.register_transition_update_callback(self._transition_update_cb)
        debug("Registered state reload/update, trasition update callbacks for explorer")
        await super().finalize(owner)
        debug(f"Finalized {self}")

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
                    category = 'crash'
                    label = repr(self._explorer._last_state)
                    try:
                        reproducer = await self._explorer.get_reproducer(
                            ex.payload, expected_exception=ex)
                    except PathNotReproducibleException as rep:
                        prefix = reduce(operator.add, (x[2] for x in rep.faulty_path))
                        reproducer = prefix + ex.payload
                        category = 'unstable'
                        label = f'crash_{label}'
                    self._generator.save_input(reproducer, category, label)
                    # TODO augment loader to dump stdout and stderr too
                except ProcessTerminatedException as pt:
                    debug("Process terminated unexpectedly? (pt=%s)", pt)
                except ChannelTimeoutException:
                    # TODO save timeout input
                    warning("Received channel timeout exception")
                    CountProfiler('timeout')(1)
                except ChannelBrokenException as ex:
                    # TODO save crashing/breaking input
                    debug("Received channel broken exception (ex=%r)", ex)
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
                critical("Encountered weird exception ex=%r", ex)
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
            warning("Failed to follow path to state; invalidating it exc=%s",
                exc)
            self._strategy.update_state(self._strategy.target,
                input=None, exc=exc)

    async def _state_update_cb(self,
            state: AbstractState, /, *, breadcrumbs: LoadableTarget,
            input: AbstractInput, orig_input: AbstractInput,
            exc: Optional[Exception]=None, **kwargs):
        if exc:
            if isinstance(exc, StatePrecisionException):
                category = 'unstable'
            elif isinstance(exc, ProcessCrashedException):
                category = 'crash'
                error("Process crashed: exc=%s", exc)
            else:
                category = None
            label = repr(state)
            if category:
                try:
                    reproducer = await self._explorer.get_reproducer(
                        input, target=breadcrumbs, expected_exception=exc)
                except PathNotReproducibleException as ex:
                    prefix = reduce(operator.add, (x[2] for x in ex.faulty_path))
                    reproducer = prefix + ex.payload
                    category = unstable
                    label = f'crash_{label}'
                self._generator.save_input(reproducer, category, label)

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
            reproducer = await self._explorer.get_reproducer(
                input, target=breadcrumbs, validate=False)
            self._generator.save_input(reproducer, 'queue', repr(destination))

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
            info("Launching fuzzing loop and hopefully never returning :)")
            await self._loop_forever()
        finally:
            info("Remove the persistent reference in the session Context")
            get_session_context().run(current_session.reset, self._clear)
