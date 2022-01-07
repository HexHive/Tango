from . import debug, critical

from input         import (InputBase,
                          PreparedInput)
from fuzzer        import FuzzerConfig
from common        import (StabilityException,
                          StatePrecisionException,
                          LoadedException,
                          ChannelTimeoutException,
                          ChannelBrokenException,
                          ChannelSetupException,
                          ProcessCrashedException)
import os

from profiler import (ProfileLambda,
                     ProfileCount,
                     ProfiledObjects,
                     ProfilingStoppedEvent,
                     ProfileTimeElapsed)

from webui import WebRenderer

class FuzzerSession:
    """
    This class initializes and tracks the global state of the fuzzer.
    """
    def __init__(self, config: FuzzerConfig):
        """
        FuzzerSession initializer.
        Takes the fuzzer configuration, initial seeds, and other parameters as
        arguments.
        Initializes the target and establishes a communication channel.

        :param      config: The fuzzer configuration object
        :type       config: FuzzerConfig
        """

        self._input_gen = config.input_generator
        self._loader = config.loader
        self._sman = config.state_manager
        self._entropy = config.entropy

        ## After this point, the StateManager and StateTracker are both
        #  initialized and should be able to identify states and populate the SM

        self._load_seeds()

    def _load_seeds(self):
        """
        Loops over the initial set of seeds to populate the state machine with
        known states.
        """
        for input in self._input_gen.seeds:
            self._sman.reset_state()
            # feed input to target and populate state machine
            self._loader.execute_input(input, self._sman)

    def _loop(self):
        # FIXME is there ever a proper terminating condition for fuzzing?
        while True:
            try:
                # FIXME should probably reset to StateManager's current target state
                self._sman.reset_state()
                while True:
                    try:
                        cur_state = self._sman.state_tracker.current_state
                        input = self._input_gen.generate(cur_state, self._entropy)
                        self._sman.step(input)
                    except LoadedException as ex:
                        try:
                            raise ex.exception
                        except StabilityException:
                            ProfileCount('unstable')(1)
                        except StatePrecisionException:
                            debug("Encountered imprecise state transition")
                            ProfileCount('imprecise')(1)
                        except ProcessCrashedException:
                            # TODO save crashing input
                            ProfileCount('crash')(1)
                        except ChannelTimeoutException:
                            # TODO save timeout input
                            critical("Received channel timeout exception")
                            ProfileCount('timeout')(1)
                        except ChannelBrokenException:
                            # TODO save crashing/breaking input
                            critical("Received channel broken exception")
                        except ChannelSetupException:
                            # TODO save broken setup input
                            critical("Received channel setup exception")
                        except Exception as ex:
                            critical(f"Encountered unhandled loaded exception {ex = }")
                        # FIXME should probably reset to StateManager's current target state
                        self._sman.reset_state()
                    except Exception as ex:
                        critical(f"Encountered weird exception {ex = }")
                        self._sman.reset_state()
            except Exception as ex:
                critical(f"Encountered exception while resetting state! {ex = }")
            except KeyboardInterrupt:
                from code import InteractiveConsole
                repl = InteractiveConsole(locals=locals())
                repl.interact(banner="Fuzzing paused", exitmsg="Fuzzing resumed")

    def start(self):
        # reset state after the seed initialization stage
        self._sman.reset_state()

        ProfileTimeElapsed('elapsed')

        WebRenderer().start()

        # launch fuzzing loop
        self._loop()

        # TODO anything else?
