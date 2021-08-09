from input         import (InputBase,
                          PreparedInput)
from fuzzer        import FuzzerConfig
from common        import StabilityException
import os
from threading import Timer
from time import sleep

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

        # stats stuff
        self._counter = 0
        self._unstable = 0

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
                cur_state = self._sman.state_tracker.current_state
                input = self._input_gen.generate(cur_state, self._entropy)
                self._loader.execute_input(input, self._sman)
                self._sman.step()
            except StabilityException:
                self._unstable += 1
                self._sman.reset_state()

            self._counter += 1

    def _stats(self, delay):
        while self._working:
            print(f"execs/s {self._counter / delay} cov {len(self._sman._sm._graph.nodes)} unstable {self._unstable}")
            self._counter = 0
            sleep(delay)

    def start(self):
        # reset state after the seed initialization stage
        self._sman.reset_state()
        self._working = True

        delay = 1.0
        Timer(delay, self._stats, (delay,)).start()

        # launch fuzzing loop
        try:
            self._loop()
        except Exception as ex:
            self._working = False
            import code
            code.interact(local=locals())
            raise

        # TODO anything else?
        pass