from loader import AbstractStateLoader
from common       import LoadedException
from dataio    import AbstractChannelFactory
from input        import AbstractInput
from generator    import AbstractInputGenerator
from tracker import AbstractStateTracker
from profiler import ValueMeanProfiler, CountProfiler

class BaseStateLoader(AbstractStateLoader):
    def __init__(self, ch_env: AbstractChannelFactory, input_generator: AbstractInputGenerator):
        self._ch_env = ch_env
        self._generator = input_generator
        self._tracker = None

    @property
    def state_tracker(self) -> AbstractStateTracker:
        return self._tracker

    @state_tracker.setter
    def state_tracker(self, value: AbstractStateTracker):
        self._tracker = value

    async def execute_input(self, input: AbstractInput):
        """
        Executes the sequence of interactions specified by the input.

        :param      input:  An object derived from the AbstractInput abstract class.
        :type       input:  AbstractInput

        :raises:    LoadedException: Forwards the exception that was raised
                    during execution, along with the input that caused it.
        """
        try:
            idx = -1
            async for interaction in input:
                idx += 1
                await interaction.perform(self.channel)
        except Exception as ex:
            raise LoadedException(ex, input[:idx + 1]) from ex
        finally:
            ValueMeanProfiler("input_len", samples=100)(idx + 1)
            CountProfiler("total_interactions")(idx + 1)
