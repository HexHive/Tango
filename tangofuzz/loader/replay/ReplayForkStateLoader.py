from loader       import ReplayStateLoader
from common import sync_to_async, GLOBAL_ASYNC_EXECUTOR

class ReplayForkStateLoader(ReplayStateLoader):
    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def _launch_target(self):
        if not self._pobj:
            ## Launch new process
            self._pobj = self._popen()
        elif self._channel:
            ## Kill current process, if any
            try:
                self._channel.close(terminate=True)
            except ProcessLookupError:
                pass

        ## Establish a connection
        self._channel = self._ch_env.create(self._pobj, self._netns_name)
