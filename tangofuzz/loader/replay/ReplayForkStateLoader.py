from loader       import ReplayStateLoader
from common import sync_to_async, GLOBAL_ASYNC_EXECUTOR
import subprocess

class ReplayForkStateLoader(ReplayStateLoader):
    @sync_to_async(executor=GLOBAL_ASYNC_EXECUTOR)
    def _launch_target(self):
        if not self._pobj:
            ## Launch new process
            self._pobj = subprocess.Popen(self._exec_env.args, shell=False,
                executable = self._exec_env.path,
                stdin  = self._exec_env.stdin,
                stdout = self._exec_env.stdout,
                stderr = self._exec_env.stderr,
                cwd = self._exec_env.cwd,
                restore_signals = True, # TODO check if this should be false
                env = self._exec_env.env,
                preexec_fn = self._prepare_process
            )
        elif self._channel:
            ## Kill current process, if any
            try:
                self._channel.close(terminate=True)
            except ProcessLookupError:
                pass

        ## Establish a connection
        self._channel = self._ch_env.create(self._pobj, self._netns_name)
