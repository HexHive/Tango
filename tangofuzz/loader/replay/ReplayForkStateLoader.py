from loader       import ReplayStateLoader
from ptrace.binding import ptrace_traceme
import subprocess

class ReplayForkStateLoader(ReplayStateLoader):
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
                preexec_fn = ptrace_traceme
            )
        elif self._channel:
            ## Kill current process, if any
            try:
                self._channel.close(terminate=True)
            except ProcessLookupError:
                pass

        ## Establish a connection
        self._channel = self._ch_env.create(self._pobj)
