#!/usr/bin/env bash

PYSCRIPT="${PYSCRIPT:-main.py}"
PYBIN="${PYBIN:-$(which python3)}"
USER=${USER:-$(whoami)}

ARGS="$@"

caps=(cap_net_admin cap_sys_admin cap_dac_override cap_chown cap_fowner \
	  cap_setpcap cap_setuid cap_setgid)
IFS=','
caps="${caps[*]}"=eip
unset IFS

sudo -E capsh --caps=$caps --user=$USER --addamb=$caps -- \
	-c "$PYBIN $PYSCRIPT $ARGS" &

pid=$!
trap "echo 'Saving progress...'; kill -sUSR1 $pid; wait" SIGUSR1
trap "echo 'Terminating fuzzer!'; sudo kill -sTERM $pid; wait" SIGTERM
wait