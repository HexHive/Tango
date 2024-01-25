#!/usr/bin/env bash

PYSCRIPT="${PYSCRIPT:-main.py}"
PYBIN="${PYBIN:-$(which python3)}"
USER=${USER:-$(whoami)}

ARGS="$@"

caps=(cap_net_admin cap_sys_admin cap_dac_override cap_chown cap_fowner \
	  cap_setpcap cap_setuid cap_setgid cap_sys_ptrace)
IFS=','
caps="${caps[*]}"=eip
unset IFS

if [ ! -z $TANGO_TIMEOUT ]; then
	timeout=(timeout -k 1m --foreground -sTERM $TANGO_TIMEOUT)
fi

exec sudo -E capsh --caps=$caps --user=$USER --addamb=$caps --\
	-c "exec ${timeout[*]} $PYBIN $PYSCRIPT $ARGS"
