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

exec sudo -E capsh --caps=$caps --user=$USER --addamb=$caps -- \
	-c "exec $PYBIN $PYSCRIPT $ARGS"
