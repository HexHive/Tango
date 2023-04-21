#!/usr/bin/env bash

PYSCRIPT="${PYSCRIPT:-main.py}"
PYBIN="${PYBIN:-$(which python3)}"
USER=${USER:-$(whoami)}

ARGS="$@"

caps=(cap_net_admin cap_sys_admin cap_dac_override cap_chown cap_fowner \
	  cap_setpcap)
IFS=','
caps="${caps[*]}"=eip
unset IFS

sudo -E capsh --user=$USER --keep=1 --caps=$caps -- \
	-c "$PYBIN $PYSCRIPT $ARGS"
