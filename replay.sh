#!/usr/bin/env bash

source .venv/bin/activate

python="$(realpath "$(which python)")"
sudo setcap \
    cap_net_admin,cap_sys_admin,cap_dac_override,cap_chown,cap_fowner,cap_setpcap,cap_setuid,cap_setgid,cap_sys_ptrace+eip \
    $python

target=$1
shift

python replay.py -v targets/$target/fuzz.json \
    -o driver.isolate_fs false \
    -o driver.exec.stdout inherit \
    -o driver.exec.stderr inherit \
    "$@"
