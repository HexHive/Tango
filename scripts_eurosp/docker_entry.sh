#!/usr/bin/env bash

cd $HOME/tango
source .venv/bin/activate

python="$(realpath "$(which python)")"
sudo setcap cap_net_admin,cap_sys_admin,cap_dac_override+eip $python

target=$1
shift
pushd targets
USE_ASAN=1 make $target/
popd

python main.py -v targets/$target/fuzz.json -o webui.http_host 0.0.0.0 $@