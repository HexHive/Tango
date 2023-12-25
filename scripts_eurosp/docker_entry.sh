#!/usr/bin/env bash

cd $HOME/tango
./scripts_eurosp/git_install.sh -y
sudo ln -s /usr/bin/clang-13 /usr/bin/clang
sudo ln -s /usr/bin/clang++-13 /usr/bin/clang++
source .venv/bin/activate

python="$(realpath "$(which python)")"
sudo setcap \
    cap_net_admin,cap_sys_admin,cap_dac_override,cap_chown,cap_fowner,cap_setpcap,cap_setuid,cap_setgid+eip \
    $python

target=$1
shift
pushd targets
USE_ASAN=1 make $target/
popd

python main.py -v targets/$target/fuzz.json -o webui.http_host 0.0.0.0 $@
