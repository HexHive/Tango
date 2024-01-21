#!/usr/bin/env bash

cd $HOME/tango
# ./scripts_eurosp/git_install.sh -y
source .venv/bin/activate

python="$(realpath "$(which python)")"
sudo setcap \
    cap_net_admin,cap_sys_admin,cap_dac_override,cap_chown,cap_fowner,cap_setpcap,cap_setuid,cap_setgid,cap_sys_ptrace+eip \
    $python

target=$1
shift
if [[ $target =~ "skip_build" ]]; then
    target=${target%"_skip_build"}
else
    pushd targets
    if [[ $target == "pureftpd" ]]; then
        make $target/
    else
        USE_ASAN=1 make $target/
    fi
    popd
fi

python main.py -v targets/$target/fuzz.json \
    -o driver.isolate_fs false \
    -o webui.http_host 0.0.0.0 \
    $@
