#!/usr/bin/env bash

if [ ! -d .venv ]; then
    ./scripts/git_install.sh -y
fi
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
    elif [[ $target == "daap" ]]; then
        USE_ASAN=1 make $target/
    else
        USE_ASAN=1 make $target/
    fi
    popd
fi

if [ -f targets/$target/postinstall.sh ]; then
    ./targets/$target/postinstall.sh
fi

python main.py -v targets/$target/fuzz.json \
    -o driver.isolate_fs false \
    -o webui.http_host 0.0.0.0 \
    $@
