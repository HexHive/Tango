# Let's Tango!

Tango can fuzz stateful targets, i.e., 1) protocol targets: bftpd, dcmtk,
dnsmasq, tinydtls, exim, lightftp, openssh, openssl, proftpd, pureftpd, live555, and
kamailio, 2) parsers: expat, llhttp, and yaml. (forked-daap is not supported.)

Please refer to [MagmaStateful](https://github.com/HexHive/MagmaStateful) to
check the performance of Tango against the recent targets.

Please check [tango-evaluation](https://github.com/cyruscyliu/tango-evaluation)
to reproduce figures in the paper if you are interested.

## Usage for Debugging Locally

Setup Tango and build the docker images.

```
git submodule update --init --recursive
pushd scripts && sudo docker build -t tango:latest . && popd
```

Then, from the repo root, start the container and watch the graph at
http://localhost:8080.

```
./scripts/docker_run.sh expat
```

To enable the state inference, use `./scripts/docker_run.inference.sh`. Two
notes, 1) a crosstest_0.csv is stored into the workdir. 2) webui profiler is not
supported during inference by default.

Use `Ctrl-c` to pause Tango and type `Exit` to exit.

To debug, try the following command lines.

```
# skip build the target
./scripts/docker_run.sh expat_skip_build

# show info-level logs
./scripts/docker_run.sh expat_skip_build -v

# show info-level and debug-level logs
./scripts/docker_run.sh expat_skip_build -vv

# show all syscalls (working with -vv)
./scripts/docker_run.sh expat_skip_build -vv --show_syscalls

# show covered and symbolized PCs (verbose ouput if the target has symbols) (working with -vv)
./scripts/docker_run.sh expat_skip_build -vv --show_syscalls --show_pcs
```

To reproduce a crash, go into the container and run `replay.sh`.

```
./scripts/docker_run.debug.sh
cd /home/tango
./replay.sh expat absolute_path_to_poc [-v[v]] [--show_syscalls]
```
