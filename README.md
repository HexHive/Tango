# Let's Tango!

Tango can fuzz multiple stateful targets, i.e., 1) protocol targets: bftpd,
daap, dcmtk, dnsmasq, dtls, exim, lightftp, openssh, openssl, proftpd, pureftpd,
rtsp, and sip, 2) parsers: expat, llhttp, and yaml.

Targets snapshotted on 20240229.
- [expat](https://github.com/libexpat/libexpat): a387201
- [llhttp](https://github.com/nodejs/llhttp): a35e183
- [yajl](https://github.com/openEuler-BaseService/yajl): 175a30f

Other targets are consistent with ProFuzzBench.

## Usage

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
