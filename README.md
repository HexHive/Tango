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

Use `Ctrl-c` to pause Tango and type `Exit` to exit.

To debug, try the following command lines.

```
# skip build the target
./scripts/docker_run.sh expat_skip_build
# show info
./scripts/docker_run.sh expat_skip_build -v
# show info and debug
./scripts/docker_run.sh expat_skip_build -vv
# show all syscalls
./scripts/docker_run.sh expat_skip_build -vv --show_syscalls
# show covered and symbolized PCs (verbose ouput if the target has symbols)
./scripts/docker_run.sh expat_skip_build -vv --show_syscalls --show_pcs
```
