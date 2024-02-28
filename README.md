# Let's Tango!

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

To debug, try the following command lines.

```
# skip to build the target
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
