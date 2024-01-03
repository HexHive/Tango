# Let's Tango!

## Usage

Build the docker images.

```
pushd scripts_eurosp && sudo docker build -t tango:latest . && popd
```

Then, from the repo root, start the container and watch the graph at
http://localhost:8080.

```
./scripts_eurosp/docker_run.sh expat
```
