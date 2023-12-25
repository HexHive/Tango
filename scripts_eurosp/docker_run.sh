#!/usr/bin/env bash

docker build -t tango . && \
docker run --rm -it --privileged -p 8080:8080 tango $@
