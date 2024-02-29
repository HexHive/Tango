#!/usr/bin/env bash

docker run \
    -v $PWD:/home/tango \
    -p 8080:8080 \
    --rm -it --privileged tango:latest $@
