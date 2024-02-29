#!/usr/bin/env bash

docker run \
    -v $PWD:/home/tango/tango \
    --entrypoint /bin/bash \
    --rm -it --privileged tango:latest
