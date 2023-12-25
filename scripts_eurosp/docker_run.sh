#!/usr/bin/env bash

docker run \
    -v $PWD:/home/tango \
    --rm -it --privileged tango $@
