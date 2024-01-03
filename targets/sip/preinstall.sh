#!/bin/bash
set -e

sudo apt-get update && \
    sudo apt-get install -y flex bison libmysqlclient-dev
