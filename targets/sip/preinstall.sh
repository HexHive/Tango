#!/bin/bash
set -e

apt-get update && \
    apt-get install -y flex bison libmysqlclient-dev