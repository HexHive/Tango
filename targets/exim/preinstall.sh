#!/bin/bash
set -e

apt-get update && \
    apt-get install -y libpcre2-dev libdb-dev