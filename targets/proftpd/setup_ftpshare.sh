#!/bin/bash
set -e

if [ ! -d "/home/ftpshare" ]; then
    mkdir /home/ftpshare
    chown -R tango:tango /home/ftpshare
fi
