#!/bin/bash
set -e

apt-get update && \
    apt-get install -y dh-autoreconf libssl-dev autoconf automake

git clone https://github.com/openssl/openssl.git "$TARGET/openssl"
git -C "$TARGET/openssl" checkout OpenSSL_1_0_2-stable
cd "$TARGET/openssl"
mkdir install
./config --prefix="$TARGET/openssl/install"
make && make install
