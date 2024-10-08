sudo apt-get update && \
sudo apt-get install -y \
    apt-utils \
    libgnutls28-dev \
    net-tools \
    bison \
    flex \
    autotools-dev autoconf automake libtool gettext gawk \
    gperf antlr3 libantlr3c-dev libconfuse-dev libunistring-dev libsqlite3-dev \
    libavcodec-dev libavformat-dev libavfilter-dev libswscale-dev libavutil-dev \
    libasound2-dev libmxml-dev libgcrypt20-dev libavahi-client-dev zlib1g-dev \
    libevent-dev libplist-dev libsodium-dev libjson-c-dev libwebsockets-dev \
    libcurl4-openssl-dev avahi-daemon cmake libpth-dev
wget https://www.openssl.org/source/openssl-1.1.1q.tar.gz && \
    tar -xf openssl-1.1.1q.tar.gz && cd openssl-1.1.1q && \
    ./config shared zlib && make && sudo make install && sudo ldconfig
