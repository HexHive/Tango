FROM ubuntu:22.04

ENV TZ="Europe/Zurich"
RUN apt-get update && apt-get install -y git sudo software-properties-common

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN mkdir -p /home && \
	groupadd -g ${GROUP_ID} tango && \
	useradd -l -u ${USER_ID} -K UMASK=0000 -d /home -g tango tango && \
	chown tango:tango /home
RUN	echo "tango:ognat" | chpasswd && usermod -a -G sudo tango
RUN echo "tango ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

USER tango:tango
ENV PATH="/home/.local/bin:$PATH"

COPY git_install.sh /home/git_install.sh
RUN cd /home/ && ./git_install.sh -y -V -D
RUN sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10

WORKDIR /home

# Install dependencies for llhttp
RUN sudo apt-get install -y \
    npm
# Update npm for llhttp
RUN sudo npm cache clean -f && sudo npm install -g n && sudo n stable
# Install dependencies for daap
RUN sudo apt-get install -y \
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
RUN wget https://www.openssl.org/source/openssl-1.1.1q.tar.gz && \
    tar -xf openssl-1.1.1q.tar.gz && cd openssl-1.1.1q && \
    ./config shared zlib && make && sudo make install && sudo ldconfig
# Install dependencies for openssh
RUN sudo apt-get install -y dh-autoreconf libssl-dev autoconf automake
RUN git clone https://github.com/openssl/openssl.git openssl && \
    git -C openssl checkout OpenSSL_1_0_2-stable && cd openssl && \
    mkdir install && ./config --prefix="/home/openssl/install" && \
    make && make install
# Install dependencies for sip
RUN sudo apt-get install -y flex bison libmysqlclient-dev pkg-config
# Install dependencies for pureftpd
RUN sudo apt-get install -y automake
# Install dependencies for lightftp
RUN sudo apt-get install -y libgnutls28-dev
# Install dependencies for exim
RUN sudo apt-get install -y libpcre2-dev libdb-dev libpcre3-dev

WORKDIR /home/tango
ENTRYPOINT ["scripts/docker_entry.sh"]
