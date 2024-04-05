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
RUN sudo update-alternatives \
    --install /usr/lib/llvm              llvm             /usr/lib/llvm-17 10 \
    --slave   /usr/bin/llvm-config       llvm-config      /usr/bin/llvm-config-17  \
    --slave   /usr/bin/llvm-ar           llvm-ar          /usr/bin/llvm-ar-17 \
    --slave   /usr/bin/llvm-as           llvm-as          /usr/bin/llvm-as-17 \
    --slave   /usr/bin/llvm-bcanalyzer   llvm-bcanalyzer  /usr/bin/llvm-bcanalyzer-17 \
    --slave   /usr/bin/llvm-c-test       llvm-c-test      /usr/bin/llvm-c-test-17 \
    --slave   /usr/bin/llvm-cov          llvm-cov         /usr/bin/llvm-cov-17 \
    --slave   /usr/bin/llvm-profdata     llvm-profdata    /usr/bin/llvm-profdata-17 \
    --slave   /usr/bin/llvm-diff         llvm-diff        /usr/bin/llvm-diff-17 \
    --slave   /usr/bin/llvm-dis          llvm-dis         /usr/bin/llvm-dis-17 \
    --slave   /usr/bin/llvm-dwarfdump    llvm-dwarfdump   /usr/bin/llvm-dwarfdump-17 \
    --slave   /usr/bin/llvm-extract      llvm-extract     /usr/bin/llvm-extract-17 \
    --slave   /usr/bin/llvm-link         llvm-link        /usr/bin/llvm-link-17 \
    --slave   /usr/bin/llvm-mc           llvm-mc          /usr/bin/llvm-mc-17 \
    --slave   /usr/bin/llvm-nm           llvm-nm          /usr/bin/llvm-nm-17 \
    --slave   /usr/bin/llvm-objdump      llvm-objdump     /usr/bin/llvm-objdump-17 \
    --slave   /usr/bin/llvm-ranlib       llvm-ranlib      /usr/bin/llvm-ranlib-17 \
    --slave   /usr/bin/llvm-readobj      llvm-readobj     /usr/bin/llvm-readobj-17 \
    --slave   /usr/bin/llvm-rtdyld       llvm-rtdyld      /usr/bin/llvm-rtdyld-17 \
    --slave   /usr/bin/llvm-size         llvm-size        /usr/bin/llvm-size-17 \
    --slave   /usr/bin/llvm-stress       llvm-stress      /usr/bin/llvm-stress-17 \
    --slave   /usr/bin/llvm-symbolizer   llvm-symbolizer  /usr/bin/llvm-symbolizer-17 \
    --slave   /usr/bin/llvm-tblgen       llvm-tblgen      /usr/bin/llvm-tblgen-17
RUN sudo update-alternatives \
  --install /usr/bin/clang                 clang                  /usr/bin/clang-17 10 \
  --slave   /usr/bin/clang++               clang++                /usr/bin/clang++-17 \
  --slave   /usr/bin/clang-cpp             clang-cpp              /usr/bin/clang-cpp-17

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
