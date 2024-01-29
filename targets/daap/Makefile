.PHONY: daapd libevent

export CFLAGS := $(CFLAGS) -DSQLITE_CORE
LIBEVENT_PATH := $(realpath ./libevent_repo/build/lib/)
export LDFLAGS := $(LDFLAGS) -L$(LIBEVENT_PATH)
export LD_LIBRARY_PATH="$(LIBEVENT_PATH)"
export LDLIBS := $(LDLIBS)
export LIBS := $(LDLIBS)

daapd: preinstall libevent
	rm -rf daapd_repo
	git clone https://github.com/ejurgensen/forked-daapd.git daapd_repo
	cd daapd_repo && \
	git checkout 2ca10d9 && \
	patch -p1 < ../forked-daapd.patch && \
	autoreconf -i && \
	./configure \
		--disable-mpd \
		--disable-itunes \
		--disable-lastfm \
		--disable-spotify \
		--disable-verification \
		--disable-shared \
		--enable-static \
		--disable-webinterface \
		--without-libevent_pthreads \
		--without-libwebsockets \
		--without-alsa \
		--without-pulseaudio && \
	make -C src/ SMARTPL2SQL.c SMARTPL.c DAAP2SQL.c DAAPLexer.c RSPLexer.c RSP2SQL.c && \
	make all && \
	cp -r htdocs ../ && \
	cp src/forked-daapd ../daapd
	sudo chown tango:root forked-daapd.log
	sudo chown tango:root forked-daapd.pid
	sudo update-rc.d avahi-daemon defaults

preinstall:
	rm -rf openssl-1.1.1q* && \
	unset CFLAGS CXXFLAGS LDFLAGS LIBS LDLIBS && \
	./preinstall.sh

libevent:
	rm -rf libevent_repo
	git clone https://github.com/libevent/libevent.git libevent_repo
	cd libevent_repo && \
	git checkout 5df3037 && \
	mkdir build && cd build && \
	unset CFLAGS CXXFLAGS LDFLAGS LIBS LDLIBS && \
	cmake ../ \
		-DEVENT__HAVE_EPOLL_CREATE=OFF \
		-DEVENT__DISABLE_TESTS=ON \
		-DEVENT__DISABLE_SAMPLES=ON \
		-DEVENT__DISABLE_REGRESS=ON \
		-DEVENT__DISABLE_BENCHMARK=ON && \
	make
