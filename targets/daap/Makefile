.PHONY: daapd libevent_noepoll

export CFLAGS := $(CFLAGS) -DSQLITE_CORE
LIBEVENT_PATH := $(realpath ./libevent_repo/build/lib/)
LIBPTH_PATH := $(realpath ./libpth_repo/.libs/)
export LDFLAGS := $(LDFLAGS) -L$(LIBEVENT_PATH) #-L$(LIBPTH_PATH)
export LDLIBS := $(LDLIBS) -lpth
export LIBS := $(LDLIBS)

daapd: libevent_noepoll #pth
	rm -rf daapd_repo
	git clone https://github.com/ejurgensen/forked-daapd.git daapd_repo
	cd daapd_repo && \
	git checkout 2ca10d9 && \
	patch -p1 < ../forked-daapd.patch && \
	autoreconf -i && \
	#LD_LIBRARY_PATH="$(LIBEVENT_PATH):$(LIBPTH_PATH)" ./configure \
	LD_LIBRARY_PATH="$(LIBEVENT_PATH)" ./configure \
		--disable-mpd \
		--disable-itunes \
		--disable-lastfm \
		--disable-spotify \
		--disable-verification \
		--disable-shared \
		--enable-static \
		--disable-webinterface \
		--without-libevent_pthreads \
		--without-libwebsockets && \
	$(MAKE) -C src/ SMARTPL2SQL.c SMARTPL.c DAAP2SQL.c DAAPLexer.c RSPLexer.c RSP2SQL.c && \
	$(MAKE) clean all && \
	cp -r htdocs ../ && \
	cp src/forked-daapd ../daapd

libevent_noepoll:
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
		-DEVENT__DISABLE_BENCHMARK=ON \
		-DEVENT__DISABLE_MBEDTLS=ON \
		-DEVENT__DISABLE_THREAD_SUPPORT=ON && \
	$(MAKE)

pth:
	rm -rf libpth_repo
	git clone https://github.com/danluu/gnu-pth.git libpth_repo
	cd libpth_repo && \
	git checkout 5e514e1 && \
	patch -p1 < ../libpth.patch && \
	unset CFLAGS CXXFLAGS LDFLAGS LIBS LDLIBS && \
	./configure --enable-pthread && \
	make