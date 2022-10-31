.PHONY: daapd libevent_noepoll

export CFLAGS := $(CFLAGS) -DSQLITE_CORE
export LDFLAGS := $(LDFLAGS) -L$(realpath ./libevent_repo/build/lib/)
export LIBS := $(LDLIBS)

daapd: libevent_noepoll
	rm -rf daapd_repo
	git clone https://github.com/ejurgensen/forked-daapd.git daapd_repo
	cd daapd_repo && \
	git checkout 2ca10d9 && \
	patch -p1 < ../forked-daapd.patch && \
	autoreconf -i && \
	./configure --disable-mpd --disable-itunes --disable-lastfm --disable-spotify --disable-verification  --disable-shared --enable-static --without-libevent_pthreads && \
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