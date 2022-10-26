.PHONY: exim

exim:
	rm -rf exim_repo
	git clone https://github.com/Exim/exim.git exim_repo
	cd exim_repo && \
	git checkout 38903fb && \
	patch -p1 < ../exim-rand.patch && \
	patch -p1 < ../exim-log-bug.patch && \
	patch -p1 < ../exim-Makefile.patch && \
	patch -p1 < ../exim-buildconfig.patch && \
	mkdir -p src/Local && \
	cp ../EDITME src/Local/Makefile && \
	sed -i 's/^EXIM_USER=/EXIM_USER=ref:${USER}/' src/Local/Makefile && \
	echo 'CC=${CC}' >> src/Local/Makefile && \
	echo 'CFLAGS+=${CFLAGS}' >> src/Local/Makefile && \
	echo 'LFLAGS+=${CFLAGS}' >> src/Local/Makefile && \
	echo 'LDFLAGS+=${LDFLAGS} ${LDLIBS}' >> src/Local/Makefile && \
	cd src && \
	$(MAKE) clean && \
	$(MAKE) exim && \
	cp ./build-Linux-x86_64/exim ../../
