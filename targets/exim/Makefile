.PHONY: exim

exim:
	rm -rf exim_repo
	git clone https://github.com/Exim/exim.git exim_repo
	cd exim_repo && \
	patch -p1 < ../editme.patch && \
	sed -i 's/^EXIM_USER=/EXIM_USER=ref:${USER}/' src/Local/Makefile && \
	echo 'CC=${CC}' >> src/Local/Makefile && \
	echo 'CFLAGS+=${CFLAGS}' >> src/Local/Makefile && \
	echo 'LFLAGS+=${CFLAGS}' >> src/Local/Makefile && \
	echo 'LDFLAGS+=${LDFLAGS} ${LDLIBS}' >> src/Local/Makefile && \
	cd src && \
	$(MAKE) makefile && \
	$(MAKE) exim && \
	cp ./build-Linux-x86_64/exim ../../
