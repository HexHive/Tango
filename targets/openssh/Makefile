.PHONY: openssh

export LIBS := ${LDLIBS} -ldl

openssh:
	rm -rf openssh_repo
	git clone https://github.com/openssh/openssh-portable.git openssh_repo
	cd openssh_repo && \
	cp ../rand.inc . && \
	patch -p1 < ../fuzz.patch && \
	autoreconf -f && \
	./configure \
		--with-sandbox=no \
		--with-zlib=no \
		--with-pie=yes && \
	$(MAKE) clean && \
	$(MAKE) && \
	cp ./sshd ../
