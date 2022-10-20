.PHONY: openssh

export LIBS := ${LDLIBS} -ldl

openssh:
	rm -rf openssh_repo
	git clone https://github.com/vegard/openssh-portable.git openssh_repo
	cd openssh_repo && \
	git checkout 7cfea58 && \
	patch -p1 < ../rand.patch && \
	patch -p1 < ../configure.patch && \
	cp ../rand.inc . && \
	autoreconf -f && \
	./configure \
		--with-sandbox=no \
		--with-privsep-path=${PWD}/var-empty \
		--with-privsep-user=${USER} && \
	$(MAKE) clean && \
	$(MAKE) && \
	cp ./sshd ../
