.PHONY: openssh

export LIBS := ${LDLIBS} -ldl

openssh:
	rm -rf openssh_repo
	git clone https://github.com/vegard/openssh-portable.git openssh_repo
	cd openssh_repo && \
	git checkout 7cfea58 && \
	cp ../rand.inc . && \
	patch -p1 < ../fuzz.patch && \
	autoreconf -f && \
	./configure \
		--with-ssl-dir="/home/openssl/install" \
		--with-sandbox=no \
		--with-privsep-path=/tmp/var-empty \
		--with-privsep-user=tango && \
	$(MAKE) all && \
	mkdir -p /tmp/var-empty && \
	cp ./sshd ../
