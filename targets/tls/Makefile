.PHONY: openssl

openssl:
	rm -rf openssl-repo
	git clone https://github.com/openssl/openssl.git openssl-repo
	cd openssl-repo && \
	git checkout 0437435a960123be1ced766d18d715f939698345 && \
	patch -p1 < ../rand.patch && \
	./config no-shared --with-rand-seed=none && \
	$(MAKE) clean && \
	$(MAKE) include/openssl/configuration.h \
						 include/openssl/opensslv.h \
						 include/crypto/bn_conf.h \
						 include/crypto/dso_conf.h && \
	$(MAKE) apps/openssl && \
	cp apps/openssl ../
