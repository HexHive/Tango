.PHONY: openssl

openssl:
	rm -rf openssl_repo
	git clone https://github.com/openssl/openssl.git openssl_repo
	cd openssl_repo && \
	git checkout 0437435a && \
	cp ../rand.inc . && \
	patch -p1 < ../rand.patch && \
	./config no-shared --with-rand-seed=none && \
	$(MAKE) build_generated && \
	$(MAKE) apps/openssl && \
	cp apps/openssl ../
