.PHONY: dtls

dtls:
	rm -rf dtls_repo
	git clone https://github.com/assist-project/tinydtls-fuzz.git dtls_repo
	cd dtls_repo && \
	git checkout 06995d4 && \
	patch -p1 < ../dtls.patch && \
	cd tests && \
	$(MAKE) clean && \
	$(MAKE) ../libtinydtls.a && \
	$(MAKE) && \
	cp dtls-server ../../dtls
