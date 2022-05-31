.PHONY: zoom

zoom:
# 	rm -rf openssl-repo
# 	git clone https://github.com/openssl/openssl.git openssl-repo
	cd repo && \
	$(MAKE) clean && \
	$(MAKE) LIBS='-lrt -lm' && \
	cp src/chocolate-doom ../
