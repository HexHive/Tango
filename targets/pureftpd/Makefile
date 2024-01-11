.PHONY: pureftpd

export LDFLAGS := $(LDFLAGS) $(LDLIBS)

pureftpd:
	./preinstall.sh
	./setup_ftpshare.sh
	rm -rf pureftpd_repo pure-ftpd 
	git clone https://github.com/jedisct1/pure-ftpd.git pureftpd_repo && \
	cd pureftpd_repo && \
	git checkout c21b45f && \
	patch -p1 < ../fuzzing.patch && \
	./autogen.sh && \
	./configure --without-privsep -without-capabilities && \
	$(MAKE) clean && \
	$(MAKE) all && \
	cp src/pure-ftpd ../
