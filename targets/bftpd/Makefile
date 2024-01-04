.PHONY: bftpd

export LDFLAGS := $(LDFLAGS) $(LDLIBS)
export CFLAGS := $(CFLAGS) -fcommon

bftpd:
	./setup_ftpshare.sh
	rm -rf bftpd_repo bftpd-5.7.tar.gz bftpd
	wget https://phoenixnap.dl.sourceforge.net/project/bftpd/bftpd/bftpd-5.7/bftpd-5.7.tar.gz
	tar -zxvf bftpd-5.7.tar.gz
	mv bftpd bftpd_repo && \
	cd bftpd_repo && \
	patch -p1 < ../fuzzing.patch && \
	patch -p1 < ../Makefile.patch && \
	$(MAKE) clean && \
	$(MAKE) all && \
	cp bftpd ../
