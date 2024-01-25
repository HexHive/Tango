.PHONY: proftpd

export LDFLAGS := $(filter-out -fPIE,$(LDFLAGS))
export LDFLAGS := $(filter-out -pie,$(LDFLAGS))
export LDFLAGS := $(LDFLAGS) $(LDLIBS)

proftd:
	./setup_ftpshare.sh
	rm -rf proftpd_repo proftpd
	git clone https://github.com/proftpd/proftpd.git proftpd_repo && \
	cd proftpd_repo && \
	git checkout 4017eff8 && \
	patch -p1 < ../scoreboard_file.patch && \
	./configure --enable-devel=nodaemon:nofork && \
	$(MAKE) clean && \
	$(MAKE) all && \
	cp proftpd ../
