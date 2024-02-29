.PHONY: lightftp

export LDFLAGS := $(LDFLAGS) $(LDLIBS)
export LIBs := $(LIBs)

lightftp:
	./setup_ftpshare.sh
	rm -rf LightFTP fftp
	git clone https://github.com/hfiref0x/LightFTP.git
	cd LightFTP && \
	git checkout 5980ea1 && \
	patch -p1 < ../fuzzing.patch && \
	cd Source/Release && \
	patch makefile ../../../makefile.patch && \
	$(MAKE) clean && \
	$(MAKE) all && \
	cp fftp ../../../





