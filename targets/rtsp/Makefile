.PHONY: live555

export CFLAGS := $(CFLAGS) -DUSE_SYSTEM_RANDOM
export CXXFLAGS := $(CXXFLAGS) -DUSE_SYSTEM_RANDOM
export LDFLAGS := $(LDFLAGS) $(LDLIBS)

live555:
	rm -rf live555_repo
	git clone https://github.com/rgaufman/live555.git live555_repo
	cd live555_repo && \
	git checkout ceeb4f4 && \
	patch -p1 < ../fuzzing.patch && \
	./genMakefiles linux && \
	$(MAKE) clean && $(MAKE) all && \
	cp testProgs/testOnDemandRTSPServer ../live555
