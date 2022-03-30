.PHONY: dcmtk

export LDFLAGS := $(LDFLAGS) $(LDLIBS)

dcmtk:
	rm -rf dcmtk
	git clone https://github.com/DCMTK/dcmtk
	cd dcmtk && \
	git checkout 7f8564c && \
	patch -p1 < ../fuzzing.patch && \
	mkdir build && cd build && \
	cmake .. && \
	$(MAKE) dcmqrscp && \
	cp bin/dcmqrscp ../../
	mkdir -p ACME_STORE
	sed 's/%PATH_TO_ACME_STORE%/$(subst /,\/,$(PWD)/dicom/ACME_STORE)/g' dcmqrscp.cfg.template > dcmqrscp.cfg