.PHONY: dcmtk

export LDFLAGS := $(LDFLAGS) $(LDLIBS)

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

dcmtk:
	rm -rf dcmtk_repo
	git clone https://github.com/DCMTK/dcmtk dcmtk_repo
	cd dcmtk_repo && \
	mkdir build && cd build && \
	cmake .. && \
	$(MAKE) dcmqrscp && \
	cp bin/dcmqrscp ../../
	mkdir -p ACME_STORE
	sed 's/%PATH_TO_ACME_STORE%/$(subst /,\/,$(mkfile_dir)/ACME_STORE)/g' dcmqrscp.cfg.template > dcmqrscp.cfg
