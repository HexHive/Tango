.PHONY: dnsmasq

export LIBS := $(LDLIBS)

dnsmasq:
	rm -rf dnsmasq_repo
	git clone git://thekelleys.org.uk/dnsmasq.git dnsmasq_repo
	cd dnsmasq_repo && \
    git checkout v2.73rc6 && \
	patch -p1 < ../dnsmasq.patch && \
	$(MAKE) clean && $(MAKE) all && \
	cp src/dnsmasq ../dnsmasq
