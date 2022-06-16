SUBDIRS = bash-completion

MANPAGE_GEN_FILES = environ.man           \
                    iwad_paths.man        \
                    doom.template         \
                    docgen                \
                    default.cfg.template  \
                    extra.cfg.template

doomdocsdir = ${docdir}/../${PROGRAM_PREFIX}doom

if HAVE_PYTHON

GENERATED_MAN_PAGES =               \
           @PROGRAM_PREFIX@doom.6         \
           default.cfg.5            \
           @PROGRAM_PREFIX@doom.cfg.5

SETUP_MAN_PAGES =                          \
           @PROGRAM_PREFIX@doom-setup.6

man_MANS = chocolate-server.6             \
           chocolate-setup.6              \
           $(GENERATED_MAN_PAGES)         \
           $(SETUP_MAN_PAGES)

doomdocs_DATA    = INSTALL.doom    CMDLINE.doom

CLEANFILES = $(GENERATED_MAN_PAGES) $(SETUP_MAN_PAGES) \
             $(doomdocs_DATA)
DOCGEN = $(srcdir)/docgen

$(SETUP_MAN_PAGES): chocolate-setup.6
	cp $(srcdir)/chocolate-setup.6 $@

@PROGRAM_PREFIX@doom.6: $(top_srcdir)/src $(MANPAGE_GEN_FILES)
	$(DOCGEN) -g doom -m $(srcdir)/doom.template \
                  $(top_srcdir)/src $(top_srcdir)/src/doom > $@

default.cfg.5: $(top_srcdir)/src $(srcdir)/default.cfg.template
	$(DOCGEN) -g doom -m $(srcdir)/default.cfg.template \
                 -c default $(top_srcdir)/src/m_config.c > $@

@PROGRAM_PREFIX@doom.cfg.5: $(top_srcdir)/src extra.cfg.template
	$(DOCGEN) -g doom -m $(srcdir)/extra.cfg.template \
                 -c extended $(top_srcdir)/src/m_config.c > $@

CMDLINE.doom : CMDLINE.template $(top_srcdir)/src $(top_srcdir)/src/doom
	$(DOCGEN) -p $(srcdir)/CMDLINE.template \
                  $(top_srcdir)/src/ $(top_srcdir)/src/doom/ > $@

INSTALL.doom: INSTALL.template
	$(srcdir)/simplecpp -DDOOM -DPRECOMPILED < $(srcdir)/INSTALL.template > $@


INSTALL: INSTALL.template
	$(srcdir)//simplecpp -DDOOM \
	            -DPRECOMPILED < $(srcdir)/INSTALL.template > $@

endif

EXTRA_DIST = $(man_MANS) $(MANPAGE_GEN_FILES)        \
             wikipages                               \
             CMDLINE.template                        \
             INSTALL.template                        \
             simplecpp

