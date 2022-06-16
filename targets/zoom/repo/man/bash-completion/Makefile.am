bashcompletiondir=@bashcompletiondir@

BASH_COMPLETION_TEMPLATES = \
    doom.template

if HAVE_PYTHON

BASH_COMPLETION_SCRIPTLETS = \
    @PROGRAM_PREFIX@doom

bashcompletion_DATA = $(BASH_COMPLETION_SCRIPTLETS)
CLEANFILES = $(BASH_COMPLETION_SCRIPTLETS)

MANDIR = $(top_srcdir)/man
DOCGEN = $(MANDIR)/docgen

@PROGRAM_PREFIX@doom: $(top_srcdir)/src $(DOCGEN) $(BASH_COMPLETION_TEMPLATES)
	$(DOCGEN) -g doom -b $(MANDIR)/doom.template \
                  $(top_srcdir)/src $(top_srcdir)/src/doom > $@

EXTRA_DIST = \
    $(BASH_COMPLETION_TEMPLATES) \
    $(BASH_COMPLETION_SCRIPTLETS)

else

EXTRA_DIST = \
    $(BASH_COMPLETION_TEMPLATES)

endif
