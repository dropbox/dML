# Makefile for MPS Thread-Safety Paper

PAPER = mps_thread_safety
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).pdf $(PAPER).toc $(PAPER).lof \
	      $(PAPER).lot $(PAPER).fls $(PAPER).fdb_latexmk

view: $(PAPER).pdf
	open $(PAPER).pdf  # macOS

# Word count (approximate)
wc:
	@texcount -inc -brief $(PAPER).tex

# Check for common issues
check:
	@echo "Checking for common issues..."
	@grep -n "TODO" $(PAPER).tex || echo "No TODOs found"
	@grep -n "FIXME" $(PAPER).tex || echo "No FIXMEs found"
	@grep -n "XXX" $(PAPER).tex || echo "No XXXs found"
