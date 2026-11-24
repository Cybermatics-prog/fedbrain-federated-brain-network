# SPDX-License-Identifier: MIT
# Unified entry points. Every target delegates to scripts/pipeline.py, which is
# platform-neutral; override the interpreter with `make PYTHON=python3.12 ...`.
PYTHON ?= python
PIPELINE = $(PYTHON) scripts/pipeline.py

.PHONY: help setup data features demo test verify serve compliance pipeline clean-venv

help:
	@echo "Targets:"
	@echo "  make setup        create .venv and install requirements"
	@echo "  make data         download public ABIDE-I derivatives (~2 GB)"
	@echo "  make features     build the FC/BOLD cache used by the health views"
	@echo "  make demo         train a federation trace the interface replays"
	@echo "  make test         run the offline test suite"
	@echo "  make verify       compliance + English-only scan + tests + data checks"
	@echo "  make serve        start the API and interface on port 8765"
	@echo "  make pipeline     run data, features, demo, test and verify"

setup:
	$(PIPELINE) setup

data:
	$(PIPELINE) data

features:
	$(PIPELINE) features

demo:
	$(PIPELINE) demo

test:
	$(PIPELINE) test

verify:
	$(PIPELINE) verify

compliance:
	$(PYTHON) scripts/check_compliance.py

serve:
	$(PIPELINE) serve --port 8765

pipeline:
	$(PIPELINE) all

clean-venv:
	rm -rf .venv
