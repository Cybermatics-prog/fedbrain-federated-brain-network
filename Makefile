# SPDX-License-Identifier: MIT
# Unified entry points. Every target delegates to scripts/reproduce.py, which is
# platform-neutral; override the interpreter with `make PYTHON=python3.12 ...`.
PYTHON ?= python
REPRO = $(PYTHON) scripts/reproduce.py

.PHONY: help setup data features demo test verify serve compliance reproduce clean-venv

help:
	@echo "Targets:"
	@echo "  make setup        create .venv and install requirements"
	@echo "  make data         download public ABIDE-I derivatives (~2 GB)"
	@echo "  make features     build the FC/BOLD cache used by the health views"
	@echo "  make demo         train a federation trace the interface replays"
	@echo "  make test         run the offline test suite"
	@echo "  make verify       compliance + English-only scan + tests + data checks"
	@echo "  make serve        start the API and interface on port 8765"
	@echo "  make reproduce    run data, features, demo, test and verify"

setup:
	$(REPRO) setup

data:
	$(REPRO) data

features:
	$(REPRO) features

demo:
	$(REPRO) demo

test:
	$(REPRO) test

verify:
	$(REPRO) verify

compliance:
	$(PYTHON) scripts/check_compliance.py

serve:
	$(REPRO) serve --port 8765

reproduce:
	$(REPRO) all

clean-venv:
	rm -rf .venv
