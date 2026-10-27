# Readable entry points into the preserved reproducibility package.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHON_EXEC := $(abspath $(shell command -v $(PYTHON)))
REPRO := reproducibility

.PHONY: help install verify test demo plans analysis check

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make install   Install pinned CPU dependencies (create .venv first)' \
	  '  make verify    Check package integrity and saved BCT2 receipts (stdlib only)' \
	  '  make test      Run CPU unit tests (requires requirements-cpu.txt)' \
	  '  make demo      Show BDT transformation on a small synthetic tensor' \
	  '  make check     Run verification, CPU tests, and the demo' \
	  '  make plans     Show offline native-study reproduction plans' \
	  '  make analysis  Recompute saved results into a new build/all_records directory'

install:
	"$(PYTHON_EXEC)" -m pip install -r $(REPRO)/requirements-cpu.txt

check: verify test demo

verify:
	cd $(REPRO) && "$(PYTHON_EXEC)" scripts/check_package.py
	cd $(REPRO) && "$(PYTHON_EXEC)" scripts/verify_bct2.py

test:
	cd $(REPRO) && "$(PYTHON_EXEC)" -m unittest discover -s tests -q

demo:
	"$(PYTHON_EXEC)" examples/depth_update_demo.py

plans:
	cd $(REPRO) && "$(PYTHON_EXEC)" scripts/native_study.py --study cms1 --action plan
	cd $(REPRO) && "$(PYTHON_EXEC)" scripts/native_study.py --study gct2 --action plan
	cd $(REPRO) && "$(PYTHON_EXEC)" scripts/native_study.py --study bac1 --action plan

analysis:
	cd $(REPRO) && "$(PYTHON_EXEC)" scripts/reproduce_all.py --out build/all_records
