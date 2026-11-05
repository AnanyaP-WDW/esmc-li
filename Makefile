PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PYTEST ?= $(PYTHON) -m pytest

.PHONY: tables paper claims verify-lock check-tables check-paper audit check wheel

tables:
	$(PYTHON) scripts/paper_assets.py
	$(PYTHON) scripts/build_paper.py

paper:
	$(PYTHON) scripts/build_paper.py

claims:
	$(PYTHON) scripts/build_claims.py

verify-lock:
	$(PYTHON) scripts/verify_lock.py

check-tables:
	$(PYTHON) scripts/paper_assets.py
	$(PYTHON) scripts/build_paper.py --check
	git diff --exit-code -- paper/tables paper/figures paper/numbers.json paper/main.md

check-paper:
	$(PYTHON) scripts/build_paper.py --check

audit: verify-lock tables claims
	$(PYTHON) scripts/check_claims.py
	$(PYTHON) scripts/check_citations.py
	git diff --exit-code -- paper/tables paper/figures paper/numbers.json paper/main.md docs/claims.md
	$(PYTEST) -q

check: audit

wheel:
	PYTHON=$(PYTHON) bash scripts/build_runtime_wheel.sh
