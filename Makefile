PYTHON ?= python3
PYTHONPATH := src

.PHONY: check stress-fixture

check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest -v \
		tests.test_metrics \
		tests.test_safety_rules \
		tests.test_stress_fixture \
		tests.test_gguf_runner \
		tests.test_render_latex_tables

baseline-reference:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m redaction_research.cli eval-fixture --fixture data/fixtures/local-vault-stress-set-v1.jsonl --mode regex --output runs/regex-stress.json

stress-fixture:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_stress_fixture.py --output data/fixtures/local-vault-stress-set-v1.jsonl
