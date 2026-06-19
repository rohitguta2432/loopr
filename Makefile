.PHONY: install test optimize example clean

install:
	python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"

test:
	PYTHONPATH=. ./.venv/bin/python -m pytest -q

optimize:
	PYTHONPATH=. ./.venv/bin/python -m loopr.cli optimize loopr/tasks/extract_json.yaml

example:
	PYTHONPATH=. ./.venv/bin/python examples/quickstart.py

clean:
	rm -rf loopr-out __pycache__ */__pycache__ .pytest_cache *.egg-info
