# Reproducible command interface for the Uganda crop-yield project.
#
#   make setup    create venv and install dependencies
#   make data     build all real-data panels (AAS 2018/2020 + climate)
#   make validate run data-quality and schema tests
#   make pca      run PCA analysis (retention, stability, loadings)
#   make models   run model comparison with grouped/temporal validation
#   make figures  regenerate all report figures
#   make report   generate summary report markdown
#   make all      data -> validate -> pca -> models -> figures -> report

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup data validate pca models figures report all clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/build_panels.py

validate:
	$(PYTHON) -m pytest tests -q

pca:
	$(PYTHON) scripts/run_pca.py

models:
	$(PYTHON) scripts/run_models.py

figures:
	$(PYTHON) scripts/make_figures.py

report:
	$(PYTHON) scripts/make_report.py

all: data validate pca models figures report

clean:
	rm -rf .pytest_cache .mypy_cache
