# Reproducible command interface for the Uganda crop-yield project.
#
#   make setup    create venv and install dependencies
#   make data     build all real-data panels (AAS 2018/2020 + climate)
#   make data-final  build the validated subregion-season-year datasets (maize + multi-crop)
#   make validate run data-quality and schema tests
#   make pca      run PCA analysis (retention, stability, loadings)
#   make models   run model comparison with grouped/temporal validation
#   make figures  regenerate all report figures
#   make report   generate summary report markdown
#   make all      data -> validate -> pca -> models -> figures -> report

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup data data-final models-final models-final-quick validate pca models diagnostics figures report all clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/build_panels.py

data-final:
	$(PYTHON) scripts/build_final_dataset.py

models-final:
	$(PYTHON) scripts/run_final_analysis.py --mode spatial

models-final-quick:
	$(PYTHON) scripts/run_final_analysis.py --mode spatial --quick

validate:
	$(PYTHON) -m pytest tests -q

pca:
	$(PYTHON) scripts/run_pca.py

models:
	$(PYTHON) scripts/run_models.py

diagnostics:
	$(PYTHON) scripts/run_diagnostics.py

figures:
	$(PYTHON) scripts/make_figures.py

report:
	$(PYTHON) scripts/make_report.py

all: data validate pca models figures report

clean:
	rm -rf .pytest_cache .mypy_cache
