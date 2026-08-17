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
#   make final-evidence  regenerate AND stage the authoritative multi-crop
#                 evidence pack (full model matrix, never --quick)
#   make all      data -> validate -> pca -> models -> figures -> report

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup data data-final models-final models-final-quick final-evidence validate pca models diagnostics figures report all clean

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

# Regenerate and stage the authoritative multi-crop evidence pack.
#
# Rules this target enforces (see docs/final_evidence.md):
#   1. Always the FULL registry (dummy_mean, ols, ridge, pls, random_forest,
#      xgboost) -- never --quick. Quick runs are smoke tests and must never
#      be committed as headline evidence.
#   2. Always the multi-crop dataset (373 rows), the only sample that passes
#      the 100-row / 5-unit quality gate.
#   3. The curated outputs are staged so the committed evidence always
#      matches the code that produced it (manifest.generated_at must
#      postdate the last change to src/uganda_crop_model/ or this script).
final-evidence:
	@test -f data/processed/final_multi_crop_subregion_season_year.csv || \
		(echo "final multi-crop dataset missing; run make data-final first" && exit 1)
	$(PYTHON) scripts/run_final_analysis.py --dataset multi_crop --mode spatial
	git add reports/tables/multi_crop_spatial_model_comparison.csv \
		reports/tables/multi_crop_analysis_manifest.json \
		reports/tables/multi_crop_spatial_conformal_coverage.csv \
		reports/tables/multi_crop_spatial_execution_status.csv \
		reports/tables/multi_crop_missingness_report.csv \
		reports/tables/multi_crop_pca_diagnostics.csv \
		reports/tables/multi_crop_model_agreement.csv \
		reports/tables/multi_crop_outlier_flags.csv \
		reports/tables/multi_crop_residual_diagnostics.csv \
		reports/tables/multi_crop_heldout_permutation_importance.csv
	git add -f reports/tables/multi_crop_spatial_fold_results.csv
	@echo "Evidence pack staged. Review with 'git status', then commit."
	@echo "Check the log for 'checkpoint skipped' -- a skipped model means"
	@echo "a missing optional dependency (install requirements.lock),"
	@echo "not a model that should silently vanish from the registry."

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
