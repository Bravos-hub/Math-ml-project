# Contributing

This is an undergraduate research project. The guiding rules:

1. **Never mix data classes in one modeling table.** Observed, assigned,
   proxy, and synthetic rows must live in separate files with explicit
   provenance columns (`yield_source`, `data_quality_score`, ...).
2. **State the source and granularity on every figure and table.**
3. **Never include the target (yield) in PCA features**, and never include
   failed-extraction (grade F) columns in PCA or modeling.
4. **Use grouped and temporal validation**, never only random CV, because
   district-year rows are spatially and temporally correlated.
5. **Compare against simple baselines** (mean, historical mean, previous-year
   yield, rainfall-only OLS) before reporting a complex model.
6. **Register every experiment** in `reports/experiments.csv` with its
   `data_version`, feature set, validation scheme, and random seed.
7. Raw downloads and API responses are cached under `data/raw/` and
   referenced by the pipeline manifest, never re-fetched silently.

## Workflow

- `make setup` once; `make validate` before and after changes.
- Add tests in `tests/` for any new data table or validation scheme.
- Run `make data && make validate` after touching data code.
- Keep `CHANGELOG.md` up to date.
