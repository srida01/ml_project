# Multi-Stage ML-Powered Intrusion Detection Engine

This repository implements the supplied CICIDS2017 research-prototype plan. It is deliberately **offline-first**: raw data is audited and cleaned before modeling, preprocessing is fitted on training data only, and the CLI, FastAPI service, and Streamlit dashboard share one inference implementation.

## Scope and honest limitations

The project accepts CICFlowMeter-style flow records. It does not capture packets, block IPs, or claim universal zero-day detection. An autoencoder score means “unusual relative to benign training traffic”; novelty claims require a withheld-family evaluation.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Unchanged CICIDS2017 CSV files are stored under data/raw/.
python scripts/00_inspect.py --input-dir data/raw
python scripts/01_clean.py --input-dir data/raw --output data/interim/cicids2017_cleaned.parquet
python scripts/02_split.py --input data/interim/cicids2017_cleaned.parquet
python scripts/03_fit_preprocessor.py
```

The downloaded archive and provenance checksums are recorded in `data/manifests/cicids2017_download.json`.

## One-command workflow

After installing `requirements.txt`, run the complete required workflow with:

```bash
python scripts/run_pipeline.py
```

This performs inspection, cleaning, leakage-resistant splitting, train-only preprocessing, baselines, SVM training, OOF stacking, autoencoder training, optional HMM training, fusion-policy export, and frozen binary test evaluation. The HMM stage is allowed to be unavailable when the local CICIDS2017 files do not contain valid sequence metadata; that limitation is reported rather than hidden. Use `--skip-autoencoder`, `--skip-hmm`, or `--skip-baselines` only for development runs.

The remaining training scripts are intentionally separate and rerunnable. They write versioned metadata under `artifacts/` and evaluation outputs under `reports/`.

For a small local smoke test without CICIDS2017, run:

```bash
python scripts/00_inspect.py --input tests/fixtures/sample_flows.csv --output data/manifests/sample_inspection.json
python scripts/01_clean.py --input-dir tests/fixtures --output data/interim/sample_cleaned.csv
python scripts/02_split.py --input data/interim/sample_cleaned.csv --output-dir data/processed
python scripts/03_fit_preprocessor.py --input data/processed/train.csv --output artifacts/preprocess/preprocessor.joblib
```

## Model stages

1. A high-recall binary SVM gate.
2. An out-of-fold RF/XGBoost/logistic stacking classifier for known attack families.
3. A benign-only PyTorch autoencoder for independent anomaly evidence.
4. An optional HMM temporal enrichment that only runs when valid ordered sequences are supplied.
5. Transparent rule-based fusion preserving every stage score.

The implementation includes graceful fallbacks for optional model libraries during lightweight development, but production experiments must record the actual model and environment used.

## Local services

After a trained inference bundle exists:

```bash
uvicorn ids.api.main:app --app-dir src --host 127.0.0.1 --port 8000
streamlit run dashboard/app.py
```

The API is local-only by default. See `/docs`, `/health`, and `/model-info`. Uploaded files are bounded and are not persisted by the service.

## Reproducibility rules

- Keep `data/raw/` immutable and record SHA-256 manifests.
- Split before fitting imputation, scaling, feature selection, or thresholds.
- Keep group/day splits disjoint where metadata permits.
- Train stacking meta-features from out-of-fold predictions only.
- Keep the final test set untouched until evaluation is frozen.
- Do not call an anomaly an unknown attack without withheld-family evidence.
