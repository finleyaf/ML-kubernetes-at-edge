# Phase 2: Data Quality Check

This phase validates collected `run_*` datasets before model training.

## Checks performed per run

- Required files and schema (`labelled.csv` and expected columns)
- Minimum row count threshold
- Null-value detection in required fields
- Expected node presence and node-sampling balance
- Label sanity:
  - control runs must contain zero anomalies
  - stress runs should meet a minimum anomaly ratio
- Timestamp coverage and dropped-sample heuristic

## Run quality check

```bash
cd anomaly-detection/online-telemetry
source ../../.venv/bin/activate

python experiments/phase2/data_quality_check.py \
  --runs-dir dataset/runs \
  --output-json experiments/results/quality_report.json \
  --output-csv experiments/results/quality_report.csv \
  --min-rows 250 \
  --min-anomaly-ratio 0.05
```

## Outputs

- `experiments/results/quality_report.json`: full per-run report + aggregate summary
- `experiments/results/quality_report.csv`: compact table for spreadsheets/reporting

## Interpreting statuses

- `pass`: run meets all checks
- `warn`: run usable but has non-critical issues (e.g., low anomaly ratio in stress run)
- `fail`: run should be excluded until corrected or recollected
