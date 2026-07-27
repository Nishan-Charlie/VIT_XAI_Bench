# Results

| Path | What it is | Tracked |
|---|---|---|
| `raw/benchmark_runs.csv` | Aggregated per-run summaries, the input to the pipeline. | yes |
| `processed/records.json` | Canonical, validated `(model, method)` records. Figures and the website read this. | yes |
| `processed/manifest.json` | Coverage matrix, provenance gaps, validation report. | yes |
| `<run_name>/` | Raw output of an individual benchmark run. | no (gitignored) |

Rebuild `processed/` from `raw/`:

```bash
python scripts/build_results.py
python scripts/validate_results.py
```

Rows belonging to other papers are excluded at build time; see
`docs/scientific_integrity.md`.
