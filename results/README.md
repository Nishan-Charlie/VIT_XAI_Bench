# Results

| Path | What it is | Tracked |
|---|---|---|
| `revision_grid_1000/` | The five-metric grid of the paper: all 1,000 images, every backbone and method, seeded, every image scored: `config.json`, `summary.csv`, `failures.json` and the per-image `results.csv`. | yes |
| `revision_grid_500/` | The same grid on the first 500 images, superseded by `revision_grid_1000/`, which extends exactly these evaluations; `prefill_1000.py` reads its shards. Same files. | yes |
| `revision_grid_seeded/`, `revision_grid_seeded_attnlrp/`, `revision_grid_seeded_50/` | The earlier grid on 100 (ResNet-50, ViT-B/16) or 50 images, superseded by `revision_grid_500/` and `revision_grid_1000/`, which extend exactly these evaluations; `prefill_500.py` reads it. Same files: `config.json`, `summary.csv`, `failures.json` and the per-image `results.csv`. | yes |
| `revision_neucom/` | The revision analyses (localisation and EBPG, multi-seed robustness, Grad-CAM++ diagnostics, accuracy, timing), per image, and their tables in `analysis/`. | yes |
| `revision_vit_native/`, `revision_gradshap_fill/` | Earlier revision runs, superseded by the seeded grid; kept for the response to the reviewers. | yes |
| `raw/computational_cost.csv` | Mean time per explanation per method (the paper's cost table). | yes |
| `raw/benchmark_runs.csv` | The July 2026 summaries: unseeded, undefined images left out, mean Max-Sensitivity. Superseded; kept for the record. | yes |
| `processed/records.json` | Canonical, validated `(model, method)` records. Figures and the website read this. | yes |
| `processed/manifest.json` | Coverage matrix, provenance gaps, validation report. | yes |
| `<run_name>/` | Output of any other run. | no (gitignored) |

Rebuild `processed/` from the grid runs and the EBPG analysis:

```bash
python scripts/build_results.py
python scripts/validate_results.py
```

Rows belonging to other papers are excluded at build time; see
`docs/scientific_integrity.md`.
