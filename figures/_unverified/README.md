# Quarantined figures — do not use in the paper

These figures are held here because **their provenance cannot be verified**, not
because they are known to be wrong.

## Why

They were produced by `scripts/plot_grouped_bar.py`, `scripts/plot_max_sensitivity.py`
and `scripts/plot_pareto_scatter.py`. Until this audit, all three scripts contained
the mapping:

```python
'hilrp': 'AttnLRP',      # removed
```

That mapping renders rows of the registry method `hilrp` — the novel method of a
**different paper** — under the display label of `attnlrp`, the published
Achtibat et al. (2024) baseline. Any figure generated from a results file that
contained `hilrp` rows would therefore show that separate paper's results
labelled as a benchmark baseline.

The scripts read `results/clean_summary.csv`, which is not present in the
repository, so it is not possible to determine after the fact whether these
particular renders drew on `hilrp` rows.

## What was done

* The `hilrp -> AttnLRP` mapping was removed from all three scripts.
* `attnlrp` now carries its own correct label.
* The canonical result set (`results/processed/records.json`) excludes every row
  originating from a `hilrp*` run; see `docs/scientific_integrity.md`.

## How to replace them

Regenerate equivalent figures from validated records:

```bash
python scripts/build_results.py
python scripts/generate_figures.py --out figures/paper
```

`figures/paper/` contains the regenerated, traceable versions. Once you have
confirmed the paper does not cite anything from this directory, it can be deleted.
