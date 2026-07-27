"""Evaluation metrics.

Importing this package registers every built-in metric across the benchmark's
dimensions: faithfulness, localisation, robustness and complexity.
"""

from . import (  # noqa: F401  (imported for registration)
    axiomatic,
    faithfulness,
    localization,
    randomisation,
    robustness,
    sparseness,
)
