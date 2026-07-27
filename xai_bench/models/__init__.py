"""Model registry.

Importing this package registers the benchmark backbones. Each is wrapped so
attribution methods can discover the metadata they need (feature layout for
Grad-CAM, whether global CLS-token attention exists, input normalisation).
"""

from . import timm_models  # noqa: F401  (imported for registration)
