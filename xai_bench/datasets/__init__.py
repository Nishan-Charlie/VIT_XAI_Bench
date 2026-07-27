"""Dataset loaders for benchmark execution.

Importing this package registers the built-in datasets, which return
``(image, target_class, metadata)`` where ``metadata`` carries ground-truth
localisation information such as bounding boxes.
"""

from . import imagenet_s, imagenet_val  # noqa: F401  (imported for registration)
