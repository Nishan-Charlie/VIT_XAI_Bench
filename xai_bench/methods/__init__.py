"""Attribution methods.

Importing this package registers every built-in method. Each conforms to the
same ``__call__(inputs, target)`` interface regardless of backend (Captum, CAM,
attention-native, or LRP).

``attnlrp_method`` is the published AttnLRP baseline (Achtibat et al., 2024).
It applies class-level monkey patches to timm and must be run in a dedicated
process; see its module docstring.
"""

from . import (  # noqa: F401  (imported for registration)
    attention_rollout,
    attnlrp_method,
    cam_methods,
    captum_methods,
    perturbation_methods,
)
