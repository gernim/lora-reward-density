from __future__ import annotations

import contextlib
import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and (if installed) torch RNGs.

    Pass ``deterministic_torch=False`` only in tests where importing torch is
    undesirable; experiment runs should leave it on.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if not deterministic_torch:
        return

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required for deterministic cuBLAS GEMMs on Ampere+.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with contextlib.suppress(AttributeError, RuntimeError):
        torch.use_deterministic_algorithms(True, warn_only=True)
