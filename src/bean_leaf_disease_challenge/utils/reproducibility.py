from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf

def set_global_seed(seed: int = 42) -> None:
    """Fixa seeds básicas para favorecer reprodutibilidade"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)