from __future__ import annotations

import tensorflow as tf


def set_global_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Fixa as sementes de Python, NumPy e TensorFlow.

    Ative a execução determinística no início de um processo novo. Para repetir
    um experimento, use as mesmas versões de software e o mesmo hardware.
    """
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
