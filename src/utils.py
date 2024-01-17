import tensorflow as tf
import os
import numpy as np
import random

SEED = 0


def set_seeds(seed=SEED):
    """
    Sets the seeds for reproducibility
    @param seed: The seed to use
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    """
    Sets global determinism for reproducibility
    @param seed: The seed to use
    """
    set_seeds(seed=seed)

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
