from __future__ import annotations

import os
import random

import numpy as np

from lora_reward_density.seeding import set_seed


def test_python_random_is_deterministic_after_set_seed():
    set_seed(42, deterministic_torch=False)
    a = [random.random() for _ in range(5)]
    set_seed(42, deterministic_torch=False)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_random_is_deterministic_after_set_seed():
    set_seed(42, deterministic_torch=False)
    a = np.random.rand(5).tolist()
    set_seed(42, deterministic_torch=False)
    b = np.random.rand(5).tolist()
    assert a == b


def test_set_seed_writes_pythonhashseed():
    set_seed(123, deterministic_torch=False)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_different_seeds_produce_different_streams():
    set_seed(1, deterministic_torch=False)
    a = np.random.rand(5).tolist()
    set_seed(2, deterministic_torch=False)
    b = np.random.rand(5).tolist()
    assert a != b
