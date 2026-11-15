import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agentic"))
sys.path.insert(0, os.path.join(ROOT, "demo"))

import stage_b_spikes_to_cursor as sb  # noqa: E402
from eval_falcon_h1 import gauss_kl  # noqa: E402


def test_r2_perfect_and_mean_prediction():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(500, 2))
    assert sb.r2(y, y) == 1.0
    assert abs(sb.r2(y, np.tile(y.mean(0), (500, 1)))) < 1e-12


def test_gauss_kl_zero_for_same_data_and_symmetric():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2000, 2))
    b = rng.normal(loc=1.0, size=(2000, 2))
    assert abs(gauss_kl(a, a)) < 1e-9
    assert gauss_kl(a, b) > 0.5
    assert abs(gauss_kl(a, b) - gauss_kl(b, a)) < 1e-9


def test_gauss_kl_grows_with_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(2000, 2))
    near = gauss_kl(a, a + 0.5)
    far = gauss_kl(a, a + 2.0)
    assert 0 < near < far
