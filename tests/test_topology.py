"""WP3 topology on toy shapes and the two controls."""
import numpy as np
import yaml

from phinn_sw import topology as T

TC = yaml.safe_load(open("config.yaml"))["topology"]
GRID = np.arange(0, 61, 1.0)


def test_circle_has_one_loop_at_mid_scale():
    a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    pts = np.c_[20 * np.cos(a), 20 * np.sin(a)]          # radius 20 nT
    b0, b1 = T.betti_curves(pts, GRID)
    assert b0[10] == 1 and b1[10] == 1 and b1[59] == 0    # loop alive at 10 nT, filled by 59 nT


def test_two_clusters_are_two_components():
    pts = np.r_[np.zeros((10, 3)), np.full((10, 3), 40.0)]
    b0, _ = T.betti_curves(pts, GRID)
    assert b0[15] == 2 and b0[60] == 2                     # 40*sqrt(3) = 69 nT apart: separate on the whole grid


def test_quiet_and_storm_fingerprints_differ():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 3, 24)
    storm = np.r_[np.zeros(8), np.linspace(0, -300, 6), np.linspace(-300, -150, 10)]
    q0, _ = T.betti_curves(T.delay_embed(quiet, 3, 3), GRID)
    s0, _ = T.betti_curves(T.delay_embed(storm, 3, 3), GRID)
    assert q0[15] <= 2 < s0[15]                            # a storm fragments the cloud at 15 nT


def test_shuffle_keeps_levels_and_iaaft_keeps_amplitudes():
    rng = np.random.default_rng(1)
    x = np.r_[np.zeros(72), np.linspace(0, -250, 10), np.linspace(-250, -20, 230)] + rng.normal(0, 3, 312)
    xs = rng.permutation(x)
    assert all(np.isclose(T.level_stats(x)[k], T.level_stats(xs)[k]) for k in T.level_stats(x))
    y = T.iaaft(x, rng, n_iter=50)
    assert np.allclose(np.sort(y), np.sort(x))            # same amplitude distribution
    assert T.curve_distance(T.storm_features(x, TC), T.storm_features(xs, TC)) > 0
