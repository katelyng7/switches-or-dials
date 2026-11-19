"""Parameter recovery at the real design size (120 prompts, 8 samples)."""
import numpy as np
import pytest

from analysis import greedy, hierfit, q2_groups

A = [0.0, 0.06, 0.09, 0.13, 0.19, 0.28, 0.42, 0.62, 0.92, 1.36, 2.0]


@pytest.mark.parametrize("sigma_t,w", [(0.8, 0.15), (0.15, 0.8), (0.5, 0.5)])
def test_recovers_S(sigma_t, w):
    true_S = sigma_t**2 / (sigma_t**2 + w**2)
    fits = [hierfit.fit(A, *hierfit.simulate(A, 120, 8, np.log(0.4), sigma_t, w, g=0.05, seed=s)) for s in range(8)]
    assert abs(np.mean([f["S"] for f in fits]) - true_S) < 0.05
    assert abs(np.mean([f["mu"] for f in fits]) - np.log(0.4)) < 0.1
    assert abs(np.mean([f["g"] for f in fits]) - 0.05) < 0.02


def test_population_curve_cannot_tell_them_apart():
    """Sanity check on the premise: switch and dial with equal w^2 + sigma_t^2
    give the same pooled curve but very different S."""
    ks = [hierfit.simulate(A, 4000, 8, np.log(0.4), s, w, seed=1)[0].mean(0) / 8 for s, w in [(0.8, 0.15), (0.15, 0.8)]]
    assert np.abs(ks[0] - ks[1]).max() < 0.02


def test_greedy_sigma():
    rng = np.random.default_rng(0)
    log_t = rng.normal(np.log(0.4), 0.6, 2000)
    a = np.array(A[1:])
    flips = [next((x for x in a if np.log(x) >= t), None) for t in log_t]
    est = greedy.fit_flips(a, flips)
    assert abs(est["sigma_t"] - 0.6) < 0.06 and abs(est["mu"] - np.log(0.4)) < 0.06


def test_permutation():
    S = dict(a=0.9, b=0.8, c=0.85, d=0.95, e=0.88, f=0.1, g=0.2, h=0.15, i=0.05, j=0.12)
    groups = {k: ("decision" if k in "abcde" else "style") for k in S}
    r = q2_groups.permutation_test(S, groups)
    assert r["n_splits"] == 252 and r["p_one_sided"] == pytest.approx(1 / 252)


def test_bimodality_coefficient():
    rng = np.random.default_rng(0)
    rows = lambda vals: [dict(a=0.5, sample=i, scores=dict(t=float(v))) for i, v in enumerate(vals)]
    two = np.concatenate([rng.normal(1, 0.6, 500), rng.normal(8, 0.6, 500)]).clip(0, 9)
    one = rng.normal(4.5, 1.2, 1000).clip(0, 9)
    assert hierfit.bimodality(rows(two), "t")["bimodality_coef"] > 5 / 9
    assert hierfit.bimodality(rows(one), "t")["bimodality_coef"] < 5 / 9
