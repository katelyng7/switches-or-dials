"""Q1: hierarchical probit fit and switchiness S.

Model, for prompt i at strength a (x = log a):

    p_i(a) = g + (1 - g) * Phi((x - log t_i) / w),    log t_i ~ N(mu, sigma_t^2)

g is the trait's baseline rate (replies that express the trait unsteered), w
the within-prompt transition width, sigma_t the spread of thresholds across
prompts. S = sigma_t^2 / (sigma_t^2 + w^2): 1 = switch, 0 = dial.

Fit by marginal maximum likelihood: the prompt threshold is integrated out on a
fixed grid. Intervals come from resampling prompts.

  python -m analysis.hierfit --model qwen3-4b            # all traits and sets
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, ndtr

from steer.common import OUT, read_jsonl, traits
from steer.judge import THRESHOLD

Z = np.linspace(-4.5, 4.5, 181)
LOG_WZ = -0.5 * Z**2
LOG_WZ -= logsumexp(LOG_WZ)
LO, HI = np.log(0.02), np.log(6.0)  # bounds on w and sigma_t (log-strength units)
EPS = 1e-9


def unpack(theta):
    mu, ls, lw, lg = theta
    return mu, np.exp(ls), np.exp(lw), expit(lg)


def nll(theta, x, k, n):
    """x: (S,) log strengths, -inf for a = 0. k, n: (N, S) counts."""
    mu, sig, w, g = unpack(theta)
    tau = mu + sig * Z  # (J,)
    with np.errstate(invalid="ignore"):
        phi = ndtr((x[None, :] - tau[:, None]) / w)  # (J, S); x=-inf -> 0
    p = np.clip(g + (1 - g) * phi, EPS, 1 - EPS)
    ll = k @ np.log(p).T + (n - k) @ np.log1p(-p).T  # (N, J)
    return -logsumexp(ll + LOG_WZ[None, :], axis=1).sum()


def fit(a, k, n, starts=None):
    """a: (S,) strengths (0 allowed). Returns dict with mu, sigma_t, w, g, S, nll."""
    a, k, n = np.asarray(a, float), np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore"):
        x = np.log(a)
    pos = a > 0
    g0 = (k[:, ~pos].sum() + 0.5) / (n[:, ~pos].sum() + 1.0) if (~pos).any() else 0.02
    lg0 = np.log(g0 / (1 - g0))
    rate = k[:, pos].sum(0) / np.maximum(n[:, pos].sum(0), 1)
    mu0 = x[pos][np.argmin(np.abs(rate - (g0 + (1 - g0) / 2)))]
    if starts is None:
        starts = [(mu0, ls, lw, lg0) for ls in np.log([0.1, 0.5, 1.5]) for lw in np.log([0.1, 0.5, 1.5])]
    bounds = [(x[pos].min() - 3, x[pos].max() + 3), (LO, HI), (LO, HI), (-9, 3)]
    best = None
    for s in starts:
        r = minimize(nll, s, args=(x, k, n), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    mu, sig, w, g = unpack(best.x)
    return dict(mu=mu, sigma_t=sig, w=w, g=g, S=sig**2 / (sig**2 + w**2), nll=best.fun,
                theta=best.x.tolist(),
                # a parameter on its lower bound is only bounded above by the data
                w_at_floor=bool(best.x[2] < LO + 1e-3), sigma_at_floor=bool(best.x[1] < LO + 1e-3))


def bootstrap(a, k, n, est, B=200, seed=0):
    """Prompt bootstrap. Returns {param: (lo, hi)} 95% percentile intervals."""
    rng = np.random.default_rng(seed)
    k, n = np.asarray(k), np.asarray(n)
    draws = defaultdict(list)
    for _ in range(B):
        idx = rng.integers(0, len(k), len(k))
        f = fit(a, k[idx], n[idx], starts=[est["theta"], (est["theta"][0], np.log(0.5), np.log(0.5), est["theta"][3])])
        for key in ("mu", "sigma_t", "w", "g", "S"):
            draws[key].append(f[key])
    return {key: tuple(np.percentile(v, [2.5, 97.5])) for key, v in draws.items()}


def simulate(a, n_prompts, n_samples, mu, sigma_t, w, g=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    log_t = rng.normal(mu, sigma_t, n_prompts)
    with np.errstate(divide="ignore"):
        p = g + (1 - g) * ndtr((np.log(a)[None, :] - log_t[:, None]) / w)
    k = rng.binomial(n_samples, p)
    return k, np.full_like(k, n_samples)


# ---------- real data ----------

def load_scored(tag, model, trait, vset, judge):
    """Join generated rows with judge scores. Returns list of dict rows."""
    gen = read_jsonl(OUT / "gen" / tag / model / f"{trait}_{vset}.jsonl")
    scores = {r["id"]: r["scores"] for r in read_jsonl(OUT / "judge" / judge / "gen" / tag / model / f"{trait}_{vset}.jsonl")}
    return [{**r, "scores": scores[r["id"]]} for r in gen if r["id"] in scores]


def coherence_break(rows):
    """Strength at which mean coherence of sampled replies crosses THRESHOLD,
    interpolated in log strength. inf if it never does."""
    by_a = defaultdict(list)
    for r in rows:
        c = r["scores"].get("coherence")
        if r["sample"] != "greedy" and c is not None:
            by_a[r["a"]].append(c)
    pts = sorted((a, float(np.mean(v))) for a, v in by_a.items())
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if c0 >= THRESHOLD > c1:
            if a0 <= 0:
                return a1
            f = (c0 - THRESHOLD) / (c0 - c1)
            return float(np.exp(np.log(a0) + f * (np.log(a1) - np.log(a0))))
    return 0.0 if pts and pts[0][1] < THRESHOLD else float("inf")


def counts(rows, trait, max_a=float("inf")):
    """(strengths, k, n) over sampled replies at strengths below max_a."""
    strengths = sorted({r["a"] for r in rows if r["a"] < max_a})
    prompts = sorted({r["prompt_idx"] for r in rows})
    ai = {a: j for j, a in enumerate(strengths)}
    pi = {p: i for i, p in enumerate(prompts)}
    k = np.zeros((len(prompts), len(strengths)))
    n = np.zeros_like(k)
    for r in rows:
        s = r["scores"].get(trait)
        if r["sample"] == "greedy" or s is None or r["a"] not in ai:
            continue
        n[pi[r["prompt_idx"]], ai[r["a"]]] += 1
        k[pi[r["prompt_idx"]], ai[r["a"]]] += s >= THRESHOLD
    return np.array(strengths), k, n


def bimodality(rows, trait, max_a=float("inf")):
    """Reply-level form of the question: at the strength whose pooled expression
    rate is closest to one half, are judge scores bimodal (replies are either
    normal or trait-like) or unimodal (replies are all half trait-like)?
    Returns Sarle's bimodality coefficient (above 5/9 suggests two modes), the
    share of scores in the middle band [3, 7), and the strength used."""
    from scipy.stats import kurtosis, skew

    by_a = defaultdict(list)
    for r in rows:
        s = r["scores"].get(trait)
        if r["sample"] != "greedy" and s is not None and 0 < r["a"] < max_a:
            by_a[r["a"]].append(s)
    if not by_a:
        return None
    a = min(by_a, key=lambda x: abs(np.mean(np.array(by_a[x]) >= THRESHOLD) - 0.5))
    v = np.array(by_a[a])
    n = len(v)
    if n < 8 or v.std() == 0:
        return None
    bc = (skew(v) ** 2 + 1) / (kurtosis(v) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return dict(bimodality_coef=float(bc), mid_share=float(((v >= 3) & (v < 7)).mean()),
                bimodality_a=a, bimodality_rate=float((v >= THRESHOLD).mean()))


def fit_run(tag, model, trait, vset, judge, B=200):
    rows = load_scored(tag, model, trait, vset, judge)
    if not rows:
        return None
    brk = coherence_break(rows)
    a, k, n = counts(rows, trait, max_a=brk)
    if (a > 0).sum() < 3:
        return dict(model=model, trait=trait, vset=vset, coherence_break=brk, error="fewer than 3 usable strengths")
    est = fit(a, k, n)
    ci = bootstrap(a, k, n, est, B=B) if B else {}
    out = dict(model=model, trait=trait, vset=vset, coherence_break=brk, n_prompts=len(k),
               n_strengths=int((a > 0).sum()), **{p: est[p] for p in ("mu", "sigma_t", "w", "g", "S", "w_at_floor", "sigma_at_floor")})
    for p, (lo, hi) in ci.items():
        out[f"{p}_lo"], out[f"{p}_hi"] = lo, hi
    out.update(bimodality(rows, trait, max_a=brk) or {})
    # usable window (Q4): log distance from the median threshold to the break
    out["window"] = float(np.log(brk) - est["mu"]) if np.isfinite(brk) and brk > 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--judge", default="mistral-small")
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    results = []
    for trait in traits():
        for p in sorted((OUT / "gen" / args.tag / args.model).glob(f"{trait}_*.jsonl")):
            r = fit_run(args.tag, args.model, trait, p.stem.split("_")[-1], args.judge, args.boot)
            if r:
                results.append(r)
                print(json.dumps(r), flush=True)
    out = OUT / "results" / f"q1_{args.tag}_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
