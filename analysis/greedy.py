"""Greedy-decoding check on sigma_t (independent of the hierarchical model).

Under greedy decoding each prompt gives one reply per strength, so it has a
single flip point: the first strength whose reply expresses the trait. The
threshold is only known to lie between two grid strengths, so mu and sigma_t
are fitted to those intervals (interval-censored normal).

  python -m analysis.greedy --model qwen3-4b
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from steer.common import OUT, traits
from steer.judge import THRESHOLD

from .hierfit import coherence_break, load_scored


def fit_flips(strengths, flips):
    """strengths: sorted positive grid. flips: per prompt, the first strength
    that expressed the trait, or None if it never did (right-censored)."""
    x = np.log(np.asarray(strengths, float))
    lo, hi = [], []
    for f in flips:
        if f is None:
            lo.append(x[-1]); hi.append(np.inf)
        else:
            j = int(np.argmin(np.abs(x - np.log(f))))
            lo.append(x[j - 1] if j > 0 else -np.inf); hi.append(x[j])
    lo, hi = np.array(lo), np.array(hi)

    def nll(th):
        mu, s = th[0], np.exp(th[1])
        return -np.log(np.clip(ndtr((hi - mu) / s) - ndtr((lo - mu) / s), 1e-12, None)).sum()

    r = min((minimize(nll, (np.median(x), ls), method="Nelder-Mead") for ls in np.log([0.1, 0.5, 1.5])), key=lambda r: r.fun)
    return dict(mu=float(r.x[0]), sigma_t=float(np.exp(r.x[1])), n=len(flips), n_never=int(sum(f is None for f in flips)))


def flips_from_rows(rows, trait, max_a):
    by_prompt = {}
    for r in rows:
        s = r["scores"].get(trait)
        if r["sample"] == "greedy" and s is not None and r["a"] < max_a:
            by_prompt.setdefault(r["prompt_idx"], {})[r["a"]] = s >= THRESHOLD
    strengths = sorted({a for d in by_prompt.values() for a in d if a > 0})
    flips, n_baseline = [], 0
    for d in by_prompt.values():
        if d.get(0.0):  # expresses the trait unsteered: no flip to locate
            n_baseline += 1
            continue
        flips.append(next((a for a in strengths if d.get(a)), None))
    return strengths, flips, n_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--judge", default="mistral-small")
    args = ap.parse_args()
    out = []
    for trait in traits():
        for p in sorted((OUT / "gen" / args.tag / args.model).glob(f"{trait}_*.jsonl")):
            vset = p.stem.split("_")[-1]
            rows = load_scored(args.tag, args.model, trait, vset, args.judge)
            strengths, flips, nb = flips_from_rows(rows, trait, coherence_break(rows))
            if len(strengths) < 3 or len(flips) < 5:
                continue
            r = dict(model=args.model, trait=trait, vset=vset, n_baseline=nb, **fit_flips(strengths, flips))
            out.append(r); print(json.dumps(r))
    path = OUT / "results" / f"greedy_{args.tag}_{args.model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
