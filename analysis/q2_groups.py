"""Q2: do decision traits have higher S than style traits?

Exact permutation test over trait labels on the difference in mean S. S per
trait is the mean over vector sets. Traits whose two judges disagree
(Spearman < 0.8, from analysis.judge_agreement) are left out.

  python -m analysis.q2_groups --model qwen3-4b
"""
import argparse
import itertools
import json

import numpy as np

from steer.common import OUT, traits


def permutation_test(S, groups):
    """S: {trait: value}; groups: {trait: 'decision'|'style'}."""
    names = sorted(S)
    vals = np.array([S[t] for t in names])
    is_dec = np.array([groups[t] == "decision" for t in names])
    stat = lambda m: vals[m].mean() - vals[~m].mean()
    obs = stat(is_dec)
    diffs = []
    for combo in itertools.combinations(range(len(names)), int(is_dec.sum())):
        m = np.zeros(len(names), bool); m[list(combo)] = True
        diffs.append(stat(m))
    diffs = np.array(diffs)
    return dict(diff=float(obs), n_splits=len(diffs), n_decision=int(is_dec.sum()), n_style=int((~is_dec).sum()),
                p_one_sided=float((diffs >= obs - 1e-12).mean()), p_two_sided=float((np.abs(diffs) >= abs(obs) - 1e-12).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    fits = json.loads((OUT / "results" / f"q1_{args.tag}_{args.model}.json").read_text())
    agree_path = OUT / "results" / f"judge_agreement_{args.tag}_{args.model}.json"
    excluded = set()
    if agree_path.exists():
        excluded = {t for t, r in json.loads(agree_path.read_text()).items() if r["spearman"] < 0.8}
    else:
        print("WARNING: no judge-agreement file; no trait excluded")
    per_trait = {}
    for f in fits:
        if "S" in f and f["trait"] not in excluded:
            per_trait.setdefault(f["trait"], []).append(f["S"])
    S = {t: float(np.mean(v)) for t, v in per_trait.items()}
    groups = {t: c["group"] for t, c in traits().items()}
    res = dict(S=S, excluded=sorted(excluded), **permutation_test(S, groups))
    print(json.dumps(res, indent=1))
    (OUT / "results" / f"q2_{args.tag}_{args.model}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
