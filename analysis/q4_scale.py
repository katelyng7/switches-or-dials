"""Q4: scale and regimes.

  python -m analysis.q4_scale ladder --models qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b
  python -m analysis.q4_scale side --model qwen3-4b --low 0.3 --high 1.5

ladder: slope of mu, S and the usable window (log coherence break - mu)
against log parameter count, from the Q1 result files; intervals by
resampling traits.

side: runs made with `steer.generate --tag side --prompts neutral
--strengths 0 LOW HIGH` and judged with `--metrics all`. For each steering
direction (ten traits plus random controls) and regime, the off-target profile
is the change from a = 0 in every other trait's score. Reports
  retest       correlation of profiles between vector sets (need r > 0.7)
  specificity  mean correlation between different directions' profiles;
               the two-regime account predicts low below the break, high above.
"""
import argparse
import itertools
import json
from collections import defaultdict

import numpy as np

from steer.common import OUT, load_yaml, read_jsonl, traits


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def ladder(args):
    sizes = load_yaml("models.yaml")["steered"]
    per = defaultdict(lambda: defaultdict(dict))  # quantity -> trait -> model -> value
    for m in args.models:
        for f in json.loads((OUT / "results" / f"q1_{args.tag}_{m}.json").read_text()):
            if "S" not in f:
                continue
            for q in ("mu", "S", "window"):
                if f.get(q) is not None:
                    per[q][f["trait"]].setdefault(m, []).append(f[q])
    x_of = {m: np.log(sizes[m]["params_b"]) for m in args.models}
    rng = np.random.default_rng(0)
    res = {}
    for q, by_trait in per.items():
        names = [t for t, d in by_trait.items() if len(d) >= 3]

        def est(ts):
            xs, ys = [], []
            for m in args.models:
                v = [np.mean(by_trait[t][m]) for t in ts if m in by_trait[t]]
                if v:
                    xs.append(x_of[m]); ys.append(np.mean(v))
            return slope(xs, ys) if len(xs) >= 3 else np.nan

        if not names:
            continue
        boots = [est(list(rng.choice(names, len(names)))) for _ in range(2000)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        res[q] = dict(slope_per_log_param=est(names), lo=float(lo), hi=float(hi), n_traits=len(names))
        print(q, res[q])
    (OUT / "results" / f"q4_ladder_{args.tag}.json").write_text(json.dumps(res, indent=1))


def profiles(model, judge, tag):
    """{(direction, vset): {a: mean score per metric}}"""
    out = {}
    for p in sorted((OUT / "gen" / tag / model).glob("*.jsonl")):
        direction, vset = p.stem.rsplit("_", 1)
        a_of = {r["id"]: r["a"] for r in read_jsonl(p) if r["sample"] != "greedy"}
        acc = defaultdict(lambda: defaultdict(list))
        for r in read_jsonl(OUT / "judge" / judge / "gen" / tag / model / p.name):
            if r["id"] in a_of:
                for metric, v in r["scores"].items():
                    if v is not None:
                        acc[a_of[r["id"]]][metric].append(v)
        out[(direction, vset)] = {a: {m: float(np.mean(v)) for m, v in d.items()} for a, d in acc.items()}
    return out


def side(args):
    prof = profiles(args.model, args.judge, args.tag)
    names = list(traits())
    res = {}
    for regime, a in (("low", args.low), ("high", args.high)):
        vecs = {}
        for key, by_a in prof.items():
            if a in by_a and 0.0 in by_a:
                # off-target change from baseline; own trait masked out
                vecs[key] = np.array([np.nan if t == key[0] else by_a[a].get(t, np.nan) - by_a[0.0].get(t, np.nan) for t in names])

        def corr(u, v):
            ok = ~np.isnan(u) & ~np.isnan(v)
            return float(np.corrcoef(u[ok], v[ok])[0, 1]) if ok.sum() >= 4 else np.nan

        retest = [corr(vecs[k1], vecs[k2]) for k1, k2 in itertools.combinations(vecs, 2)
                  if k1[0] == k2[0] and not k1[0].startswith("random")]
        across = [corr(vecs[k1], vecs[k2]) for k1, k2 in itertools.combinations(vecs, 2) if k1[0] != k2[0]]
        res[regime] = dict(a=a, retest_r=float(np.nanmean(retest)) if retest else None,
                           between_direction_r=float(np.nanmean(across)) if across else None,
                           coherence=float(np.mean([prof[k][a]["coherence"] for k in vecs])) if vecs else None,
                           n_directions=len({k[0] for k in vecs}))
        print(regime, res[regime])
    (OUT / "results" / f"q4_side_{args.model}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ladder"); a.add_argument("--models", nargs="+", required=True); a.add_argument("--tag", default="main")
    b = sub.add_parser("side"); b.add_argument("--model", required=True); b.add_argument("--tag", default="side")
    b.add_argument("--judge", default="mistral-small"); b.add_argument("--low", type=float, required=True); b.add_argument("--high", type=float, required=True)
    args = ap.parse_args()
    {"ladder": ladder, "side": side}[args.cmd](args)


if __name__ == "__main__":
    main()
