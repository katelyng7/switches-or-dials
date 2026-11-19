"""Q3: where is the switch decided? Switching probability against k for each
intervention (prompt-only, first k generated tokens, after an unsteered prefix
of k), from runs made with `steer.generate --tag q3 --mode ... --k ...`.

  python -m analysis.q3_commit --model qwen3-4b
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from steer.common import OUT, traits
from steer.judge import THRESHOLD

from .hierfit import load_scored


def curve(rows, trait, B=1000, seed=0):
    """{(mode, k): (p, lo, hi)} with prompt-bootstrap 95% intervals."""
    cells = defaultdict(lambda: defaultdict(list))  # (mode,k) -> prompt -> [0/1]
    for r in rows:
        s = r["scores"].get(trait)
        if r["sample"] != "greedy" and s is not None:
            cells[(r["mode"], r["k"], r["a"])][r["prompt_idx"]].append(s >= THRESHOLD)
    rng = np.random.default_rng(seed)
    out = {}
    for key, by_prompt in cells.items():
        m = np.array([np.mean(v) for v in by_prompt.values()])
        boots = m[rng.integers(0, len(m), (B, len(m)))].mean(1)
        out[key] = (float(m.mean()), *map(float, np.percentile(boots, [2.5, 97.5])), len(m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="q3")
    ap.add_argument("--judge", default="mistral-small")
    args = ap.parse_args()
    res = []
    for trait in traits():
        for p in sorted((OUT / "gen" / args.tag / args.model).glob(f"{trait}_*.jsonl")):
            vset = p.stem.split("_")[-1]
            rows = load_scored(args.tag, args.model, trait, vset, args.judge)
            for (mode, k, a), (pm, lo, hi, n) in sorted(curve(rows, trait).items()):
                res.append(dict(trait=trait, vset=vset, mode=mode, k=k, a=a, p=pm, lo=lo, hi=hi, n_prompts=n))
                print(f"{trait:13s} {vset} {mode:9s} k={k:<3d} a={a:<5g} p={pm:.3f} [{lo:.3f}, {hi:.3f}]")
    out = OUT / "results" / f"q3_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
