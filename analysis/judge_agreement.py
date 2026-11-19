"""Spearman correlation between the primary and second judge, per trait, on
the rows both scored. Traits below 0.8 are excluded from the Q2 test.

  python -m analysis.judge_agreement --model qwen3-4b
"""
import argparse
import json

from scipy.stats import spearmanr

from steer.common import OUT, read_jsonl, traits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--judge", default="mistral-small")
    ap.add_argument("--second", default="qwen3-14b")
    args = ap.parse_args()
    res = {}
    for trait in traits():
        a, b = [], []
        for p in sorted((OUT / "judge" / args.second / "gen" / args.tag / args.model).glob(f"{trait}_*.jsonl")):
            first = {r["id"]: r["scores"].get(trait) for r in read_jsonl(OUT / "judge" / args.judge / "gen" / args.tag / args.model / p.name)}
            for r in read_jsonl(p):
                x, y = first.get(r["id"]), r["scores"].get(trait)
                if x is not None and y is not None:
                    a.append(x); b.append(y)
        if len(a) >= 20:
            res[trait] = dict(spearman=float(spearmanr(a, b).statistic), n=len(a))
            print(f"{trait:13s} rho={res[trait]['spearman']:.3f} n={len(a)}")
    out = OUT / "results" / f"judge_agreement_{args.tag}_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
