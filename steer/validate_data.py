"""Check trait data files: python -m steer.validate_data [trait ...]

Rules: extract_{A,B,C}.json each have 5 pos/neg instruction pairs and 20
questions; eval.json has 120 questions; no question is repeated anywhere within
a trait (compared after lowercasing and stripping punctuation), so vectors are
never evaluated on prompts that produced them.
"""
import json
import re
import sys

from .common import ROOT, traits

SETS = ("extract_A", "extract_B", "extract_C")


def norm(q):
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def check(trait):
    errs = []
    d = ROOT / "data" / "traits" / trait
    seen = {}

    def add(qs, where):
        for q in qs:
            if not isinstance(q, str) or len(q.strip()) < 8:
                errs.append(f"{where}: bad question {q!r}")
                continue
            k = norm(q)
            if k in seen:
                errs.append(f"{where}: duplicate of {seen[k]}: {q!r}")
            seen[k] = where

    for s in SETS:
        p = d / f"{s}.json"
        if not p.exists():
            errs.append(f"missing {p.name}")
            continue
        j = json.loads(p.read_text())
        ins = j.get("instruction", [])
        if len(ins) != 5 or any(set(x) != {"pos", "neg"} for x in ins):
            errs.append(f"{s}: need 5 instruction pairs with keys pos, neg")
        if len(j.get("questions", [])) != 20:
            errs.append(f"{s}: need 20 questions, has {len(j.get('questions', []))}")
        add(j.get("questions", []), s)
    p = d / "eval.json"
    if not p.exists():
        errs.append("missing eval.json")
    else:
        qs = json.loads(p.read_text()).get("questions", [])
        if len(qs) != 120:
            errs.append(f"eval: need 120 questions, has {len(qs)}")
        add(qs, "eval")
    return errs


def main(argv):
    names = argv or list(traits())
    bad = 0
    for t in names:
        errs = check(t)
        bad += bool(errs)
        print(f"{t}: {'OK' if not errs else 'FAIL'}")
        for e in errs[:20]:
            print("   ", e)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
