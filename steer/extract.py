"""Steering-vector extraction (persona-vectors recipe, Chen et al. 2025).

Three stages, because the judge is a different model:

  python -m steer.extract gen --model qwen3-4b --trait humorous --vset A
  python -m steer.judge --input out/extract/qwen3-4b/humorous_A.jsonl
  python -m steer.extract vec --model qwen3-4b --trait humorous --vset A

`gen` writes replies under positive and negative system prompts. `vec` keeps
pairs where the positive reply expresses the trait, the negative one does not,
and both are coherent; averages residuals over reply tokens at every layer; and
saves pos-mean minus neg-mean as out/vectors/<model>/<trait>_<vset>.pt
(shape [layers+1, hidden]).

  python -m steer.extract agree --model qwen3-4b      # cosine between sets
"""
import argparse
import itertools
import json

import torch

from . import models
from .common import OUT, JsonlWriter, read_jsonl, trait_file, traits
from .generate import generate_batch
from .judge import THRESHOLD


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def system_prompt(trait_cfg, pol, instruction):
    name = trait_cfg["pos_name"] if pol == "pos" else trait_cfg["neg_name"]
    return f"You are {a_or_an(name)} {name} assistant. {instruction}"


def gen(args):
    cfg = traits()[args.trait]
    data = trait_file(args.trait, f"extract_{args.vset}")
    out_path = OUT / "extract" / args.model / f"{args.trait}_{args.vset}.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)}
    tasks = []
    for pol in ("pos", "neg"):
        for j, pair in enumerate(data["instruction"]):
            for qi, q in enumerate(data["questions"][: args.n_questions]):
                for r in range(args.reps):
                    tid = f"{pol}|{j}|{qi}|{r}"
                    if tid not in done:
                        tasks.append(dict(id=tid, pair=f"{j}|{qi}|{r}", pol=pol, question=q,
                                          system=system_prompt(cfg, pol, pair[pol]), trait=args.trait))
    print(f"{out_path}: {len(done)} done, {len(tasks)} to do", flush=True)
    if not tasks:
        return
    model, tok = models.load(args.model)
    writer = JsonlWriter(out_path)
    torch.manual_seed(args.seed)
    for bi in range(0, len(tasks), args.batch_size):
        batch = tasks[bi : bi + args.batch_size]
        texts = [models.chat(tok, t["question"], t["system"]) for t in batch]
        answers = generate_batch(model, tok, texts, max_new_tokens=args.max_new_tokens)
        writer.write([{**t, "answer": a} for t, a in zip(batch, answers)])
        print(f"  {bi + len(batch)}/{len(tasks)}", flush=True)
    writer.close()


@torch.no_grad()
def response_means(model, tok, rows):
    """Mean residual over reply tokens, per layer, averaged over rows. [L+1, d]"""
    total = None
    for r in rows:
        prompt = models.chat(tok, r["question"], r["system"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(r["answer"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + a_ids], device=model.device)
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        m = torch.stack([h[0, len(p_ids):, :].float().mean(0) for h in hs]).cpu()
        total = m if total is None else total + m
    return total / len(rows)


def vec(args):
    in_path = OUT / "extract" / args.model / f"{args.trait}_{args.vset}.jsonl"
    judged = {r["id"]: r["scores"] for r in read_jsonl(OUT / "judge" / args.judge / "extract" / args.model / in_path.name)}
    rows = {r["id"]: r for r in read_jsonl(in_path) if r["answer"].strip()}

    def ok(rid, want_trait):
        s = judged.get(rid)
        if not s or s.get(args.trait) is None or s.get("coherence") is None:
            return False
        return (s[args.trait] >= THRESHOLD) == want_trait and s["coherence"] >= THRESHOLD

    pairs = sorted({r["pair"] for r in rows.values()})
    kept = [p for p in pairs if f"pos|{p}" in rows and f"neg|{p}" in rows and ok(f"pos|{p}", True) and ok(f"neg|{p}", False)]
    print(f"{args.trait}_{args.vset}: kept {len(kept)} of {len(pairs)} pos/neg pairs", flush=True)
    if len(kept) < args.min_pairs:
        raise SystemExit(f"fewer than {args.min_pairs} usable pairs; inspect {in_path}")
    model, tok = models.load(args.model)
    diff = response_means(model, tok, [rows[f"pos|{p}"] for p in kept]) - response_means(
        model, tok, [rows[f"neg|{p}"] for p in kept]
    )
    out = OUT / "vectors" / args.model / f"{args.trait}_{args.vset}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(diff, out)
    layer = models.middle_layer(model)
    meta = dict(n_pairs=len(kept), n_candidates=len(pairs), layer=layer, norm_at_layer=diff[layer].norm().item())
    out.with_suffix(".json").write_text(json.dumps(meta))
    print(f"saved {out}  {meta}", flush=True)


def agree(args):
    """Cosine similarity at the steered layer between vector sets of each trait."""
    d = OUT / "vectors" / args.model
    for trait in traits():
        vs = {}
        for p in sorted(d.glob(f"{trait}_*.pt")):
            layer = json.loads(p.with_suffix(".json").read_text())["layer"]
            vs[p.stem.split("_")[-1]] = torch.load(p, weights_only=True)[layer]
        cos = [
            f"{a}{b}={torch.nn.functional.cosine_similarity(vs[a], vs[b], dim=0).item():.3f}"
            for a, b in itertools.combinations(sorted(vs), 2)
        ]
        print(f"{trait:14s} sets={''.join(sorted(vs)) or '-'}  {'  '.join(cos)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "vec", "agree"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--trait")
    ap.add_argument("--vset", default="A")
    ap.add_argument("--judge", default="mistral-small")
    ap.add_argument("--reps", type=int, default=2, help="replies per (instruction, question)")
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--min-pairs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"gen": gen, "vec": vec, "agree": agree}[args.stage](args)


if __name__ == "__main__":
    main()
