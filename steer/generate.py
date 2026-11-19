"""Steered generation over a strength grid. Resumable: rerun the same command.

  python -m steer.generate --model qwen3-4b --trait humorous --vset A --grid main
  python -m steer.generate ... --tag q3 --mode first_k --k 4 --strengths 0.4
  python -m steer.generate ... --tag side --prompts neutral --strengths 0.3 1.5

Output: out/gen/<tag>/<model>/<trait>_<vset>.jsonl, one reply per line.
`--trait randomN` steers along a seeded random direction (side-effect control).
"""
import argparse
import json
import time
import zlib

import torch

from . import models
from .common import OUT, ROOT, JsonlWriter, grid, read_jsonl, trait_file
from .hooks import PositionSteerer


def load_direction(model_short, trait, vset, layer, hidden):
    if trait.startswith("random"):
        g = torch.Generator().manual_seed(1000 + int(trait[len("random"):] or 0))
        return torch.randn(hidden, generator=g)
    path = OUT / "vectors" / model_short / f"{trait}_{vset}.pt"
    return torch.load(path, weights_only=True)[layer].float()


@torch.no_grad()
def generate_batch(model, tok, texts, steerer=None, do_sample=True, temperature=1.0, max_new_tokens=200):
    """Generate one reply per chat-formatted text. Returns decoded replies."""
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tok.pad_token_id)
    if do_sample:
        kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
    else:
        kwargs.update(temperature=None, top_p=None, top_k=None)
    if steerer is None:
        out = model.generate(**enc, **kwargs)
    else:
        with steerer:
            out = model.generate(**enc, **kwargs)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(o, skip_special_tokens=True).strip() for o in new]


def build_tasks(questions, strengths, n_samples, greedy, mode, k):
    tasks = []
    for qi, q in enumerate(questions):
        for a in strengths:
            reps = list(range(n_samples)) + (["greedy"] if greedy else [])
            for s in reps:
                tasks.append(
                    dict(id=f"{qi}|{a:g}|{s}|{mode}|{k}", prompt_idx=qi, question=q, a=float(a), sample=s)
                )
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--vset", default="A")
    ap.add_argument("--grid", default="main")
    ap.add_argument("--tag", default=None, help="output subfolder; defaults to the grid name")
    ap.add_argument("--strengths", type=float, nargs="*", help="override the grid's strengths")
    ap.add_argument("--n-samples", type=int)
    ap.add_argument("--n-prompts", type=int)
    ap.add_argument("--prompts", default="eval", help="'eval' (the trait's held-out set) or 'neutral'")
    ap.add_argument("--mode", default="response")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = grid(args.grid)
    strengths = args.strengths if args.strengths else g["strengths"]
    n_samples = args.n_samples if args.n_samples is not None else g["n_samples"]
    n_prompts = args.n_prompts or g["n_prompts"]
    if args.prompts == "neutral":
        questions = json.loads((ROOT / "data" / "neutral.json").read_text())["questions"]
    else:
        questions = trait_file(args.trait, "eval")["questions"]
    questions = questions[:n_prompts]

    out_path = OUT / "gen" / (args.tag or args.grid) / args.model / f"{args.trait}_{args.vset}.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)}
    tasks = build_tasks(questions, strengths, n_samples, not args.no_greedy, args.mode, args.k)
    todo = [t for t in tasks if t["id"] not in done]
    print(f"{out_path}: {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    model, tok = models.load(args.model)
    layer = models.middle_layer(model)
    scale = models.median_norm(model, tok, layer, cache_key=args.model)
    block = models.block_for(model, layer)
    hidden = model.get_input_embeddings().embedding_dim
    direction = load_direction(args.model, args.trait, args.vset, layer, hidden)

    texts = {q: models.chat(tok, q) for q in questions}
    lens = {q: len(tok(texts[q], add_special_tokens=False)["input_ids"]) for q in questions}
    writer = JsonlWriter(out_path)
    meta = dict(model=args.model, trait=args.trait, vset=args.vset, mode=args.mode, k=args.k, layer=layer)
    t0, n_done = time.time(), 0
    for is_greedy in (False, True):
        group = [t for t in todo if (t["sample"] == "greedy") == is_greedy]
        group.sort(key=lambda t: lens[t["question"]])  # similar lengths -> little padding
        for bi in range(0, len(group), args.batch_size):
            batch = group[bi : bi + args.batch_size]
            torch.manual_seed(args.seed * 1_000_003 + zlib.crc32(batch[0]["id"].encode()) % 1_000_000)
            steerer = PositionSteerer(
                block, direction, scale, torch.tensor([t["a"] for t in batch]), mode=args.mode, k=args.k
            )
            answers = generate_batch(
                model, tok, [texts[t["question"]] for t in batch], steerer,
                do_sample=not is_greedy, temperature=g["temperature"], max_new_tokens=g["max_new_tokens"],
            )
            writer.write([{**meta, **t, "answer": ans} for t, ans in zip(batch, answers)])
            n_done += len(batch)
            rate = n_done / (time.time() - t0)
            print(f"  {n_done}/{len(todo)}  {rate:.1f} replies/s", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
