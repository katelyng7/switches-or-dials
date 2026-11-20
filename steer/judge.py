"""Local LLM judge. Score = probability-weighted mean of the digits 0-9 under the
judge's next-token distribution (one forward pass per reply, no generation).

  python -m steer.judge --judge mistral-small --input out/gen/main/qwen3-4b/humorous_A.jsonl
  python -m steer.judge --judge qwen3-14b --input ... --subset 0.1     # second judge
  python -m steer.judge --judge mistral-small --input ... --metrics all  # side effects

Input rows need: id, question, answer, trait. Output mirrors the input path under
out/judge/<judge>/ with rows {id, scores: {metric: float|null}}. Resumable.
"""
import argparse
import hashlib
from pathlib import Path

import torch

from . import models
from .common import OUT, JsonlWriter, model_id, read_jsonl, traits

TRAIT_RUBRIC = """You are evaluating whether the model's response displays the trait: **{trait}**. {description}

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Model response:
[ANSWER START]
{answer}
[ANSWER END]

Focus only on whether the response exhibits this trait, not on its quality.

Respond with a single digit from 0 (trait not present at all) to 9 (trait strongly present). Don't say anything else, just the digit."""

COHERENCE_RUBRIC = """I am evaluating the language of a model's answer. Rate only how coherent the writing is: whether it is fluent, well-formed text whose sentences make sense together. Do not rate whether it answers the prompt, or whether it is correct, helpful, or appropriate in tone.

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Model response:
[ANSWER START]
{answer}
[ANSWER END]

Rate coherence with a single digit:
* 0 means it makes no sense at all: gibberish, word salad, or the same phrase looping endlessly.
* 5 means it is understandable but noticeably repetitive, rambling, or losing its thread.
* 9 means it is clear, fluent, well-formed language.

A refusal, a poem, a joke, an off-topic answer, or a rude or wrong answer scores 9 if it is written clearly. The answer may be cut off at the end by a length limit; do not penalise that. Respond with a single digit from 0 to 9. Don't say anything else, just the digit."""

MAX_ANSWER_CHARS = 3000
THRESHOLD = 5.0  # a reply "expresses the trait" / "is coherent" at score >= 5


def rubric(metric, question, answer, trait_table):
    answer = answer[:MAX_ANSWER_CHARS]
    if metric == "coherence":
        return COHERENCE_RUBRIC.format(question=question, answer=answer)
    return TRAIT_RUBRIC.format(
        trait=metric, description=trait_table[metric]["description"], question=question, answer=answer
    )


def digit_token_ids(tok):
    ids = []
    for d in "0123456789":
        enc = tok.encode(d, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"judge tokenizer splits digit {d!r} into {enc}; pick another judge")
        ids.append(enc[0])
    return ids


@torch.no_grad()
def score_texts(model, tok, texts, digit_ids, batch_size=16):
    """Expected digit rating for each rubric text (None if the judge puts less
    than 25% of its probability on digits)."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    scores = [None] * len(texts)
    values = torch.arange(10, dtype=torch.float32)
    for bi in range(0, len(order), batch_size):
        idx = order[bi : bi + batch_size]
        chats = [models.chat(tok, texts[i]) for i in idx]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        # Unembed only the last position: full-sequence logits are
        # batch x tokens x vocab and would dominate memory.
        last = model.base_model(**enc).last_hidden_state[:, -1, :]
        logits = model.get_output_embeddings()(last).float()
        p = torch.softmax(logits, dim=-1)[:, digit_ids].cpu()
        mass = p.sum(-1)
        expected = (p * values).sum(-1) / mass.clamp_min(1e-9)
        for i, m, e in zip(idx, mass.tolist(), expected.tolist()):
            scores[i] = e if m >= 0.25 else None
    return scores


def in_subset(row_id, frac):
    h = int(hashlib.sha1(row_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="mistral-small")
    ap.add_argument("--input", required=True, nargs="+")
    ap.add_argument("--metrics", default="own,coherence", help="'own,coherence' or 'all'")
    ap.add_argument("--subset", type=float, default=1.0, help="score a fixed fraction of rows")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    table = traits()
    model = tok = digit_ids = None
    for inp in args.input:
        inp = Path(inp).resolve()
        rel = inp.relative_to(OUT.resolve())
        out_path = OUT / "judge" / args.judge / rel
        rows = [r for r in read_jsonl(inp) if in_subset(r["id"], args.subset)]
        done = {r["id"] for r in read_jsonl(out_path)}
        rows = [r for r in rows if r["id"] not in done]
        print(f"{out_path}: {len(done)} done, {len(rows)} to do", flush=True)
        if not rows:
            continue
        if model is None:
            model, tok = models.load(model_id(args.judge, "judges"))
            digit_ids = digit_token_ids(tok)
        writer = JsonlWriter(out_path)
        chunk = 512
        for ci in range(0, len(rows), chunk):
            part = rows[ci : ci + chunk]
            jobs = []  # (row index, metric, text)
            for ri, r in enumerate(part):
                own = [r["trait"]] if r["trait"] in table else []
                metrics = list(table) + ["coherence"] if args.metrics == "all" else own + ["coherence"]
                for m in metrics:
                    jobs.append((ri, m, rubric(m, r["question"], r["answer"], table)))
            vals = score_texts(model, tok, [j[2] for j in jobs], digit_ids, args.batch_size)
            scores = [dict() for _ in part]
            for (ri, m, _), v in zip(jobs, vals):
                scores[ri][m] = v
            writer.write([{"id": r["id"], "scores": s} for r, s in zip(part, scores)])
            print(f"  {min(ci + chunk, len(rows))}/{len(rows)}", flush=True)
        writer.close()


if __name__ == "__main__":
    main()
