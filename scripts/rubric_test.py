"""Compare coherence-rubric wordings on replies whose coherence we know by construction.

Should score high: unsteered replies, replies written under a trait system prompt
(in character but fluent), poems at a moderate strength. Should score low: replies
at the largest strength, where the text has visibly broken down.

  python scripts/rubric_test.py --judge mistral-small

Result on Qwen3-4B replies, Mistral-Small judge (share scoring >= 5):
                         with_question  no_question  broken
  unsteered                  100%          100%       100%
  neurotic persona             8%           99%        20%
  impolite persona            79%           97%        88%
  refusing persona            98%           98%       100%
  poetic a=0.38               92%          100%        94%
  strength 1.2 (3 traits)      0%            0%         0%
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steer import judge, models  # noqa: E402
from steer.common import OUT, model_id, read_jsonl  # noqa: E402

TAIL = "Respond with a single digit from 0 to 9. Don't say anything else, just the digit."

VARIANTS = {
    "with_question": """I am evaluating the language of a model's answer. Rate only how coherent the writing is: whether it is fluent, well-formed text whose sentences make sense together. Do not rate whether it answers the prompt, or whether it is correct, helpful, or appropriate in tone.

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

A refusal, a poem, a joke, an off-topic answer, or a rude or wrong answer scores 9 if it is written clearly. The answer may be cut off at the end by a length limit; do not penalise that. Respond with a single digit from 0 to 9. Don't say anything else, just the digit.""",
    "no_question": judge.COHERENCE_RUBRIC,  # adopted
    "broken": """I am checking whether a language model's output has broken down. A broken output is one no competent writer would produce on purpose: gibberish, word salad, sentences that start mid-thought, or a phrase repeated in a loop.

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Model response:
[ANSWER START]
{answer}
[ANSWER END]

Rate with a single digit:
* 0 means completely broken.
* 5 means readable, but with clear signs of breakdown such as repeated sentences or a lost thread.
* 9 means not broken at all: a competent writer could have written this on purpose.

Judge only breakdown. A response that ignores the prompt, refuses, is wrong, or is written in a strong style or persona (anxious, rude, comic, poetic) scores 9 as long as the writing itself is intact. The response may be cut off at the end by a length limit; do not penalise that. """ + TAIL,
}


def sample(rows, n, rng):
    rows = list(rows)
    return rng.sample(rows, min(n, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="mistral-small")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    rng = random.Random(0)
    ex = lambda t: read_jsonl(OUT / "extract" / args.model / f"{t}_A.jsonl")
    gen = lambda t: [r for r in read_jsonl(OUT / "gen" / "main" / args.model / f"{t}_A.jsonl") if r["sample"] != "greedy"]
    sets = {
        "HIGH unsteered": [r for t in ("poetic", "refusing") for r in gen(t) if r["a"] == 0],
        "HIGH neurotic persona": [r for r in ex("neurotic") if r["id"].startswith("pos")],
        "HIGH impolite persona": [r for r in ex("impolite") if r["id"].startswith("pos")],
        "HIGH refusing persona": [r for r in ex("refusing") if r["id"].startswith("pos")],
        "HIGH poetic a=0.38": [r for r in gen("poetic") if r["a"] == 0.38],
        "MID  poetic a=0.76": [r for r in gen("poetic") if r["a"] == 0.76],
        "MID  humorous a=0.6": [r for r in gen("humorous") if r["a"] == 0.6],
        "MID  refusing a=0.76": [r for r in gen("refusing") if r["a"] == 0.76],
        "LOW  humorous a=1.2": [r for r in gen("humorous") if r["a"] == 1.2],
        "LOW  refusing a=1.2": [r for r in gen("refusing") if r["a"] == 1.2],
        "LOW  clarifying a=1.2": [r for r in gen("clarifying") if r["a"] == 1.2],
    }
    sets = {k: sample(v, args.n, rng) for k, v in sets.items()}
    model, tok = models.load(model_id(args.judge, "judges"))
    digits = judge.digit_token_ids(tok)
    print(f"{'set':26s} " + " ".join(f"{v:>18s}" for v in VARIANTS))
    for name, rows in sets.items():
        cells = []
        for v, text in VARIANTS.items():
            s = judge.score_texts(model, tok, [text.format(question=r["question"], answer=r["answer"][: judge.MAX_ANSWER_CHARS]) for r in rows], digits, 32)
            s = np.array([x for x in s if x is not None])
            cells.append(f"{s.mean():5.2f} ({(s >= 5).mean():4.0%} ok)")
        print(f"{name:26s} " + " ".join(f"{c:>18s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
