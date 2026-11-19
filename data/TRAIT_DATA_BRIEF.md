# Brief for writing trait data files

Study: we steer chat LLMs (Qwen3, Gemma 3) with a "trait" vector at increasing
strength and measure, per user prompt, the strength at which the reply starts
to express the trait. Trait definitions are in `configs/traits.yaml`.

For each trait, `data/traits/<trait>/` needs four JSON files:

- `extract_A.json`, `extract_B.json`, `extract_C.json`, each
  `{"instruction": [{"pos": "...", "neg": "..."} x5], "questions": ["..." x20]}`
- `eval.json`: `{"questions": ["..." x120]}`

If `extract_A.json` already exists (taken from the persona-vectors release), do
not change it; read it as the format example and write the other three files.
`data/upstream/extract/*.json` has more format examples.

## Extraction sets (used to compute the steering vector)
- `pos` instructions are system-prompt sentences telling the assistant to show
  the trait; `neg` tell it to show the opposite. One to two sentences each. The
  five pairs in a set cover different aspects of the trait.
- The 20 questions are user messages that could elicit the trait under the pos
  instruction but do not ask for it.
- Sets A, B, C must be independent: different wording of instructions and
  different question topics, so that three vectors computed from them are a
  fair test of repeatability. Do not paraphrase one set to get another.

## Evaluation set (120 user prompts, never used for extraction)
The unsteered model should usually NOT show the trait on these prompts, and
showing it must be a natural possibility, so that steering can flip the reply:
- refusing: ordinary harmless requests the model would normally just do
  (a few may be mildly sensitive but clearly acceptable). No harmful requests.
- sycophantic: the user states a mistaken belief, flawed plan, or weak piece of
  work and asks what the assistant thinks.
- hallucinating: questions about things that do not exist or cannot be known
  (invented people, papers, events, products, statistics), phrased as if real.
- deceptive: situations where a misleading answer is possible and tempting
  (unwelcome truths, questions about the assistant's own abilities or errors,
  advice where one could shade the facts). No system prompt is available.
- clarifying: under-specified requests where the model would normally assume
  details and answer, but could reasonably ask first.
- style traits (humorous, optimistic, poetic, neurotic, impolite): everyday
  questions and requests across many topics that would normally get a plain
  answer. Do not ask for jokes, poems, opinions on mood, etc.

Rules for all 120: self-contained single user messages; varied topics
(science, work, health, money, cooking, tech, relationships, travel, history,
school, hobbies ...); varied length (some one line, some two or three
sentences); none mention the trait or ask for it; no near-duplicates; plain
English; nothing harmful.

## Finish
Run `.venv/bin/python -m steer.validate_data <trait>` from the project root
for each of your traits and fix everything until it prints OK. Edit only files
under `data/traits/<your traits>/`.
