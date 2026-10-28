---
name: deslop-writing
description: Rewrite technical and research-facing text (dataset cards, model cards, Space/demo copy, READMEs, result summaries, blog drafts) so it reads like a careful researcher wrote it, with no AI writing tells and every claim traceable to a measured number.
---

# Deslop writing (technical and research text)

Use this when writing or editing any text people will read about our work: HF dataset and model cards, Space
copy, READMEs, results sections, blog drafts, PR descriptions. It extends the general `deslop` skill
(ryukijano-agent-skills:deslop, read its references for the phrase and trope catalogs) with rules for technical
and benchmark writing.

## Workflow

1. **Collect the facts first.** List every number, claim and caveat in the text, with its source (results file,
   log, CSV, paper). Anything without a source is cut or marked as a hypothesis.
2. **Decide the reader and their first question.** For a dataset card: "what is in it and how do I use it?"
   For a demo: "what am I looking at and what should I try?" For results: "what did you find and how sure are you?"
   Open with the answer to that question in one or two plain sentences.
3. **Rewrite** with the rules below.
4. **Run the checks** at the bottom, then diff old vs new and confirm no fact, number, licence term or caveat
   was lost or changed. Re-check numbers against the source, not against the old text.

## Rules

### Content
- Lead with what the thing is and what it shows. Motivation comes after, briefly, or not at all.
- Every claim carries its number, unit and condition: "overall Spearman .789 on 3,000 pretraining-kitchen states",
  not "strong ranking performance".
- Say each fact once. No summary paragraph that repeats the table, no "Headline" that restates the intro.
- Caveats are specific and stated once, near the claim they qualify ("simulation only, no physical robot"),
  not sprinkled as disclaimers at the end of every section.
- Separate measured results from interpretation. Mark inference with "we think" or "likely", and say what
  test would settle it.
- Credit other people's work by name and link (model, dataset, paper). Never let reused weights or code read as ours.
- Do not oversell novelty. If prior work exists, name it and state the specific addition.

### Language
- Plain words. Define each metric in words the first time ("Spearman: how well the model's ordering of the 64
  actions matches the true ordering, 1 is perfect, 0 is chance"). Keep the technical name for people who want it.
- Explain internal labels or drop them (`det_sub500_long`, `cw_goal25`, `noise25`). A reader should never need
  the codebase to parse a sentence.
- Active voice with a real subject: "we replay each action in MuJoCo", not "each action is replayed".
- No hype or AI-tell words: leverage, robust, seamless, cutting-edge, comprehensive, delve, unlock, empower,
  game-changing, novel (unless defended), state-of-the-art (unless benchmarked), "it's worth noting".
- No rhetorical openers in headings or body ("Can your model...?", "Ever wondered...?"). Headings are labels.
- Numbers: consistent precision within a table, units on every quantity (cm, s, GB), thousands separators,
  "22M to 8B" not "22M–8B" or "22M → 8B" in prose. Leading-zero-free correlations (.789) are fine if used everywhere.

### Formatting
- No em dashes, no unicode arrows in prose, no emoji.
- No bold-first bullets. Bold at most one phrase per section, only for the thing the reader must not miss.
- Tables for comparisons of 3+ items on the same columns; prose for reasoning. No bullet lists of one-word items.
- Short paragraphs (2 to 4 sentences). Headings that say what is under them ("Files", "How the ground truth is
  measured", "Results", "Limits", "Licence").
- Code blocks only for things the reader will copy and run.

## Checks before delivering
- Could a new reader say in one sentence what this is after the first paragraph?
- Does every number match its source file? Is any number repeated?
- Any unexplained acronym, internal label, or metric without a definition?
- Any hype word, em dash, arrow, emoji, bold-first bullet, rhetorical question?
- Any caveat or licence term lost in the rewrite?
- Any sentence that would still be true if you deleted it from the document? Delete it.
- Score with the `deslop` table (directness, rhythm, trust, authenticity, density); revise below 40/50 for
  public-facing text.
