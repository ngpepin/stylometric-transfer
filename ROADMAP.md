# ROADMAP

This document reviews the papers in `background/` and proposes a practical, staged roadmap for improving `stylometric-transfer` while preserving the project's core constraints:

- Explicit, interpretable, versionable JSON style fingerprints (no opaque latent style vectors as the core artifact)
- Local, explainable measurements as the grounding layer
- LLM used only for synthesis and rewriting (plus optional assistive parsing/ranking where already in scope)
- Meaning preservation and deviation reporting as non-negotiable

The roadmap is written to be actionable for this repo (Python-first, lightweight dependencies by default).

---

## Literature Review (by file in `background/`)

Notes on scope:
- Some PDFs are snapshots of web pages (Medium/Wikipedia/tutorial sites). These are useful context but not peer-reviewed.
- Some PDFs are paywalled previews (Springer/ScienceDirect) and thus only partially reviewable from the extracted text.
- A few filenames in `background/` contain Unicode punctuation (e.g., curly quotes, en dashes). This document normalizes those to ASCII for readability; use `ls background/` for exact spellings when opening files.

### `background/RJ-2016-007.pdf` - *Stylometry with R: A Package for Computational Text Analysis* (Eder, Rybicki, Kestemont; 2016)

Core contribution:
- Describes `stylo`, a widely used stylometric toolkit focused on language-independent feature extraction (token + character n-grams) and established distance metrics (notably Burrows' Delta), with both a GUI for exploration and an API for pipelines.

Key takeaways:
- Shallow, language-agnostic features (character/token n-grams, frequent words) remain strong baselines for authorship/style work.
- Distance metrics (Delta and related measures) are useful as explainable, quantitative comparisons.
- Reproducible pipelines + "exploration-first" tooling (visualization) are central for real-world use.

Implications for this repo:
- Add/expand multi-level n-gram profiles (character + word) and Burrows' Delta as optional local measurements/scorers.
- Treat distance metrics as first-class validators/compliance signals (with clear interpretations).

### `background/s41599-025-05986-3.pdf` - *Stylometric comparisons of human versus AI-generated creative writing* (O'Sullivan; 2025)

Core contribution:
- Uses Burrows' Delta + clustering/MDS on most frequent words to show clear separation between human and LLM creative writing.
- Finds human writing more heterogeneous; LLM outputs cluster tightly by model (higher stylistic uniformity).

Key takeaways:
- "Uniformity" is an empirical tell: LLMs often reduce stylistic variance.
- Delta on frequent words is a robust, content-resistant baseline for distinguishing sources (especially in controlled prompt settings).

Implications for this repo:
- Add "variance / uniformity" signals to the post-rewrite humanization report (without turning the project into a "detector").
- Use Delta-distance as a local quantitative check: output should move *toward the target style* without collapsing diversity.

### `background/Stylometry_Recognizes_Human_and_Llm-Generated_Text.pdf` - *Stylometry recognizes human and LLM-generated texts in short samples* (Przystalski et al.; arXiv 2025)

Core contribution:
- Classifies 10-sentence samples across multiple sources (Wikipedia + summarizers + multiple LLMs) using stylometric feature sets and tree-based models with SHAP analysis.

Key takeaways:
- Short samples can still be separable by stylometry (with enough feature coverage).
- Discriminative cues include punctuation/whitespace artifacts, word overuse patterns, and grammatical "standardization".
- Practical issue: some LLM outputs can "stop mid-sentence", changing sentence/word ratios and punctuation counts.

Implications for this repo:
- Keep "incomplete sentence / truncated output" detection as a hard quality gate (it directly conflicts with meaning preservation).
- Expand local validators to explicitly flag whitespace/punctuation artifacts and "standardization drift".

### `background/Beyond the surface_ stylometric analysis of GPT-4o's capacity for literary style imitation _ Digital Scholarship in the Humanities _ Oxford Academic.pdf` (2025)

Core contribution:
- Evaluates GPT-4o's ability to imitate literary style using multiple feature groups (frequent words, n-grams, readability/complexity, LIWC-like features, etc.).
- Shows LLMs can mimic surface features but often fail to replicate deeper, multi-level stylometric signatures.

Key takeaways:
- "Style" is multi-level; optimizing only surface constraints yields shallow imitation.
- Different feature groups vary in discriminative power; n-gram/multi-level profiles tend to separate originals from imitations more reliably.

Implications for this repo:
- Improve fingerprints by explicitly representing *multi-level* constraints (lexical + syntactic texture + discourse structure) and validating them quantitatively.
- Don't over-index on any single metric; prefer a suite of complementary, interpretable signals.

### `background/2510.13302v3.pdf` - *LLM One-Shot Style Transfer for Authorship Attribution and Verification* (Miralles-Gonzalez et al.; arXiv 2025)

Core contribution:
- Proposes an unsupervised authorship attribution/verification framework using **LLM conditional log-probabilities** as a style signal.
- Introduces a "neutralization" step: rewrite texts into a neutral style and compute style similarity with neutral versions to reduce topic leakage.

Key takeaways:
- Topic/style confounds are severe; evaluation pipelines need explicit countermeasures (neutralization, domain-shift tests).
- Using the language model as a scorer can outperform prompting-only baselines, but has clear scaling and cost/latency tradeoffs.
- Text length and distribution shift meaningfully affect decision thresholds.

Implications for this repo:
- Add an *optional* "LLM-as-scorer" validator mode (only when the backend supports usage/logprob-style scoring), separate from the explicit fingerprint JSON.
- Make "neutralized evaluation" a first-class option for checking style compliance while minimizing topical contamination.

### `background/TDRLM_ Stylometric learning for authorship verification by Topic-Debiasing - ScienceDirect.pdf` (Hu et al.; 2023) - **preview-only**

Core contribution:
- Argues that topic leakage harms stylometric generalization; proposes topic-debiasing representations using token-level topic scoring.

Key takeaways:
- Topic/style entanglement is a persistent failure mode; naive feature learning can "cheat" via topical cues.
- Topic-debiasing improves robustness under domain shifts.

Implications for this repo (without adopting deep latent representations as the primary artifact):
- Strengthen deterministic topic-debiasing for phrase/lexicon extraction (TF-IDF-like downweighting, entity/date suppression, corpus-frequency checks).
- Consider an *optional* topic-normalization step for evaluation (e.g., neutralized paraphrase + style scoring) while keeping the fingerprint JSON explicit and editable.

### `background/autextification-paper9.pdf` - *AI-Writing Detection Using an Ensemble of Transformers and Stylometric Features* (Mikros et al.; IberLEF/AuTexTification 2023)

Core contribution:
- Shows value of combining stylometric feature groups, especially multi-level n-gram profiles (AMNP), lexical diversity/readability indices, and dictionary-based psycho-linguistic features.

Key takeaways:
- AMNP-style representations (character + word n-grams + frequent words) are strong, compact, and interpretable enough to use as robust signals.
- Lexical diversity and readability indices are useful complements; relative frequencies help control for length.

Implications for this repo:
- Add AMNP-style measurements (explicitly: top char n-grams, top word n-grams, top words) as local stats and optional validators.
- Expand readability/lexical-diversity metrics (still explainable) and include them in "before/after" reports.

### `background/superuser,+cs.1430.pdf` - *An open stylometric system based on multilevel text analysis* (Eder, Piasecki, Walkowiak; 2017)

Core contribution:
- Advocates multi-level feature extraction (from frequent words to deeper linguistic features) and makes stylometry accessible via web tooling.

Key takeaways:
- Multi-level stylometry is valuable beyond authorship attribution (genre, chronology, semantic tagging support).
- Deep features require preprocessing/tooling but can be optional modules layered on top of shallow baselines.

Implications for this repo:
- Keep default pipeline lightweight; add "optional deep-features pack" (POS n-grams/lemmata) behind flags and documented dependency installs.
- Continue investing in usability (dashboards, reports, reproducible runs).

### `background/Introduction to stylometry with Python _ Programming Historian.pdf` (Laramee; 2018, updated 2023)

Core contribution:
- Pedagogical overview of classic stylometric methods (Mendenhall curves, chi-squared, Burrows' Delta).

Key takeaways:
- Word-length distributions and Delta-like measures are simple, explainable, and effective.
- Language/tokenization details matter; diacritics and tokenization assumptions should be explicit.

Implications for this repo:
- Add word-length distribution and/or Mendenhall-style curves as optional report charts.
- Make tokenization rules explicit and testable (especially around punctuation, quotes, and Unicode).

### `background/Computational Text Analysis with Stylometry and R  - CCDHHN.pdf` (workshop page; 2025)

Core contribution:
- High-level survey of stylometry use cases and methods (keywords, feature selection, supervised/unsupervised ML, visualization).

Key takeaways:
- Stylometry can target "voices within works" (e.g., character idiolects), chronology, translators, genre shifts.

Implications for this repo:
- For fiction, consider split measurements for narration vs dialogue, or per-character dialogue segments (optional, explicit).

### `background/Stylometric_analysis_of_AI-generated_texts_a_compa.pdf` (Jaashan & Bin-Hady; 2025)

Core contribution:
- Compares stylometric differences across AI models (ChatGPT vs DeepSeek) using readability and POS density differences.

Key takeaways:
- Readability and POS-density signals differ by model; these signals can also differentiate registers.

Implications for this repo:
- Keep POS-based signals optional (dependency-gated), but provide lightweight heuristics by default (e.g., function word profiles, clause markers).

### `background/e3sconf_afe2023_03007.pdf` (Mikherskii & Mikherskii; 2023)

Core contribution:
- Demonstrates stylometric classification and surveys related tasks (politeness, formality).

Key takeaways:
- Stylometry applies to register/politeness/formality -- useful "persona" dimensions when explicit and controlled.

Implications for this repo:
- Add optional persona/register targets as *explicit knobs* (e.g., formality, directness) and measure them with transparent heuristics.

### `background/Stylometric Methods in Comparative Text Analysis _ Springer Nature Link.pdf` (Grebennikov et al.; 2023) - **preview-only**

Core contribution:
- Frequency + association analyses across translations show that surface keyword frequency can match while association fields diverge.

Key takeaways:
- Translation shifts associations; "equivalence" is layered (keywords vs context).

Implications for this repo:
- Meaning preservation checks should consider both lexical preservation and contextual association shifts; at minimum, track named entities, numbers, and key terminology.

### `background/Stylometric Watermarks vs. LLM Watermarks_...Medium.pdf` (Topraksoy; 2025) - **non-peer-reviewed**

Core contribution:
- Overview of stylometric vs token-level watermarking and the arms-race dynamics (obfuscation, paraphrase).

Key takeaways:
- Style signals can be scrubbed or mimicked; no single signal is definitive.

Implications for this repo:
- Avoid claiming that any single metric "proves" human-ness; treat metrics as risk indicators.

### `background/Adversarial stylometry - Wikipedia.pdf` - **non-peer-reviewed**

Core contribution:
- Surveys authorship obfuscation/imitations, highlighting how style can be intentionally altered while preserving meaning.

Key takeaways:
- Adversarial settings are different from cooperative stylometry; robust detection is difficult.

Implications for this repo:
- Reinforces the importance of explicit ethics controls and deviation reporting; also motivates robust meaning-preservation checks.

### `background/Neural Language Style Transfer With StyleTransfer.pdf` (DataChef blog; updated 2024) - **non-peer-reviewed**

Core contribution:
- Narrative intro to stylometry/style transfer; highlights function words and distributions (Zipf) as style cues.

Implications for this repo:
- Mostly contextual; reinforces the value of function word / distributional signals and the need for interpretability.

---

## Synthesis: What the literature suggests (and what it means here)

Across these sources, the most consistent, actionable points for `stylometric-transfer` are:

1) **Style vs topic disentanglement is a first-order problem.**
   - Proper nouns are only the most obvious topical leakage.
   - Robust systems need topic-debiasing at multiple points: phrase mining, lexicon building, evaluation.

2) **"Deeper" style is multi-level (lexical + syntactic + discourse), not a single knob.**
   - Char/word n-grams and frequent-word distributions repeatedly show up as strong baselines.
   - POS/grammar-based signals help, but should be optional if they add heavy deps.

3) **Evaluation needs complementary, interpretable metrics.**
   - Delta-style distances + variance/uniformity checks are repeatedly validated.
   - Post-rewrite checks must catch truncation/incompleteness (a common LLM failure mode).

4) **Tooling matters.**
   - Reproducibility, dashboards, and clear reporting make stylometry usable outside specialist circles.

---

## Proposed Roadmap (actionable phases)

Each phase lists deliverables plus acceptance criteria. Phases are ordered for maximum product value under the repo's constraints.

### Phase 1 - Evaluation Harness + Reliability Gates (high impact, low regret)

Deliverables:
- Add an "evaluation harness" script that can run:
  - fingerprint stats on a corpus
  - rewrite on a document set
  - before/after metric reports and deltas
- Add hard quality gates for rewriting:
  - "non-empty output" invariant per chunk
  - incomplete-sentence detection (end punctuation / sentence boundary heuristics)
  - structural preservation checks (headings, lists, footnotes) + explicit warnings when violated
- Make all LLM calls share the same retry/backoff/timeout policy and consistently report:
  - error details
  - "succeeded after N retries" messages

Acceptance criteria:
- Running rewrite on a multi-chunk document never silently produces empty output; failures are retried; final failures are explicit and non-destructive.
- A regression test covers "empty final_markdown" and verifies fallback behavior.

Literature support:
- Truncation/incomplete output issues noted in Przystalski et al. (2025).

### Phase 2 - Add Multi-Level, Interpretable Stylometric Metrics (default-lightweight, optional deep pack)

Deliverables (default, no heavy deps):
- AMNP-style profiles:
  - top character n-grams (2-grams and/or 3-grams)
  - top word n-grams (2-grams)
  - top words (already present, but unify with AMNP)
- Burrows' Delta (or Delta-like) distance based on frequent words.
- Word-length distribution curves (Mendenhall-style) and summary statistics.
- Expanded lexical diversity / readability indices (carefully selected to stay robust and explainable).

Deliverables (optional deep pack behind a flag):
- POS-tag distributions + POS n-grams (only if dependency installed).
- Lemma-level features for morphologically rich languages (optional).

Acceptance criteria:
- Fingerprint JSON embeds the new measurements verbatim under `measurements.*` with schema support.
- `apply_fingerprint.py --metrics` can emit a before/after table including Delta distance, diversity, readability, and AMNP drift.

Literature support:
- `stylo` (Eder et al. 2016) and tutorials (Laramee) validate Delta + distributions.
- O'Sullivan (2025) shows Delta separates human vs LLM in creative writing.
- Mikros et al. (2023) highlight AMNP utility.

### Phase 3 - Topic-Debiasing and "Proper-Name Suppression" 2.0 (beyond blacklists)

Deliverables:
- Deterministic topic-debiasing for phrase/lexicon mining:
  - TF-IDF-like downweighting of rare topical terms across documents
  - stronger proper noun heuristics (caps ratio, honorific patterns, digit/date patterns)
  - entity blacklist support (already present; keep fast matching)
- Keep "content lexicon" separate from "style lexicon":
  - store topical "terms of art" separately (if needed for meaning preservation)
  - avoid treating them as "preferred style phrases"
- Optional "neutralization" evaluation:
  - neutralize candidate texts (LLM rewrite to plain neutral style) and compute style distance on neutralized versions to reduce topic leakage.

Acceptance criteria:
- `common_phrases` (and similar lists like sentence openers) become dominated by reusable lexical patterns rather than named entities.
- A test corpus with many entities (news/politics) does not produce a phrase list primarily composed of entities/dates.

Literature support:
- Topic leakage emphasis: Hu et al. (2023), Miralles-Gonzalez et al. (2025).
- Proper-name features appear strongly in stylometric classification explanations: Przystalski et al. (2025).

### Phase 4 - Humanization Score: Make it Research-Backed, Not Magical

Deliverables:
- Expand the "quantitative humanization" report (input and output) to include:
  - variance/uniformity signals (across sentences/paragraphs)
  - Delta distance to "human baseline" (if you maintain a small local baseline set) or to the source text (to avoid collapse)
  - punctuation/whitespace artifact checks
- Document exactly what the score is (and is not), with tunable weights.

Acceptance criteria:
- Score is stable across reruns (given same seed/tunables) and explains its components.
- Docs explicitly say it's a *risk indicator* and not a proof of authorship.

Literature support:
- Uniform clustering of LLM outputs: O'Sullivan (2025).
- Stylometric standardization and punctuation artifacts: Przystalski et al. (2025).

### Phase 5 - Fiction-Specific Modeling (dialogue vs narration; voices within works)

Deliverables:
- When fiction is detected or forced:
  - treat multi-word quoted spans as "author voice" for profiling and rewriting (already policy-level; extend metrics)
  - split measurements into narration vs dialogue buckets for dashboards/reports
- Add optional "character voice" analysis for dialogue-heavy texts (explicit, opt-in).

Acceptance criteria:
- Fiction vs non-fiction behavior is deterministic, test-covered, and clearly logged.
- Quotes handling covers straight and curly quotes and is documented.

Literature support:
- Workshop/tutorial sources highlight "voices within works" and different narrative styles.

### Phase 6 - Tooling & UX (dashboards, reproducibility, batch runs)

Deliverables:
- Extend `show_fingerprint.py` (and related wrappers) to show:
  - the new multi-level metrics and Delta/AMNP charts
  - before/after comparisons when provided both original and rewritten docs
- Add a "batch apply" mode with aggregated metrics and failure summaries.

Acceptance criteria:
- A single command can produce an HTML report for a run: inputs, outputs, deviations, metrics deltas.

Literature support:
- Strong emphasis on usability and exploratory workflows in `stylo` and open stylometric system papers.

---

## Suggested sequencing (what to do next)

If you want the shortest path to a noticeably more robust system:

1) Phase 1 (reliability gates + evaluation harness)
2) Phase 2 (Delta + AMNP + diversity/readability metrics)
3) Phase 3 (topic-debiasing for lexicon/phrases)

This ordering gives you: fewer catastrophic failures, better measurement depth, and better "style not topic" discipline.
