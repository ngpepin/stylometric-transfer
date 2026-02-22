
**Repository:** `stylometric-transfer`  
**Keywords:** stylometry, computational stylistics, authorship attribution, controllable text generation, text style transfer, interpretability

(c) 2026 Nicolas Pepin

---

## Abstract

Stylometric-Transfer provides a method for stylometric profiling - extracting an author's corpus into a structured, interpretable JSON artifact (the style fingerprint) - and for meaning-preserving style transfer, rewriting new text to match the fingerprint using a large language model. The method combines classic stylometric measurement, such as punctuation rates and sentence-length distributions, with LLM-driven synthesis. This produces human-editable constraints: ranges, histograms, lexicon rules, and rhetorical templates. The fingerprint is formalized as a constraint set, and a constraint-satisfaction decoding approach is used for LLM rewriting, with compliance scored by measuring distributional divergences. The framework unifies stylometric transfer and humanization by quantifying a conflict-resolution layer that filters humanization guidelines against fingerprint constraints, and by supporting bounded stochastic variance under explicit controls. This design offers an auditable alternative to latent style embeddings, remaining consistent with established stylometry and text style transfer research.

**Reader's guide:** Sections 1–3 describe the system's purpose and rationale. Section 4 explains the measurements with simple examples. Sections 5–6 show how those measurements become constraints for rewriting. Section 2.5 details the humanization mathematics and chunk-sizing logic. Section 7 connects the ideas to the code. The appendices provide further mathematical and algorithmic detail for expert readers.

---

## 1. Introduction

Stylometry examines quantitative signals of writing style for tasks such as authorship attribution and author profiling. One example is the Federalist Papers analysis, where frequent-word statistics support Bayesian inference over disputed authorship ([press.uchicago.edu](https://press.uchicago.edu/ucp/books/book/distributed/I/bo5667096.html?utm_source=chatgpt.com)).

Text style transfer, on the other hand, seeks to transform text so that stylistic properties match a target style, while preserving content. Separating content from style without parallel data remains a challenge, leading to cross-alignment approaches and ongoing debate about evaluation and ethics ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com)).

Stylometry treats an author's style as a set of measurable habits: sentence length, punctuation frequency, recurring transitions, and words that are rarely used. Style transfer attempts to generate new text as if those habits were followed, without altering meaning. However, ensuring that a model does not introduce stylistic artifacts or factual drift is difficult. This paper addresses how that tension can be measured and managed.

A hybrid approach is proposed: style is represented explicitly as a stylometric style fingerprint (in JSON), and an LLM acts as a constrained rewriter, guided by both the fingerprint and locally measured statistics from the author corpus and candidate text. Humanization guidelines are integrated by means of a conflict-resolution layer, which deterministically filters guideline rules when they contradict fingerprint signals or the input's stylistic structure. Where enabled, bounded stochastic variance applies a small, seeded number of micro-edits to reduce AI-typical uniformity while preserving meaning and constraints.

### 1.1 Stylometry in Plain Language

Stylometry is best understood as a set of comparisons. The same set of signals is measured on two texts (or on a text and a corpus baseline), and the question is whether the numbers appear to come from the same distribution of writing habits.

In this project, "style" is defined as:

- **Habitual choices:** punctuation density, sentence length dispersion, paragraph cadence, transition habits.
- **Low-salience lexical defaults:** function-word balance, common collocations, preferred connectives.
- **Structural preferences:** frequency of one-sentence paragraphs, heading formation, list punctuation.

Style is not treated as:

- **Topic** (such as "spacecraft", "inflation", "ancient Athens") or proper-name density.
- **Facts** (entities, dates, quantities) - these are meaning and must not be fabricated or "smoothed".
- **Genre constraints** that derive from the medium rather than the author (Markdown boilerplate, license notices).

This distinction matters because naïve "style models" often learn topic proxies. A practical check: if all named entities are swapped or deleted but grammar and rhythm are retained, does the author still sound like themselves? If so, style rather than topic is likely being measured.

### 1.2 A Taxonomy of Stylometric Features

Stylometric features span multiple levels. The system focuses on what is both measurable with low complexity and stable enough to be useful.

1) **Orthography and typography (surface layer):** spelling variants (for example, Canadian vs US), contractions, dash/ellipsis conventions, heading case, and typographic habits. These are often consistent and editorially relevant.

>    In the implementation, orthography is localized at rewrite time: the fingerprint lexicon is normalized to a US baseline for cross-profile consistency, and a locale-specific ruleset (from `config.local_spelling_rules.json`) is applied after the LLM rewrite. Hard avoids (`config.avoid.txt`) are treated literally (no spelling normalization), so any desired spelling variants must be listed explicitly.

2) **Lexical statistics (word choice layer):** rare words (words that appear but are not repeated) and lexical avoidance (common words that are absent). These are treated as soft signals: they guide the rewrite but should not become brittle prohibitions.

3) **Function words and stance (cognitive layer):** function words (the, of, to, and, but, although, ...) and stance markers (hedges, boosters, directives) are historically strong stylometric signals because authors use them with little conscious control, and they survive topic changes better than content words.

4) **Syntax texture and discourse structure (compositional layer):** paragraph rhythm, sentence-opener patterns, transition placement, and rhetorical move signals (claim, evidence, counterpoint, concession, synthesis). These are particularly useful in style transfer because they shape the reader's experience without requiring new facts.

The important lesson is not that any one feature "identifies the author", but that many weak, interpretable features can jointly constrain the rewrite in a way that feels coherent while remaining auditable.

### 1.3 Why Distributions and Ranges Matter

Averages are fragile. Two writers can share the same mean sentence length while having very different distributions:

- Writer A: mostly 12–18 words, few long sentences (low variance).
- Writer B: mixes 6-word stingers with 45-word periodic sentences (high variance).

Encoding only the mean can force Writer B into Writer A's smoothness - an AI-typical regularity. Therefore, the fingerprint stores:

- **Histograms** (shape, not just central tendency),
- **Ranges** (tolerance for within-author variability),
- **Rates** (per-1000-word normalizations) rather than raw counts.

In practice, these distributions serve as both a measurement baseline for extracting style and a post-rewrite audit that detects compression, expansion, and other artifacts.

### 1.4 Humanization

"Humanization" is used here in a narrow sense:

> **Reducing AI‑typical artefacts** (over‑regular rhythm, self‑echo, template‑like transitions, suspiciously uniform paragraphing) **without adding facts, changing claims, or hiding provenance**.

It is not intended to defeat detectors or obscure authorship. The design constraint is the opposite: any humanization mechanism must be explicit, bounded, and observable in a metrics report.

This definition leads to a practical workflow:

1) Extract an author's natural variability (distributions, not just averages).
2) Rewrite with constraints.
3) Measure the result.
4) If the rewrite is "too smooth" or self-echoing, apply bounded, logged mechanisms (controller overlays; stochastic micro-operations) that nudge variability towards the author's baseline rather than towards arbitrary randomness.

### 1.5 Business Strategy and Opportunity

The spread of LLM writing systems changes not only how text is produced but also the economics of editorial work. As text becomes cheaper to generate, the binding constraints shift: attention, trust, compliance, and organizational coherence become scarcer and therefore more valuable. Stylometry and quantitative humanization can be read as managerial responses to that shift - they provide instruments for governance of writing at scale.

This section frames the opportunity in management terms: macro drivers, value creation and capture, adoption barriers and complements, and strategic risks and ethics.

#### 1.5.1 Macro drivers: why "writing operations" becomes a strategic capability

Several converging forces make explicit style control and human-likeness measurement increasingly important:

- **Supply shock in text production:** organizations can now generate large volumes of drafts, variants, and personalized messages. This makes quality assurance and voice coherence limiting factors.
- **Rising cost of trust failures:** hallucinated facts, inconsistent policy phrasing, and tone drift carry reputational and legal risk. The managerial problem becomes: how do we control and audit what gets published?
- **Regulatory and contractual pressure:** sectors such as finance, health, and government are increasingly constrained by record-keeping, disclosure obligations, and controlled language. A measurable style layer supports internal controls and external defensibility.
- **Channel fragmentation:** brands communicate across web, email, support, social, and internal documentation. A consistent voice is a coordination device; without it, the organization sounds like many organizations.

In this context, stylometry is not merely an academic method for attribution. It becomes a tool for operationalizing "voice" into measurable signals that can be monitored, tuned, and enforced.

#### 1.5.2 Value creation mechanisms

From a resource-based view, an organization's distinctive voice can be treated as an intangible asset. LLMs, however, make imitation cheap at the surface level, which increases the value of systems that preserve and audit the underlying asset. Stylometric fingerprints support three mechanisms:

1) **Editorial productivity without identity dilution:** faster drafting and rewriting, but with explicit constraints that preserve voice and meaning.
2) **Quality assurance via measurable proxies:** distributional audits (function words, punctuation densities, rhythm) catch "too-smooth" or overly templated artifacts that human editors frequently flag but cannot easily quantify.
3) **Traceable governance:** a versioned JSON fingerprint is an auditable policy object: it documents what the system believed the style to be, what constraints were prioritized, and where conflicts occurred (via deviations).

Humanization layers add a complementary capability: they do not aim to "make text undetectable", but to reduce systematic artifacts (over-regularity, self-echo) that degrade perceived quality. In managerial terms, this is variance management: introducing bounded, author-consistent dispersion so that outputs do not converge to the same generic, model-average cadence.

#### 1.5.3 Strategic use cases and buyer value

The opportunity spans multiple organizational functions:

- **Brand and communications:** preserve brand voice across campaigns, regions, and agencies; reduce "tone drift" when producing variants.
- **Knowledge management:** standardize internal documentation and onboarding materials while preserving domain precision; keep technical detail intact.
- **Customer support:** rewrite responses for clarity and empathy within bounded tone constraints; reduce template-like repetition that triggers user distrust.
- **Legal and compliance:** enforce hard avoids and controlled phrasing; maintain consistent hedging levels; log deviations for audit.
- **Publishing and media:** accelerate copy-editing, house-style conformity, and localization while keeping author identity legible.

Across these domains, the managerial value proposition is to increase throughput while reducing variance in the wrong dimensions (facts, policy wording, tone) and increasing variance in the right dimensions (natural rhythm, non-templated phrasing).

#### 1.5.4 Competitive landscape: explicit fingerprints as a differentiator

Many pipelines for generation rely on latent representations (fine-tuning, adapters, embeddings) that can be powerful but are difficult to inspect. For organizations, this creates governance friction: a latent style vector cannot easily be reviewed by an editor or a compliance officer.

An explicit fingerprint positions differently:

- **Interpretability as a product feature:** editors can read the rules, modify them, and reason about expected effects.
- **Versionability and change control:** fingerprints can be reviewed like policy documents: diffed, approved, rolled back.
- **Auditability and blame assignment:** when something goes wrong, deviations and measurements provide a partial causal trail.

The explicit-constraint approach trades some peak imitation fidelity for lower coordination and risk costs - a trade many organizations rationally prefer.

#### 1.5.5 Adoption barriers and complements

The main barriers are organizational:

- **Defining "voice" operationally:** teams must agree on what matters (clarity, warmth, formality) and accept that some elements are measurable proxies.
- **Domain drift and genre shift:** a fingerprint built on op-eds may not transfer cleanly to technical manuals. Governance requires multiple fingerprints or conditional policies.
- **Human-in-the-loop processes:** high-stakes publishing still benefits from editorial review. The system is best seen as an amplifier and consistency engine, not a replacement.
- **Infrastructure reliability:** chunking, retries, and timeouts are not implementation details; they shape the feasible operational envelope.

The most important complement is measurement literacy: users must understand what a metric movement means (for example, a 25% word-count drop suggests summarization risk). Without that literacy, a dashboard becomes decoration rather than control.

#### 1.5.6 Strategic risks and ethics

The strategic opportunity comes with governance risks:

- **Misuse risk:** systems that mimic authorial signatures can be used for impersonation. Guardrails and intended-use policies matter.
- **Over-optimization:** if organizations optimize for superficial "human" metrics, they can harm clarity or accuracy. Metrics must remain subordinate to meaning preservation.
- **Arms-race framing:** treating humanization as "evading detectors" is ethically fraught and strategically brittle. A better framing is editorial quality: avoid artifacts that readers dislike and that editors reject.

For these reasons, this project makes humanization explicit, deterministic by default, bounded when stochastic, and always logged. From a management perspective, this is the key design choice: it turns a vague aspiration ("make it sound human") into an auditable operational capability.

#### 1.5.7 Chat trust operations: API-assisted human-interaction validation

A practical business use case is chat trust operations: estimating whether an account that appears to be a human user is still being operated by that same human, rather than by an automated agent proxy. In this setting, the local API (`/make`, `/rate`) supports a behavioral assurance layer on top of conventional controls.

At a high level, the operating pattern is:

1) Build a reference fingerprint from trusted user-authored text (`POST /make`).
2) At periodic intervals (or event triggers), score new ostensibly user-provided text (`POST /rate`).
3) Optionally build rolling session fingerprints and compare them to the baseline with `POST /similarity` to detect profile-level drift.
4) Treat low probability, low similarity, or abrupt drift as risk signals for secondary checks (step-up authentication, moderation review, or conversation throttling), not as automatic ban decisions.

From a business perspective, this creates a lightweight control surface that can reduce abuse costs (credential sharing, scripted account farming, synthetic support interactions) while preserving user experience for low-risk traffic. The value is strongest when stylometry is fused with orthogonal signals (device reputation, session telemetry, challenge-response), because stylometric evidence alone is probabilistic and can be noisy on short text.

**Suggested operating model**

- **Enrollment window:** build the initial fingerprint only from text collected under higher trust conditions (for example, successful recent MFA sessions and low-risk device posture).
- **Scoring cadence:** score every $k$ user turns, and also on risk triggers (sudden session geo-change, unusual API call patterns, rapid prompt-copy behavior).
- **Evidence floor:** defer high-impact actions when token evidence is below a minimum threshold; short segments should contribute weakly to risk.
- **Rolling updates:** refresh fingerprints periodically to capture natural drift in user style (life events, domain changes, language shifts).
- **Case logging:** store probability, confidence interval, evidence size, and escalation rationale for audit and model governance.

**Decisioning pattern**

Let $p_t$ be the style-match probability from `/rate` at turn $t$. A practical risk score can be formed as a rolling aggregation:

$$
\bar{p}_t = \frac{1}{m}\sum_{i=t-m+1}^{t} p_i, \qquad
R_t = \alpha(1-\bar{p}_t) + \beta D_t + \gamma C_t
$$

where:
- $D_t$ captures drift volatility (for example, recent variance or slope in $p_i$),
- $C_t$ captures contextual risk from non-stylometric controls,
- $\alpha,\beta,\gamma$ are operational weights.

This allows tiered response rather than binary classification:
- **Low risk:** continue session normally.
- **Medium risk:** passive friction (light challenge, reduced tool privileges).
- **High risk:** step-up verification or human review queue.
- **Critical risk:** temporary action constraints pending adjudication.

**Business KPIs for deployment**

Program success should be measured as an operations portfolio, not a single classifier score:
- reduction in confirmed automated-abuse incidents per 1,000 sessions,
- false-positive escalation rate (human users sent to review),
- median time-to-resolution for escalated sessions,
- user-friction impact (challenge rate, abandonment rate),
- review precision (fraction of flagged sessions confirmed risky),
- net cost impact (abuse losses avoided minus review and friction costs).

**Failure modes and mitigation controls**

- **Legitimate style drift:** users can change writing register by context or stress; mitigate via rolling baselines and recency weighting.
- **Cross-genre mismatch:** support chats, appeals, and technical prompts may have different style regimes; mitigate with per-context fingerprints or context-aware thresholds.
- **Adversarial mimicry:** sophisticated attackers may imitate prior text; mitigate by combining stylometry with independent session controls and anomaly features.
- **Sparse evidence:** short or formulaic text can yield unstable estimates; mitigate by delaying irreversible actions until evidence accumulates.
- **Population bias:** non-native language users or accessibility tools may alter stylometric signals; mitigate through fairness monitoring and calibrated overrides.

**Governance and compliance implications**

The core governance point is proportional response: stylometric mismatch is an indicator, not identity proof. In regulated or high-stakes environments, the signal should inform risk tiering and review queues, never replace due process or explicit user authentication controls. Data minimization is also essential: retain only what is needed for model operation and audit, define retention windows, and document user-notice and appeal pathways where policy requires them.

In short, API-based stylometric validation is most effective as a risk amplifier in a defense-in-depth architecture, not as a standalone gatekeeper.

---

## 2. Related Work

This section provides a brief overview for non-specialists and references for specialists. The central point is that measurements are kept interpretable, and the LLM is tasked with following constraints rather than relying on hidden style embeddings.

### 2.1 Stylometry and Distance-Based Measures

Stylometric authorship attribution typically relies on robust, interpretable features such as word frequency profiles and distance measures. Burrows's Delta and its variants remain widely used; recent work explains the decomposition of feature selection, scaling (for example, z-transformation), and distance metrics, clarifying the effectiveness of Delta-style measures ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com)).

For those new to the field, these methods are intentionally simple, relying on counts and distributions rather than opaque neural features. This makes them suitable for interpretable fingerprints.

### 2.2 Text Style Transfer and Evaluation

Non-parallel TST methods, such as cross-alignment, demonstrate that certain stylistic attributes can be changed without parallel sentence pairs ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com)). Recent surveys discuss broad applications alongside challenges in evaluation and ethical risk, including concerns about misuse for impersonation, and support explicit safeguards and transparency in TST pipelines ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com)).

The literature indicates that style is difficult to define precisely. The approach here is to make style explicit and auditable, rather than learnt and hidden.

### 2.3 Corpus Size and Diminishing Returns

Stylometric signals stabilize as corpus size grows. A practical rule of thumb: approximately 20–50k words often yields a stable fingerprint for a single author or genre, while around 100k words typically captures most steady signals. If key rates (sentence and paragraph distributions, punctuation per 1k words, function-word profile, stance rates) drift by less than 1–2% after adding another 10–20k words, diminishing returns are likely. Additional data remains valuable when the corpus spans multiple genres or eras, or when capturing sparse phenomena such as rare rhetorical moves or infrequent lexical patterns.

### 2.4 Humanization-Aware Stylometric Transfer

Most style-transfer pipelines treat humanization as a separate editing step. Stylometric-Transfer incorporates humanization directly into constraint-guided rewriting by formalizing a conflict-resolution layer: humanization guidelines are applied only when they do not violate fingerprint-derived constraints or the input's structural features (such as heading case or inline-header lists). The guideline list is parsed into structured rules by an LLM (with deterministic fallback), then filtered against fingerprint signals before any rewrite prompt is constructed. This produces a single, auditable framework that balances stylistic fidelity with the removal of AI artifacts, rather than relying on post-hoc edits that may diverge from the author's voice.

### 2.5 Humanization Mechanisms and Benefits

Humanization in this system is a set of explicit, inspectable mechanisms designed to target known LLM artifacts while preserving the author's voice. These mechanisms include:

- **Conflict-filtered guidelines:** generic humanization rules are only applied when they do not contradict the fingerprint's statistical baselines (for example, avoiding "don’t use em-dashes" when the author's corpus uses them frequently).
- **Mandatory hygiene rules:** optional hard constraints (such as removing em-dashes or replacing emojis) are enforced deterministically and recorded in the deviations log.
- **Structural preservation:** blockquotes, citations, footnotes, code spans, and (for non-fiction) multi-word quotations are shielded from stylistic edits to prevent false "humanization" changes in non-authorial content.
- **Bounded stochastic variance:** when enabled, a small, seeded number of micro-edits (for example, swapping transition words or dropping filler terms) introduce controlled irregularity without semantic drift.

The practical benefit is a measurable reduction in AI-typical uniformity while keeping the transformation aligned to the author's measurable habits. Because the humanization layer is explicit, deterministic by default, and logged, it can be audited and tuned without introducing opaque behaviour. In short, humanization is treated as a constrained post-processing step integrated into the same interpretability framework as stylometric profiling itself.

#### 2.5.1 Quantitative humanization metrics

When `--metrics` is enabled, a compact quantitative report is emitted for both the input and the output. Let a text contain $N$ word tokens and $V$ unique types. Let $f_i$ be the frequency of type $i$.

The intent is not to "detect AI" in a forensic sense. These metrics are operational proxies for failure modes that practitioners repeatedly observe in LLM rewrites:

- **Over-regularity:** too even a rhythm, too consistent sentence length, too little local fluctuation.
- **Self-echo:** repeating the same phrases or n-grams.
- **Distributional drift:** function-word and punctuation patterns that stop resembling the target author.
- **Over-compression:** dropping detail, merging sentences, or collapsing paragraphs in a way that changes pacing.

Because the system's primary invariants are meaning preservation and interpretability, each metric is deliberately simple and inspectable. The best use of the report is comparative: observe which metrics move between input and output, and whether they move in a plausible direction for the author and genre.

A high aggregate score does not constitute proof of humanness; however, a pronounced, repeated regression in a single dimension can serve as a useful debugging signal.

**Lexical diversity family**

$$\mathrm{TTR}=\frac{V}{N}, \qquad C=\frac{\log V}{\log N}, \qquad R=\frac{V}{\sqrt{N}}.$$

The Maas index is calculated as

$$a^2=\frac{\log N-\log V}{(\log N)^2},$$

The reported score is calculated using its inverse, so a smaller $a^2$ indicates greater diversity.

**Repetition and distributional diversity**

Let $m_1 = N$ and $m_2=\sum_i f_i^2$. Yule's $K$ and Simpson's $D$ are defined as:

$$K = 10^4 \cdot \frac{m_2-m_1}{m_1^2}, \qquad D=\frac{\sum_i f_i(f_i-1)}{N(N-1)}.$$

Both metrics are inverted for scoring; lower repetition yields a higher score.

Self-echo repetition is measured using repeated n-grams. For bigrams and trigrams:

$$r_n=\frac{\sum_{g: c_g\ge 3} c_g}{|G_n|}, \qquad r=\frac{r_2+r_3}{2}.$$

The TTR/Herdan/Guiraud/Maas group captures the breadth of distinct words, whereas the Yule/Simpson/repeat-rate group assesses the concentration of vocabulary and phrasing. In practice, synonym substitution can increase diversity; these signals are most informative when considered alongside checks for meaning preservation and the author's baseline.

**Burstiness and rhythm**

For sentence lengths $\ell_s$ (words) and paragraph lengths $\ell_p$ (sentences):

$$B_s=\frac{\sigma(\ell_s)}{\mu(\ell_s)}, \qquad B_p=\frac{\sigma(\ell_p)}{\mu(\ell_p)}.$$

These values describe rhythmic variability rather than average length.

**Punctuation and function-word entropy**

Let $c_k$ represent the counts for each punctuation type. Define $p_k = c_k / \sum c_k$.

$$H_{\text{punct}} = -\sum_k p_k \log_2 p_k.$$

Function-word entropy is calculated in the same way, using function-word counts. If $P$ is the output function-word distribution and $Q$ is the fingerprint’s distribution, then:

$$D_{\mathrm{KL}}(P\|Q) = \sum_i P_i \log_2 \frac{P_i}{Q_i}.$$

The scoring system uses a clipped inverse of $D_{\mathrm{KL}}$, so lower divergence yields a higher score.

This approach is effective because function words are difficult to manipulate deliberately at scale and tend to remain stable within an author and genre. If KL divergence remains high across runs, it usually means the rewrite is too aggressive, the input is outside the fingerprint’s domain, or the chunking and constraints are causing paraphrases that shift the function-word balance.

**Sentence length divergence vs fingerprint**

Let $P$ and $Q$ be sentence-length histograms (with bins matching the fingerprint), and $M = \tfrac{1}{2}(P + Q)$.

$$\mathrm{JSD}(P, Q) = \tfrac{1}{2} D_{\mathrm{KL}}(P\|M) + \tfrac{1}{2} D_{\mathrm{KL}}(Q\|M).$$

The score uses $1 - \mathrm{JSD}$ (clipped).

**Character trigram entropy**

Let $t$ be character trigrams over letters (non-letters removed). With $p_t$ as the trigram distribution:

$$H_{3\text{-gram}} = -\sum_t p_t \log_2 p_t.$$

Character-level entropy gives a rough measure of orthographic texture. It reacts to repeated patterns, boilerplate phrasing, frequent suffixes, and templated text. Technical writing can lower trigram entropy because of repeated terminology, while literary writing may raise it.

**Average word length**

$$\bar{\ell} = \frac{1}{N} \sum_{j=1}^N \ell(w_j),$$

scored by distance from a neutral reference (for example, 5 characters).

**Metric summary (symbols and intent)**

| Metric | Symbol / definition | Intent |
| --- | --- | --- |
| Type–token ratio | $\mathrm{TTR} = V/N$ | Surface lexical diversity (length-sensitive). |
| Herdan’s $C$ | $C = \log V / \log N$ | Length-normalized diversity. |
| Guiraud’s $R$ | $R = V / \sqrt{N}$ | Length-normalized diversity. |
| Maas index (inverse) | $a^2 = (\log N - \log V) / (\log N)^2$ | Diversity via inverse of $a^2$. |
| Yule’s $K$ (inverse) | $K = 10^4 (m_2 - m_1) / m_1^2$ | Penalizes repetition. |
| Simpson’s $D$ (inverse) | $D = \sum f_i(f_i-1) / (N(N-1))$ | Penalizes repetition. |
| Repeat rate | $r = \tfrac{1}{2}(r_2 + r_3)$ | Self-echo (reused n-grams). |
| Sentence burstiness | $B_s = \sigma(\ell_s) / \mu(\ell_s)$ | Rhythm variability at sentence level. |
| Paragraph burstiness | $B_p = \sigma(\ell_p) / \mu(\ell_p)$ | Rhythm variability at paragraph level. |
| Punctuation entropy | $H_{\text{punct}} = -\sum p_k \log_2 p_k$ | Variety and balance of punctuation. |
| Punctuation variety | $\#\{k: c_k > 0\}$ | Breadth of punctuation usage. |
| Function-word entropy | $H_{\text{fw}} = -\sum p_i \log_2 p_i$ | Balance of function words. |
| Function-word KL (inverse) | $D_{\mathrm{KL}}(P\|Q)$ | Divergence from fingerprint profile. |
| Sentence-length JSD (inverse) | $\mathrm{JSD}(P, Q)$ | Divergence from fingerprint histogram. |
| Char trigram entropy | $H_{3\text{-gram}}$ | Low-level orthographic diversity. |
| Avg word length | $\bar{\ell}$ | Neutrality vs extreme short/long bias. |

#### 2.5.2 Aggregate 0–100 humanization score

Each metric is normalized into $[0,1]$ by a simple clipping rule. For metrics where higher values are better:

$$s_i = \min\left(1, \frac{x_i}{c_i}\right),$$

and for inverse metrics (like repetition or divergence):

$$s_i = \max\left(0, 1 - \frac{x_i}{c_i}\right).$$

Given weights $w_i$ (from `humanization_metrics.weights`), the aggregate score is:

$$S = 100 \cdot \frac{\sum_i w_i s_i}{\sum_i w_i}.$$

This produces a compact, interpretable 0–100 score, while keeping the underlying per-metric diagnostics for review.

The aggregate score acts as a dashboard indicator, not a scientific measurement. It is useful for regression testing (to check if a change has degraded outputs) and for quickly spotting suspicious runs (such as outputs that are unusually repetitive or flat). It is not meant for comparing unrelated authors, nor should it be used to optimize outputs at the expense of meaning preservation.

**How to read the metrics report (practical checklist)**

1) **Check invariants:** scan the deviations for meaning-preservation problems (missing sections, changed numerals, broken citations or quotations). If these fail, ignore the score and fix preservation first.
2) **Look for compression artifacts:** large drops in word or paragraph counts often mean over-summarization. Check if chunk sizing or aggressive rewrite policies are causing compression.
3) **Check repetition and rhythm:** if `repetition_inverse` drops or burstiness collapses, the output may be templated. Consider enabling controller overlays, adding mild stochastic variance, or reducing chunk size.
4) **Watch for distributional drift:** if both function-word KL divergence and sentence-length JS divergence rise, the output may be drifting from the author’s profile (often due to out-of-domain input or weak constraints).
5) **Consult the aggregate score last:** treat it as a regression indicator. Compare runs with identical inputs and settings; do not compare unrelated authors or genres.

#### 2.5.3 Corpus baselines and controller overlays

To avoid over-normalization that erases an author’s natural variability, the fingerprint can include corpus-derived humanization baselines (rolling windows). For each metric $m$, the corpus is scanned in windows of size $W$ with stride $S$ and summarized into quantiles $\{p10, p25, p50, p75, p90\}$ plus mean. These baseline distributions are stored in `measurements.humanization_baseline` and withheld from the LLM prompt.

Human writing is not stationary: within a single author, local passages may be more clipped, discursive, list-heavy, or more punctuated than the global average. A single global target can create uniformity the system aims to avoid. Windowed baselines make this variability explicit and measurable.

During rewriting, a controller overlay can apply chunk-level targets sampled from these baselines. For a metric $m$ and quantile $q$, let $v = \mathcal{Q}_m(q)$ and define a symmetric target band:

$$w = \min(\mathrm{max\_width}, \ \max(\mathrm{min\_width}, \ |v| \cdot \mathrm{range\_pct})), \qquad [v-w, v+w].$$

Ratios (like one-sentence paragraph rate) are clamped to $[0,1]$. These per-chunk ranges are merged into the fingerprint only for that chunk, and the overlay is logged for audit. If the observed metric $o$ falls outside the band by more than a tolerance $\tau$, the next retry receives targeted feedback:

$$\text{deviation} = \frac{|o - \mathrm{clip}(o, [v-w, v+w])|}{\max(\epsilon, 2w)} > \tau.$$

In practice, the overlay is a gentle nudge, not a strict constraint. It broadens the set of acceptable outputs by shifting targets within plausible regions of the author’s distribution. The overlay is computed locally and logged, so it is auditable and reversible. The feedback loop is intentionally limited to prevent endless refinement.

#### 2.5.4 Bounded stochastic variance

A small, seeded perturbation layer introduces controlled irregularity without changing meaning. If $\mathrm{ops}_{1000}$ is the maximum micro-operations per 1000 words, then for a chunk of $N$ tokens:

$$n_{\text{ops}} = \left\lfloor \mathrm{ops}_{1000} \cdot \frac{N}{1000} \right\rfloor.$$

Permitted operations (such as `swap_transition`, `drop_filler`) are sampled deterministically from the seed; all applied operations are logged as deviations. This layer is explicit and bounded by design.

The mechanism is intentionally conservative. It aims to reduce obvious templating and repetition, not to introduce stylistic novelty. From an engineering standpoint, it provides a deterministic salt that helps break repeated local patterns while remaining easy to audit (the applied operations are explicit).

#### 2.5.5 Chunk sizing and variance-aware splitting

Chunking is computed before any LLM call, so the chunk count is deterministic and fully known. Let $T_{\text{max}}$ be the model’s maximum prompt token budget, and $T_{\text{base}}$ the prompt overhead (system, scaffold, fingerprint). The base budget is:

$$T_{\text{in}} = \max(400, T_{\text{max}} - T_{\text{base}}).$$

If a tunable cap $T_{\text{cap}}$ is set, then:

$$T_{\text{in}} = \min(T_{\text{in}}, T_{\text{cap}}).$$

When variance-aware chunking is enabled, a scaling factor $f \in [f_{\min}, f_{\max}]$ from baseline variability is applied:

$$T_{\text{in}} = \max(200, \lfloor f \cdot T_{\text{in}} \rfloor).$$

A rough character budget follows $C_{\text{max}} \approx 4 \cdot T_{\text{in}}$.

The two main objectives of chunking are:

1) **Reliability:** stay well below the endpoint’s practical limits (timeouts, rate limits, response truncation).
2) **Coherence:** keep enough local context so the LLM can preserve meaning and Markdown structure without over-summarizing.

Smaller chunks improve reliability (especially on unstable endpoints) but can increase total runtime and the risk of compression artifacts at boundaries. Larger chunks improve coherence but may cause timeouts and make retries more costly.

**Split strategy.** The tunable `chunk_split_on` selects the main unit: paragraph, sentence, or word. If a paragraph exceeds $C_{\text{max}}$, the algorithm falls back to sentence splitting for that paragraph; if a single sentence still exceeds the limit, it falls back to word splitting for that sentence. Bullet and numbered list lines are treated as sentence units even without terminal punctuation. This avoids oversize chunks while preserving the highest-level structure possible.

**Perturbation guardrail.** When perturbations are enabled (controller overlay or stochastic variance), the system enforces a minimum chunk count $K_{\min}$ so variability has room to appear; the largest chunks are split until $K_{\min}$ is reached or further splitting would be unproductive.

**Rolling summary for semantic continuity (optional).** When enabled, each chunk requests a brief summary of about $S$ words (for example, $S = 25$). The summary is not included in the final output; it is carried forward as a compact semantic thread for the next chunk. The next chunk receives both the prior summary and the new text, then produces a refreshed summary of the combined material. This creates a lightweight semantic memory across chunks without using much token budget. The token budget includes the fixed summary length as part of prompt overhead:

$$T_{\text{base}} = T_{\text{system}} + T_{\text{fingerprint}} + T_{\text{summary}}, \qquad T_{\text{summary}} \approx \alpha S,$$

where $\alpha$ converts words to tokens (about 1–1.3 for plain English).

**Worked example.** Suppose $T_{\text{max}} = 16000$ and the fingerprint plus prompt overhead is $T_{\text{base}} = 9000$. Then $T_{\text{in}} = \max(400, 7000) = 7000$. If `chunking.max_input_tokens=6000`, then $T_{\text{in}} = 6000$ and $C_{\text{max}} \approx 24000$ characters. If variance-aware scaling gives $f = 0.85$, then $T_{\text{in}} = \lfloor 0.85 \cdot 6000 \rfloor = 5100$ tokens ($C_{\text{max}} \approx 20400$ characters). At that point, the number of chunks depends entirely on how much text fits under $C_{\text{max}}$, plus any minimum-chunk enforcement for perturbations.

After rewriting, the system emits the input and output metric profiles (with the aggregate score) for rapid inspection and regression analysis.

#### 2.5.6 API style-match probability (`/rate`)

The local API includes a `rate` method that returns a probability that a text segment is stylometrically consistent with a stored fingerprint. The method intentionally starts from the same explicit local signal stack already used for rewrite retries, then adds a calibrated probabilistic layer.

Let $s \in [0,1]$ be the local style-compliance score from `compute_style_compliance` (histogram and rate divergence against the fingerprint). Distances such as Delta-like stylometric dissimilarities are informative but are not probabilities by themselves [Evert et al., 2017]. Therefore, the API maps $s$ through a logistic calibration:

$$
p_0 = \sigma\left(\frac{s - \tau}{\kappa}\right), \qquad \sigma(z)=\frac{1}{1+e^{-z}}
$$

where:
- $\tau$ is the operating point (default: `style_retry.threshold` from tunables),
- $\kappa$ controls calibration slope.

This follows the same family of score-to-probability calibration used in Platt-style sigmoid mapping and modern calibration practice [Platt, 1999; Guo et al., 2017; scikit-learn calibration docs].

Short segments provide weak stylometric evidence, and authorship attribution reliability drops substantially for short or topically varied samples [Luyckx & Daelemans, 2011; Boenninghoff et al., 2019; ALMs study, 2024]. To reflect this, the API applies evidence-dependent shrinkage toward an uninformative prior of $0.5$:

$$
r = 1 - e^{-n/n_0}, \qquad
p = 0.5 + r \cdot (p_0 - 0.5)
$$

where $n$ is token count in the rated segment and $n_0$ is an evidence scale (default 250 tokens). For very short text ($n \ll n_0$), $r \approx 0$ and $p$ stays near 0.5. For longer text, $r \to 1$ and $p \to p_0$.

The API also returns an uncertainty interval that expands when evidence is weak:

$$
w = 0.5(1-r), \qquad
\mathrm{CI}_{90} = [\max(0,p-w),\ \min(1,p+w)].
$$

This design is practical for operations: it preserves interpretability, keeps all scoring grounded in explicit stylometric features, and emits probabilities in $[0,1]$ aligned with authorship-verification evaluation conventions (including Brier-oriented assessments) [PAN/CLEF AV metrics].

#### 2.5.7 Fingerprint-to-fingerprint similarity (`/similarity`)

The API also supports direct fingerprint comparison (`POST /similarity`) to answer a different question from `/rate`: not "does this text match fingerprint $F$?" but "how similar are fingerprints $F_a$ and $F_b$?" This is useful for profile deduplication, drift monitoring, cohort analysis, and chat trust workflows that compare a baseline user fingerprint with a rolling session fingerprint.

Let the fingerprint similarity be a weighted aggregation over interpretable components:

$$
S(F_a,F_b) = \frac{\sum_{j \in \mathcal{J}_\text{avail}} w_j s_j}{\sum_{j \in \mathcal{J}_\text{avail}} w_j},
\qquad s_j \in [0,1].
$$

Each component score $s_j$ is derived from explicit features already present in the fingerprint JSON:

- **Distribution components (sentence/paragraph histograms, function-word profiles):** use Jensen-Shannon divergence and map similarity as $s=1-\mathrm{JSD}(P,Q)$. JSD is symmetric, bounded, and well-suited for probability distributions [Lin, 1991; Endres & Schindelin, 2003].
- **Rate components (punctuation, stance, rhetoric, syntax, cadence, repetition):** use symmetric relative similarity
$$
s = \frac{1}{1 + \frac{|a-b|}{\max\left(\frac{|a|+|b|}{2},\epsilon\right)}}
$$
which stays bounded and interpretable while avoiding instability near zero.
- **Lexical overlap components (preferred and avoided lexicon):** use Jaccard overlap
$$
s = \frac{|A \cap B|}{|A \cup B|}.
$$

The method returns not only $S$ but also component-level diagnostics (worst-matching components/keys), because operational users need causal hints ("where are these profiles different?"), not only a scalar. This is consistent with stylometric best practice where distance values are interpreted through feature decomposition rather than as opaque truth values [Evert et al., 2017].

To prevent overconfidence when fingerprints are sparse, the API also emits:
- **coverage ratio:** fraction of expected weighted components actually comparable,
- **confidence hint:** a bounded signal combining feature coverage with available corpus size estimates.

This keeps decisions auditable: a low similarity with low coverage should trigger data collection, not hard enforcement.

**Genre-aware quotation handling:** the pipeline auto-detects fiction versus non-fiction using quote-density signals (multi-word quote spans, quoted-word ratio, and quote-paragraph ratio). In non-fiction, multi-word quotations are excluded from profiling and preserved verbatim during rewriting; in fiction, they remain part of the author’s voice. These thresholds are tunable via `config.tunables.json` (see `fiction_detection.*`) and can be overridden explicitly (`--fiction` / `--non‑fiction`).

For orthography, local spelling rules are not applied to preserved quotations in non-fiction: quoted passages are masked before rewriting and reinserted verbatim afterward. In fiction, quoted dialogue is rewritten along with surrounding prose and receives the same local spelling normalization as the rest of the output.

---

## 3. Problem Setup

Let an author corpus be a set of documents

$$\mathcal{D} = \{d_1, \dots, d_N\}, \quad d_i \in \Sigma^*$$

where $\Sigma$ is a character alphabet and $\Sigma^*$ is the set of all finite strings over that alphabet. In practice, each document $d_i$ is a sequence of characters, later tokenized into words and sentences by the measurement layer. The index $N$ is the number of documents in the corpus.

Define:

- An interpretable feature extractor $\phi: \Sigma^* \to \mathbb{R}^K$ that produces measurable statistics (rates, histograms, counts).
- A style fingerprint $\mathcal{F}$ that stores target statistics, distributions, and discrete constraints (lexicon rules, templates).
- A rewriter $\mathcal{R}_\theta$ (LLM with parameters $\theta$) mapping input text $x$ to output $y$:

$$y = \mathcal{R}_\theta(x \mid \mathcal{F}).$$

Here, "$\mid \mathcal{F}$" shows that rewriting is conditioned on the fingerprint: the same input $x$ can produce different outputs depending on the constraints in $\mathcal{F}$. The fingerprint becomes the explicit, versioned control surface for the LLM.

The main constraint is meaning preservation: no new facts, claims, or examples; entities and numerals are retained unless explicitly permitted.

In short, the author’s writing habits are measured and compressed into a JSON fingerprint. The LLM is then tasked with rewriting new text so that these habits are respected, while the underlying meaning remains unchanged.

---

## 4. Stylometric Measurements

This section outlines the simple, interpretable statistics measured by the system. The goal is not linguistic perfection, but stability, explainability, and auditability. These measurements are the ground truth the LLM must follow.

Before describing individual features, it helps to clarify what is counted and what is excluded.

**Tokens, types, and normalization.** The code uses lightweight tokenization (whitespace and punctuation heuristics) to get approximate word tokens. For many stylometric signals, this is enough because the goal is comparative stability across runs, not linguistic annotation. The key step is normalization:

- Counts are scaled per 1000 words (or per 100 words for some densities) so that documents of different lengths are comparable.
- Histograms are normalized to probability distributions so that divergence measures are meaningful.

This approach is interpretable: every reported number can be traced to the frequency of an event and the number of opportunities for that event.

**Author-voice filtering.** Not all text in a corpus reflects the author’s voice. Blockquotes, reference sections, footnotes, inline citations, and boilerplate notices (copyright, terms, privacy) are excluded from measurement and excerpt selection. The pipeline also detects whether a text is likely fiction or non-fiction. In non-fiction, multi-word quoted passages are treated as quotations and excluded from profiling (and preserved during rewriting), while in fiction, quoted dialogue is considered part of the author’s style.

**Why these heuristics are acceptable.** One might ask if these measurements are scientifically pure. They are engineered for a specific purpose. For constraint-guided rewriting, the best measurement is one that is stable, inexpensive to compute, easy to interpret, and aligned with editorial recognition of stylistic drift. Metrics that are too fragile or costly will not be practical.

### 4.1 Rate and Density Features

Let $W(d)$ be an approximate word-token count and $C_e(d)$ the count of an event $e$ (such as commas). Define per-1000-word rates:

$$r_e(d) = 1000 \cdot \frac{C_e(d)}{\max(1, W(d))}.$$

The $\max(1, \cdot)$ guard prevents division by zero on degenerate or heavily filtered text. Normalizing to "per 1000 words" makes rates comparable across documents and corpora of different sizes.

**Worked example (rate normalization).** Suppose a document has $W(d) = 2000$ words and $C_{\text{comma}}(d) = 120$ commas. Then:

$$r_{\text{comma}}(d) = 1000 \cdot \frac{120}{2000} = 60\ \text{commas per 1000 words}.$$

The fingerprint stores targets as tolerance intervals:

$$r_e \in [\underline{r}_e, \overline{r}_e],$$

reflecting intra-author variability across topics and subgenres. The interval shows that the author’s comma rate usually falls within these bounds, so a rewrite that deviates a lot is likely drifting stylistically (or compressing or expanding the prose).

### 4.2 Histogram Features

For sentence lengths $\ell_1, \dots, \ell_m$ (in words), define a binned histogram

$$\mathbf{h} \in \Delta^{B-1}, \quad h_b = \frac{1}{m} \sum_{i=1}^m \mathbf{1}[\ell_i \in \text{bin}(b)],$$

where $\mathbf{1}[\cdot]$ is an indicator function (1 if the condition holds, 0 otherwise), $\Delta^{B-1}$ is the probability simplex (all nonnegative vectors summing to 1), and bins are ordinal intervals (for example, $<10$, 10–17, 18–25, 26–40, $>40$). The system counts how many sentences fall into each length bin and then normalizes by the total number of sentences $m$.

**Worked example (histogram).** Suppose $m = 5$ sentence lengths (in words): $\ell = [8, 12, 12, 23, 41]$, and bins are $<10$, 10–17, 18–25, 26–40, $>40$. Then the histogram mass is:

- $h_{<10} = 1/5$ (one sentence of length 8),
- $h_{10-17} = 2/5$ (two sentences of length 12),
- $h_{18-25} = 1/5$ (one sentence of length 23),
- $h_{26-40} = 0$,
- $h_{>40} = 1/5$ (one sentence of length 41).

Paragraph rhythm is also measured using the one-sentence paragraph rate:

$$\rho_{1}(d) = \frac{\#\{\text{paragraphs with exactly one sentence}\}}{\max(1, \#\{\text{paragraphs}\})}.$$

This rate serves as a stylistic baseline. Too many one-sentence paragraphs are flagged as an AI artifact only if they exceed the author’s $\rho_1$ range. One-sentence paragraphs are not inherently problematic; they are a stylistic choice. The fingerprint makes this choice measurable so the system can tell authorial habit from LLM overuse.

**Worked example (one-sentence paragraph rate).** If a document contains 10 paragraphs and 4 have exactly one sentence, then $\rho_1 = 4/10 = 0.4$.

### 4.3 Rare-Word Signals

Let $f(w)$ be the corpus frequency of a token $w$ after filtering stopwords, numerals, and short tokens. A rare-word list is recorded:

$$\mathcal{R} = \{w : f(w) \le c_{\max}\},$$

where $c_{\max}$ is a small threshold (for example, 2–5 occurrences). This set is not a ban list; it is a cue. If a word appears only once in a large corpus, frequent use in a rewrite may sound unlike the author even if it is semantically correct. Domain-specific corpora may legitimately contain rare technical terms; the system treats the list as a hint and allows overrides.

### 4.4 Rhetorical and Epistemic Signals

Beyond surface statistics, the system tracks interpretable rhetorical moves and certainty bands. Let $\mathcal{S}$ be sentences in the corpus. For a rhetorical marker set $\mathcal{M}_k$ (such as claim or concession indicators), define:

$$r_k = 1000 \cdot \frac{\#\{s \in \mathcal{S} : s \text{ contains any marker in } \mathcal{M}_k\}}{\max(1, W)}.$$

Here $W$ is the corpus (or chunk) word count used for normalization. Rates are computed for claim, evidence, counterpoint, concession, and synthesis markers. Epistemic stance bands (speculative, probabilistic, assertive, directive) are calculated using simple token lists. These signals are approximate; they set tolerances, not perfect classifications.

The system measures rhetorical tendencies - specifically, how often particular moves appear - rather than doing full discourse parsing.

### 4.5 Paragraph Cadence and Discourse Marker Position

Let $s_1$ and $s_n$ be the opening and closing sentences of a paragraph.

The system measures the distributions of opening and closing sentence lengths (means and standard deviations) to capture cadence. It also records the position of discourse markers (like "however" or "therefore") as start-of-sentence versus mid-sentence rates:

$$r_{\text{start}} = 1000 \cdot \frac{\#\{\text{markers at sentence start}\}}{\max(1, W)}, \quad r_{\text{mid}} = 1000 \cdot \frac{\#\{\text{markers mid‑sentence}\}}{\max(1, W)}.$$

These features are intended to preserve the typical placement of transitions in an author's style. Many writers have strong positional habits - for example, "However," at the start of a sentence versus "..., however, ..." mid-sentence. In practice, maintaining these positions often matters more than the exact marker choice.

### 4.6 Repetition Signals (Self‑Echo)

AI-generated text often repeats phrases locally. The system measures this by checking repetition rates for bigrams and trigrams:

$$\rho_n = \frac{\sum_{g \in \mathcal{G}_n} \mathbf{1}[c(g) \ge c_{\min}] \cdot c(g)}{\max(1, |\mathcal{G}_n|)},$$

where $\mathcal{G}_n$ is the multiset of n-grams and $c_{\min}$ is a small repeat threshold (default 3). The numerator counts repeated n-gram occurrences beyond the threshold, and the denominator normalizes by the total number of n-grams. These rates set ceilings for acceptable self-echo. In other words, too much repetition of bigrams or trigrams makes output sound templated.

### 4.7 Delta-Style Diagnostics (Optional)

Stylometric-Transfer is not an authorship attribution system. However, Delta-style distances serve as diagnostic measures of stylistic proximity. After standardization and Manhattan-style aggregation:

$$\Delta(d, d') = \frac{1}{K} \sum_{k=1}^K \left|z_k(d) - z_k(d')\right|,$$

where $z_k$ is the z-transformed version of feature $k$ (subtracting the corpus mean and dividing by the corpus standard deviation). The absolute difference is averaged across features, producing a distance-like score: smaller values indicate greater similarity under these normalized features. Explanations of Delta variants motivate this approach ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com)).

---

## 5. The Style Fingerprint as a Constraint Model

A fingerprint is handled as a set of weighted constraints.

$$\mathcal{F} = \{(\psi_j, \mathcal{C}_j, w_j)\}_{j=1}^J,$$

Here, $\psi_j(y)$ is a measurable statistic of the output (such as comma rate or a histogram vector), $\mathcal{C}_j$ is an admissible set (for example, a range or a forbidden list), and $w_j$ is a weight indicating priority.

This notation makes the engineering intent clear. Each constraint consists of a measurement ($\psi_j$), an admissible region ($\mathcal{C}_j$), and a weight ($w_j$). The system's goal is to find an output $y$ whose measurements fall within as many admissible regions as possible, with the highest-weighted constraints satisfied most reliably.

Typical constraint types are:

1. Range constraints: $\psi_j(y) \in [a,b]$
2. Histogram constraints: $D(\mathbf{h}^*, \mathbf{h}(y)) \le \tau$
3. Lexicon constraints: forbidden phrases or words, preferred synonyms, avoid-rare words $\mathcal{R}$
4. Template constraints: rhetorical move frequency bounds

The JSON format includes practical control fields, such as `priority_order` and `strictness`, to specify constraint precedence.

In plain terms, the fingerprint acts as a weighted checklist. Some items are strict (such as "never use em-dashes"), while others are soft (such as "prefer shorter sentences"). The system tracks adherence to each requirement.

Constraint normalization and controller overlays keep constraints compact and interpretable. The system deterministically de-duplicates `rewrite_policy` clauses and filters `priority_order` to short, token-like entries. A corpus-derived variability baseline can drive per-chunk target overlays during rewriting. A deterministic controller samples quantiles of within-author variability (such as sentence length or punctuation density) and nudges chunk-level targets so the output reflects natural intra-author dispersion without altering the core fingerprint. When perturbations are enabled (controller overlays or stochastic variance), the pipeline enforces a minimum chunk count so variability can be expressed. These overlays are applied locally and logged for audit.

---

## 6. Constraint Satisfaction Decoding and Compliance Scoring

During generation, the language model produces a candidate rewrite, which is immediately measured using the same metrics applied to the author’s corpus. This closes the loop: if the output diverges, the model receives precise feedback on which metrics have drifted and can be prompted for correction.

This section develops the mathematical framing of rewriting as a constraint satisfaction problem.

### 6.1 Soft-Constrained Objective

Let $p_\theta(y\mid x)$ denote the model’s conditional probability of output $y$ given input $x$. The soft-constrained objective is:

$$\max_{y \in \mathcal{Y}} \; \log p_\theta(y \mid x) - \lambda\, \mathcal{L}_{style}(y;\mathcal{F}) - \mu\,\mathcal{L}_{sem}(y;x),$$

where $\mathcal{L}_{style}$ penalizes deviation from the style fingerprint and $\mathcal{L}_{sem}$ penalizes semantic drift (estimated conservatively via invariants or, optionally, semantic similarity models).

Here, $\log p_\theta(y\mid x)$ reflects the model’s fluency preference, while $\mathcal{L}_{style}$ and $\mathcal{L}_{sem}$ are penalties that pull the output towards the desired style and meaning. The hyperparameters $\lambda$ and $\mu$ control the trade-off between what the model would naturally say and what is required.

A typical decomposition is:

$$\mathcal{L}_{style}(y;\mathcal{F}) = \sum_{j=1}^J w_j\, \ell_j(\psi_j(y), \mathcal{C}_j).$$

This expresses style loss as a weighted sum of per-constraint penalties. If a particular constraint is crucial - such as orthography, hard avoids, or a tight sentence-length range - it receives higher weight and dominates the loss.

Examples of penalties include:

**Range penalty** for $\mathcal{C}_j=[a,b]$:

$$\ell_j(v,[a,b]) = \big(\max(0,a-v)\big)^2 + \big(\max(0,v-b)\big)^2.$$

This penalty is zero when $v$ is inside $[a,b]$. If $v$ falls below $a$, it grows quadratically with the gap $(a-v)$; if it exceeds $b$, it grows quadratically with $(v-b)$. Quadratic growth discourages large violations while remaining gentle near the boundary.

**Worked example (range penalty):** If the target interval is $[a,b]=[5,8]$ and the observed value is $v=3$, then:

$$\ell(v,[a,b])=(5-3)^2+0=4.$$

If $v=9$, then:

$$\ell(v,[a,b])=0+(9-8)^2=1.$$

**Histogram penalty** using KL divergence:

$$\ell_j(\mathbf{h},\mathbf{h}^{*}) = D_{KL}(\mathbf{h}^{*}\|\mathbf{h}) = \sum_{b=1}^B h_b^{*} \log \frac{h_b^{*}}{\max(\epsilon,h_b)}.$$

Here $\mathbf{h}^*$ is the target histogram from the fingerprint and $\mathbf{h}(y)$ is the output histogram. KL divergence is asymmetric; it quantifies how surprising the target distribution is under the output distribution. The $\epsilon$ guard prevents division by zero when a bin is empty. Other divergences (such as Jensen–Shannon or Wasserstein) may be substituted if ordinal bin geometry matters.

**Worked example (KL on histograms):** Let the target be $\mathbf{h}^*=[0.5,0.5]$ and the output be $\mathbf{h}=[0.9,0.1]$ over two bins. Then:

$$D_{KL}(\mathbf{h}^*\|\mathbf{h}) = 0.5\log\frac{0.5}{0.9} + 0.5\log\frac{0.5}{0.1}.$$

The first term is negative (the output over-allocates mass to bin 1), while the second is strongly positive (the output under-allocates bin 2). The sum is positive, indicating that the output distribution does not match the target well.

For ordinal bins, the Wasserstein distance $W_1$ may be preferable; the implementation can support either approach.

### 6.2 Hard Constraints (Feasibility)

Some constraints are best enforced as hard feasibility requirements:

- Entity and number preservation constraints must be satisfied unless explicitly overridden.
- Hard forbidden lexicon constraints (for example, terms that must not appear).

The feasible set is defined as:

$$\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y\in \mathcal{Y} : \forall j\in \mathcal{H},\; \psi_j(y)\in \mathcal{C}_j\},$$

where $\mathcal{H}\subseteq\{1,\dots,J\}$ indexes the hard constraints.

In other words, $\mathcal{Y}_{hard}$ is the subset of all possible rewrites that satisfy every non-negotiable rule. If meaning preservation forbids altering a quoted passage, any rewrite that changes it is outside the feasible set. This expresses, mathematically, that some rules are deal-breakers.

Decoding then becomes:

$$\max_{y\in \mathcal{Y}_{hard}(x,\mathcal{F})} \log p_\theta(y\mid x) - \lambda\sum_{j\notin \mathcal{H}} w_j\,\ell_j(\psi_j(y),\mathcal{C}_j) - \mu\,\mathcal{L}_{sem}(y;x).$$

This is the same soft-constrained objective as before, but restricted to feasible outputs. In implementations that cannot perform true constrained decoding, the system approximates this restriction via prompt instructions, frozen placeholders, and post-hoc repairs.

### 6.3 Practical Constraint-Satisfaction Decoding Procedure

Exact constrained decoding over $\mathcal{Y}_{hard}$ is rarely available in production language models. Stylometric-Transfer approximates constraint satisfaction through instruction prompting, self-audit, and repair.

A practical approximation proceeds as follows:

1. Generate a candidate rewrite $y^{(0)}$ using explicit instructions that encode $\mathcal{F}$.
2. Compute local measurements $\phi(y^{(t)})$ and audit for constraint violations.
3. If violations are detected, re-prompt the model with a structured report to obtain $y^{(t+1)}$.
4. Stop when compliance exceeds a threshold or an iteration limit is reached.

### 6.4 Compliance Scoring

A normalized compliance score $S(y;\mathcal{F})\in[0,1]$ aggregates constraint satisfaction:

$$S(y;\mathcal{F}) = \sigma\Big(\sum_{j=1}^J w_j\, s_j(y)\Big), \quad \sum_j w_j = 1,$$

where $\sigma$ is a squashing function (such as the identity clipped to $[0,1]$ or a logistic function), and $s_j(y)\in[0,1]$ is a per-constraint score.

The normalization $\sum_j w_j=1$ makes the weighted sum interpretable: it becomes an average of per-constraint scores. The squashing function $\sigma$ is optional; it can be used to make the score more sensitive to low compliance (logistic) or simply to clip numerical noise (identity plus clipping).

**Worked example (weighted compliance):** Suppose three constraint scores $s=[0.9, 0.7, 0.8]$ are tracked with weights $w=[0.5, 0.2, 0.3]$. The weighted sum is:

$$\sum_j w_js_j = 0.5\cdot 0.9 + 0.2\cdot 0.7 + 0.3\cdot 0.8 = 0.83.$$

If $\sigma$ is identity-with-clipping, then $S=0.83$. This reads as "83% compliant under these weighted checks."

Examples include:

- **Range score**:
$$s_j(y) = 1 - \min\left(1, \frac{\ell_j(\psi_j(y),[a,b])}{\kappa_j}\right)$$
for a scaling constant $\kappa_j>0$.

Here $\kappa_j$ turns a raw penalty into a unitless score: if the penalty equals $\kappa_j$, the score reaches 0. This allows different constraint families (rates, divergences, boolean checks) to be combined into a single compliance number without one metric dominating due to scale.

- **Histogram score** (KL):
$$s_j(y) = \exp\big(-\alpha_j\, D_{KL}(\mathbf{h}^*\|\mathbf{h}(y))\big).$$

The exponential mapping converts a divergence into a score: small divergences yield scores near 1, while large divergences decay smoothly towards 0. The parameter $\alpha_j$ controls how sharply the score drops.

- **Lexicon hard constraint score**:
$$s_j(y)=\mathbf{1}[\text{no forbidden term appears}].$$

This is the simplest score: either the forbidden term appears (0) or it does not (1). Hard constraints are typically treated as boolean because any violation is unacceptable. This compliance score supports reporting (via `validators.weights` and `checks` in JSON), iterative repair thresholds, and regression tests for stability.

---

## 7. Implementation Notes (Stylometric-Transfer)

This section connects the theoretical framework to the codebase. Readers seeking a practical overview may treat it as an annotated guide to system operation; experts may regard it as an implementation-level specification.

The repository provides:

1. **Local measurement stage**
   - Sentence-length histogram
   - Paragraph-length histogram
   - Punctuation rates per 1,000 words
   - Contraction, dash, and ellipsis signals
   - Frequent n-grams (diagnostic lexicon hints)

2. **LLM synthesis stage**
   - Schema-guided, JSON-only prompting
   - Embedding of measurements verbatim
   - Automated JSON repair if parsing fails
   - Deterministic de-duplication of verbose `rewrite_policy` clauses and token filtering for `priority_order`

3. **Rewrite stage**
   - Fingerprint, input measurements, and markdown text
   - JSON output: rewritten markdown, deviations, and self-check

These design choices reflect stylometric traditions that favour interpretable features, as well as concerns about evaluation and ethical risk in text style transfer ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com)).

---

## 8. Ethical Considerations

Transparency and non-impersonation are central. The system is intended for personal writing, editing support, and self-modelling, not for imitating living authors without consent.

Stylometric-Transfer could be misused for impersonation; recent surveys highlight ethical risks and the need for safeguards ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com)).

Intended uses include:
- Self-authored corpora
- Licensed or public-domain corpora
- Editing support and personal voice consistency

Recommended safeguards:
- Provenance tracking in `metadata`
- Default controls that discourage third-party imitation
- Deviation reporting when constraints conflict with meaning preservation

---

## 9. Limitations and Failure Modes (Stylometry + Humanization)

This project is intentionally pragmatic. It aims for auditable control rather than maximum imitation fidelity. That design yields predictable strengths (interpretability, editability, local grounding) and equally predictable limitations. This section summarizes the important ones, with emphasis on practical risks.

### 9.1 Stylometry Limitations

**Genre and domain dependence.** Stylometric signals are often genre constraints rather than "author essence." A newspaper leader, a technical manual, and a literary short story produce different punctuation rates, sentence distributions, and function-word profiles - even for the same writer. If the input is out-of-domain relative to the corpus used to build the fingerprint, the system can only satisfy some constraints without threatening meaning preservation. In practice, this leads to more deviations, more retries, or a conservative rewrite.

**Topic and named-entity leakage.** Frequent n-grams and "common phrases" are especially vulnerable to domination by topical entities (people, places, organizations). Deterministic filtering and blacklist strategies reduce this, but cannot eliminate the underlying issue: a corpus can be stylistically consistent yet topically narrow. The solution is not more filtering alone; it is also broader corpora (within the same genre) or a higher-level representation of lexical preference (for example, conceptual rather than named entities).

**Sparse features and small corpora.** Rare rhetorical moves, rare punctuation events (such as semicolons), and "rare words" require sufficient data to stabilize. For small corpora, estimates are noisy and can lead to brittle targets. The project mitigates this by favouring ranges and histograms over point targets, but sparse phenomena remain difficult.

**Language and tokenization assumptions.** The measurement layer is intentionally lightweight; it uses heuristic sentence and paragraph splitting and simple token counting. This is robust for plain English prose but less reliable for mixed scripts, heavy mathematics, tables, code-dense documents, or languages with different punctuation conventions. Where segmentation fails, all downstream metrics (and therefore targets) inherit the error.

**Interpretability versus completeness.** "Explicit and simple" measurements omit many subtle signals: syntax trees, discourse relations, register shifts, pragmatic implicatures, and long-range narrative structure. These omissions are deliberate for auditability, but they limit what can be captured quantitatively.

### 9.2 Humanization Limitations

**Humanization is not a single axis.** "Human-like" prose is not a stable target: authors differ, genres differ, and even within one document the distribution shifts (opening versus closing, dialogue versus exposition, summary versus detail). Any single scalar score necessarily compresses nuance and should be treated as a monitoring signal, not an optimization objective.

**Metric gaming and Goodhart’s law.** If an operator tunes prompts or post-processing to maximize the aggregate score $S$, the system may learn to "game" the proxies (for example, add synonym churn to boost diversity, or inject punctuation to raise entropy) without improving the text. The project defends against this by keeping meaning preservation and structural invariants primary, logging per-metric diagnostics, and bounding perturbations. Nonetheless, the risk is intrinsic to proxy scoring.

**Local edits can cause global drift.** Micro-operations and controller overlays are designed to be small, but they can interact: a swap of transition words can change sentence boundaries; punctuation edits can change clause structure; a change in paragraphing can affect cadence metrics. The system therefore keeps the perturbation budget low and uses chunk-level overlays rather than global optimization, but complex interactions remain possible.

**Chunk boundary artifacts.** Chunking is essential for reliability, but it can introduce boundary effects: repeated openers at chunk starts, inconsistent local voice, or over-compression in smaller chunks. Variance-aware chunking and minimum-chunk enforcement mitigate this, but there is no free lunch: smaller chunks improve reliability, larger chunks improve coherence.

**Quotations are ambiguous.** The fiction versus non-fiction heuristic treats multi-word quotes as quotations in non-fiction, but real documents contain mixed modes (quoted slogans, reported speech, epigraphs, and stylized "air quotes"). The system allows explicit overrides, but automatic classification will sometimes be incorrect.

**Hard hygiene rules can be stylistically incorrect.** Deterministic bans (such as "no em dashes") increase consistency across outputs, but they can conflict with the author’s authentic style. The project treats these as optional tunables because they are editorial choices, not stylometric truths.

### 9.3 LLM and Systems Limitations

**Model variability and endpoint reliability.** Even with fixed prompts, LLMs can be nondeterministic (sampling, backend variation), and endpoints can be slow or unreliable (timeouts, transient 5xx). The pipeline uses retries and bounded refinement loops, but it cannot guarantee that every request succeeds within a fixed time budget.

**Meaning preservation is not formally verified.** The system enforces meaning preservation primarily through prompting, structural freezing, and deviation reporting. It does not yet provide a formal semantic equivalence proof. As a result, human review remains necessary for high-stakes text.

**Evaluation remains multi-objective.** The system optimizes for multiple goals: meaning preservation, adherence to explicit style constraints, reduced AI-typical artifacts, and Markdown validity. These goals sometimes conflict. The project’s design makes those conflicts explicit (via priorities and deviations), but it cannot eliminate them.

**Comparison context.** Many production-grade pipelines combine LLMs with training-time specialization, agentic critique loops, or detector-guided objectives. Stylometric-Transfer deliberately prioritizes auditability and explicit control; Appendix F provides a structured comparison against common "best-of-class" alternatives and where each tends to win.

---

## 10. Conclusion

For newcomers, the main point is that classic stylometry and modern language models can be combined without sacrificing interpretability. For experts, the contribution is a concrete, auditable constraint model and a measurable conflict-resolution layer unifying stylometric transfer and humanization.

Stylometric-Transfer connects classic stylometry and LLM-based rewriting by pairing interpretable, versionable style models with constraint-guided generation. The explicit JSON fingerprint enhances auditability and editorial control, drawing on established stylometric measurement and style transfer research ([press.uchicago.edu](https://press.uchicago.edu/ucp/books/book/distributed/I/bo5667096.html?utm_source=chatgpt.com)).

---

## References

- Mosteller, F., & Wallace, D. L. *Inference and Disputed Authorship: The Federalist.* Addison-Wesley (1964). ([archive.org](https://archive.org/details/inferencedispute00most?utm_source=chatgpt.com))
- Evert, S., et al. "Understanding and explaining Delta measures for authorship attribution." *Digital Scholarship in the Humanities* (2017). ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))
- Lin, J. "Divergence Measures Based on the Shannon Entropy." *IEEE Transactions on Information Theory* (1991). ([doi.org](https://doi.org/10.1109/18.61115))
- Endres, D. M., & Schindelin, J. E. "A new metric for probability distributions." *IEEE Transactions on Information Theory* (2003). ([doi.org](https://doi.org/10.1109/TIT.2003.813506))
- Platt, J. "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." (1999). ([researchgate.net](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods))
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On Calibration of Modern Neural Networks." *ICML/PMLR* (2017). ([proceedings.mlr.press](https://proceedings.mlr.press/v70/guo17a.html))
- scikit-learn documentation, "Probability calibration" (sigmoid/Platt and isotonic calibration notes). ([scikit-learn.org](https://scikit-learn.org/stable/modules/calibration.html))
- PAN/CLEF Authorship Verification task (evaluation with probability-like scores and Brier-oriented metrics). ([pan.webis.de](https://pan.webis.de/clef23/pan23-web/author-identification.html))
- Luyckx, K., & Daelemans, W. "The Effect of Author Set Size and Data Size in Authorship Attribution." *Literary and Linguistic Computing* (2011). ([academic.oup.com](https://academic.oup.com/dsh/article-abstract/26/1/35/982424))
- Boenninghoff, B., et al. "Cross-Domain Authorship Attribution Combining Instance-Based and Profile-Based Features." *CLEF/PAN* (2019). ([arxiv.org](https://arxiv.org/abs/1912.03363))
- ALMs study: "Author style can be measured with local metrics over short segments (20–400 tokens)." (2024). ([arxiv.org](https://arxiv.org/abs/2401.12005))
- Shen, T., Lei, T., Barzilay, R., & Jaakkola, T. "Style Transfer from Non-Parallel Text by Cross-Alignment." (2017). ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com))
- Mukherjee, S., et al. "A Survey of Text Style Transfer: Applications and Ethical Implications." (2024). ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com))
- Hu, Z., et al. "Text Style Transfer: A Review and Experimental Evaluation." *KDD Explorations* (PDF). ([kdd.org](https://www.kdd.org/exploration_files/vol24issue1_2._Text_Style_Transfer__A_Review_and_Experimental_Evaluation.pdf?utm_source=chatgpt.com))

---

## Appendix a. Methods (Pseudocode)

This appendix presents pseudocode for the fingerprinter (extractor) and rewriter stages.

This document is intended for readers unfamiliar with stylometry, providing a procedural overview of the system.

### A.1 Fingerprint Extraction (Corpus → Style Fingerprint JSON)

**Inputs:** Corpus archive $A$, language model $\mathcal{R}_\theta$, schema template $S$  
**Output:** Style fingerprint $\mathcal{F}$ (JSON)

```text
procedure FINGERPRINT_STYLE(archive A, output_path out, llm_config C):
    tmp_dir ← extract_archive(A)
    files ← list_textlike_files(tmp_dir, extensions={.txt,.md,.rst,.html,.docx})
    texts ← []
    for f in files:
        t ← read_and_normalize(f)
        t ← normalize_ocr(t)  # ligatures, hyphenated line breaks
        t ← strip_base64_images(t)
        t ← filter_non_voice(t)  # blockquotes, references, footnotes, inline citations, boilerplate; drop multi-word quotes if non-fiction
        if length(t) ≥ MIN_LEN:
            texts.append(t)

    M ← compute_measurements(texts)
        # includes sentence and paragraph histograms, punctuation rates, contractions and oxford comma, function words, stance signals, sentence-openers/templates, n-grams

    if phrase_validation_enabled:
        V ← prefilter_proper_names(M.common_phrases)
        V ← validate_common_phrases(V, llm=C)
        M.common_phrases_validation ← V

    E ← pick_representative_excerpts(files, char_budget=B, voice_scoring=on)
    L ← load_lexicon_hints(optional)

    prompt ← build_fingerprint_prompt(schema=S, measurements=M, excerpts=E, lexicon_hints=L, model=C.model)
    if prompt_too_large(prompt, C.max_prompt_tokens):
        batches ← chunk_excerpts(E, budget=C.max_prompt_tokens)
        partials ← []
        for b in batches:
            prompt_b ← build_fingerprint_prompt(schema=S, measurements=M, excerpts=b, lexicon_hints=L, model=C.model)
            raw_b ← call_llm_chat_completions(prompt_b, C)
            partials.append(parse_or_repair(raw_b))
        F ← partials[0]
        for p in partials[1:]:
            merge_prompt ← build_merge_prompt(F, p, schema=S, measurements=M)
            raw_m ← call_llm_chat_completions(merge_prompt, C)
            F ← parse_or_repair(raw_m)
        goto enforce_invariants
    raw ← call_llm_chat_completions(prompt, C)

    if is_valid_json(raw):
        F ← parse_json(raw)
    else:
        repair_prompt ← build_json_repair_prompt(raw)
        raw2 ← call_llm_chat_completions(repair_prompt, C)
        F ← parse_json(raw2)

    enforce_invariants:
    F.schema_version ← default_if_missing(F.schema_version, "1.0.0")
    F.measurements ← M

    write_json(out, F)
    return F
end procedure
```

Phrase validation is conducted in two stages: an initial deterministic filter (using heuristics based on honorifics and capitalization), followed by an LLM-based ranking that deprioritizes probable proper names prior to final selection.

Rare-word selection may also be ranked by the same LLM validation, with capitalization ratios used to reduce the prominence of proper names before truncation.

### A.2 Rewrite (Fingerprint Draft Styled Draft Styled Draft)

**Inputs:** Fingerprint $\mathcal{F}$, input Markdown $x$, language model $\mathcal{R}_\theta$  
**Output:** Rewritten Markdown $y$ and deviations report

```text
procedure APPLY_FINGERPRINT(fingerprint F, markdown_path in, output_path out, llm_config C):
    x ← read_text(in)
    x ← strip_base64_images(x)
    x ← mask_non_voice_blocks(x)  # blockquotes, references, footnotes, and multi-word quotations if non-fiction
    x ← mask_inline_citations(x)
    Mx ← compute_measurements(filter_non_voice(x))
    Hraw ← load_humanizer_guidelines(optional)
    H ← parse_humanizer_rules_llm(Hraw)
    if H is empty:
        H ← parse_humanizer_rules_regex(Hraw)
    H ← filter_conflicting(H, F.measurements, F.targets)

    style_feedback ← null
    for r in 0..R:
        prompt ← build_rewrite_prompt(fingerprint=F, input_measurements=Mx, input_text=x, style_feedback=style_feedback, humanizer_guidelines=H)
        raw ← call_llm_chat_completions(prompt, C)

        if is_valid_json(raw):
            obj ← parse_json(raw)
        else:
            obj ← parse_json(call_llm_chat_completions(build_json_repair_prompt(raw), C))

        y ← obj.final_markdown
        y ← restore_placeholders(y)
        audit ← style_compliance(F.measurements, y)
        if audit.score ≥ τ or r == R:
            break
        style_feedback ← audit.deltas

    deviations ← obj.deviations ∪ {style_compliance: audit}

    write_text(out, y)
    if deviations not empty:
        write_json(out + ".deviations.json", deviations)

    return y
end procedure
```

### A.3 Optional: Audit-and-Repair Loop

```text
procedure REWRITE_WITH_ITERATIVE_REPAIR(fingerprint F, input x, llm_config C, max_iters T):
    y ← call_llm_rewrite(F, x, C)

    for t in 1..T:
        audit ← style_compliance(F.measurements, y)
        if audit.score ≥ τ and audit.hard_violations == 0:
            break
        y ← call_llm_repair(F, x, y, audit.deltas, C)

    return y
end procedure
```

### A.4 Worked Examples

This section presents direct calculations for verification.

**A.4.1 Punctuation rate per 1000 words.**

For a filtered segment of 500 words with 18 semicolons:

$$r_{\text{semicolon}} = 1000\cdot\frac{18}{500} = 36\ \text{per 1000 words}.$$

**A.4.2 One-sentence paragraph rate.**

For 20 paragraphs, 7 of which have exactly one sentence:

$$\rho_1 = 7/20 = 0.35.$$

**A.4.3 Controller overlay band.**

If the baseline quantile gives $v=5.0$ for comma density per 100 words, and `range_pct=0.15`, `min_width=0.05`, `max_width=6.0` are satisfied:

$$w=\max(0.05, |5.0|\cdot 0.15)=0.75,$$

so the target band is $[4.25,5.75]$ for that chunk.

**A.4.4 Burstiness.**

If sentence lengths are $\ell_s=[10, 10, 30]$, then $\mu=16.67$ and (population) $\sigma\approx 9.43$, so:

$$B_s=\sigma/\mu \approx 0.57.$$

---

## Appendix B. Formal Constrained Decoding

This section reframes decoding as a constrained optimization problem or Markov decision process. For those unfamiliar with the formalism, the main point is that the LLM is directed by measurable constraints, not latent embeddings.

### B.1 Constrained Maximum a Posteriori Decoding

Let $p_\theta(y\mid x)$ be the base LLM distribution. Constraints are indexed by $j=1,\dots,J$, each with statistics $\psi_j(y)$ and admissible sets $\mathcal{C}_j$.

The feasible set of hard constraints:

$$\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y : \forall j \in \mathcal{H},\; \psi_j(y) \in \mathcal{C}_j\}$$

The constrained MAP problem:

$$\hat y = \arg\max_{y \in \mathcal{Y}_{hard}(x,\mathcal{F})} \; \log p_\theta(y\mid x)$$

In practice, $\mathcal{Y}_{hard}$ cannot be enumerated directly. The problem is relaxed using a Lagrangian penalty:

$$\hat y = \arg\max_{y \in \mathcal{Y}} \; \log p_\theta(y\mid x) - \sum_{j=1}^J \lambda_j \cdot g_j(\psi_j(y)) - \mu \cdot \mathcal{L}_{sem}(y;x),$$

where $g_j(\cdot)$ is a non-negative violation function ($g_j(v)=0$ if and only if $v \in \mathcal{C}_j$), $\lambda_j \ge 0$ are Lagrange multipliers (from `validators.weights`), and $\mathcal{L}_{sem}$ enforces meaning preservation.

This approach is consistent with soft-constrained decoding in controllable generation.

---

### B.2 Projection View

Alternatively, rewriting can be seen as projecting an unconstrained sample $y^{(0)} \sim p_\theta(\cdot \mid x)$ onto the admissible region:

$$\hat y = \Pi_{\mathcal{C}}(y^{(0)}) = \arg\min_{y} \; d(y, y^{(0)}) + \sum_j \lambda_j g_j(\psi_j(y)),$$

where $d(\cdot,\cdot)$ measures edit or semantic divergence. In practice, $\Pi_{\mathcal{C}}$ is approximated by LLM self-repair passes, each guided by explicit audit reports.

---

### B.3 Constrained Markov Decision Process (CMDP)

Token generation can be framed as a CMDP:

- States: $s_t = y_{1:t}$
- Actions: $a_t = y_{t+1}$
- Reward: $r_t = \log p_\theta(a_t\mid s_t,x)$
- Costs: $c_{j,t}$, which accumulate toward $\psi_j(y)$

with terminal constraints:

$$\mathbb{E}\Big[ \sum_t c_{j,t} \Big] \le \tau_j$$

This clarifies that the system approximates policy optimization under global style budgets, implemented through instruction-guided generation and post-hoc repair.

---

## Appendix C. Evaluation and Acceptance Criteria

This section specifies divergence metrics and acceptance thresholds, each mapped to a corresponding fingerprint JSON field. Each metric aligns with a specific JSON field.

### C.1 Metric Families

#### (1) Rate constraints

Given a target interval $[a,b]$ and observed value $v$:

$$\text{viol}_r(v) = \max(0,a-v) + \max(0,v-b)$$

Score:

$$s_r(v) = \exp(-\alpha_r \cdot \text{viol}_r(v))$$

Mapped JSON paths: - `/targets/punctuation/comma_density_per_100w` - `/targets/orthography/contractions_rate`

---

#### (2) Histogram constraints

The main metric is the L1 histogram distance:

$$d_h(\mathbf{h}^*, \mathbf{h}) = \\frac{1}{2} \sum_{b=1}^{B} |h_b - h_b^*|$$

Score:

$$s_h = 1 - \min(1, d_h)$$

Mapped JSON: - `/targets/sentence/length_words/distribution` - `/targets/paragraph/length_sentences`

---

#### (3) Lexicon constraints

Hard constraints:

$$s_{lex}^{hard} = \mathbf{1}[\text{no forbidden term appears}]$$

Soft constraints:

$$s_{lex}^{soft} = \exp(-\alpha_{lex} \cdot |f_y - f^*|)$$

Mapped JSON: - `/lexicon/avoid_words` - `/lexicon/avoid_phrases` - `/lexicon/preferred_phrases`

---

#### (4) Function-word and stance signals

For each rate signal (such as `hedge_rate`, `first_person_rate`), the relative deviation is:

$$d_s(v, v^*) = \\frac{|v - v^*|}{\\max(|v^*|, 1)}$$

Score:

$$s_s = 1 - \min(1, d_s)$$

Mapped JSON: - `/measurements/function_words` - `/measurements/stance_signals`

---

### C.2 Aggregated Compliance Score

Let weights $w_j$ be specified in `validators.weights` with $\sum_j w_j = 1$.

$$S(y;\mathcal{F}) = \sum_{j=1}^J w_j s_j(y)$$

If weights are not provided, use the unweighted mean:

$$S(y;\mathcal{F}) = \\frac{1}{J} \sum_{j=1}^J s_j(y)$$

Acceptance levels:

| Level | Condition |
|------|-----------|
| **Pass** | $S \ge 0.75$ and no hard violations |
| **Warn** | $0.60 \le S < 0.75$ |
| **Fail** | $S < 0.60$ or any hard violation |

Mapped JSON: - `/validators/scoring/overall_threshold/pass` - `/validators/scoring/overall_threshold/warn`

---

### C.3 Field-Level Thresholds (Defaults)

| Field family | Metric | Threshold |
|-------------|--------|-----------|
| Sentence histogram | $W_1$ | $\le 0.08$ |
| Paragraph histogram | $W_1$ | $\le 0.10$ |
| Punctuation rates | relative error | $\le 20\%$ |
| One-sentence paras | abs diff | $\le 0.05$ |
| Exclamations | hard max | must satisfy |
| Forbidden lexicon | indicator | must satisfy |

----

### C.4 Iterative Repair Stopping Rule

Let $S_t$ be the score at iteration $t$. The process terminates when:

$$S_t \ge S_{pass} \quad \text{and} \quad H_t = 0$$

Otherwise, continue for up to $T_{max}$ repair passes.

---

## Appendix D. Mechanism of Fingerprint-Conditioned Rewriting

This section outlines how an explicit stylometric fingerprint guides an LLM to rewrite text in the target author style, even though the LLM's internal representations remain latent. The process is formalized as external style conditioning through instruction embedding, constraint activation, and iterative projection.

For non-experts, the fingerprint serves as a checklist, with the audit loop enforcing adherence.

---

## D.1 from Stylometric Profile to Control Signals

The style fingerprint $\mathcal{F}$ is not supplied to the LLM as raw statistics, but as a compiled control representation comprising:

1. Numeric constraints (ranges, histograms, tolerances)
2. Discrete symbolic constraints (lexicon rules, rhetorical templates, structural policies)
3. Priority and strictness controls (ordering, hard versus soft constraints)
4. Derived natural-language instructions (compiled in `derived_instructions.*`)
5. Optional bounded humanizer variance (seeded micro-variations within constraints)

The compiled instruction set:

$$\mathcal{I}(\mathcal{F}) = \text{Compile}(\mathcal{F})$$

where $\mathcal{I}(\mathcal{F})$ is a structured textual representation injected into the LLM prompt.

This compilation involves three main transformations:

### (I) Constraint Verbalization

Numeric constraints are rendered as qualitative instructions, for example:

- "Use short-to-medium sentences (10-18 words typical)"
- "Favour one-sentence paragraphs occasionally (~15%)"
- "Avoid heavy semicolon usage; commas preferred"

This converts $\psi_j(y)\in\mathcal{C}_j$ into behavioural descriptors.

### (II) Salience Weighting

Constraint weights $w_j$ are reflected in prompt ordering, emphasis (phrasing, repetition), and explicit language such as "must" or "prefer".

### (III) Conflict Resolution Policy

The `controls.priority_order` field induces a partial order:

$$\text{meaning preservation} \succ \text{lexicon} \succ \text{sentence rhythm} \succ \text{punctuation} \succ \text{templates}$$

This ordering is verbalized to ensure that stylistic fidelity does not override semantic fidelity.

Bounded stochastic variance, when enabled, allows a limited number of seeded micro-operations (such as transition swaps or filler drops) per 1000 words. These edits are constrained, logged, and subordinate to the fingerprint, introducing human-like irregularity without semantic drift.

---

## D.2 Conditioning as External Latent Space Steering

Let $h(x)$ denote the latent representation of the input text under the LLM, and $c(\mathcal{I})$ the latent encoding of the instruction set.

The model samples from:

$$p_\theta(y \mid x, \mathcal{I}) = p_\theta(y \mid h(x), c(\mathcal{I}))$$

Here, $c(\mathcal{I})$ induces a soft bias over stylistic manifolds in latent space.

Rather than learning a new style embedding, the fingerprint activates latent regions associated with sentence rhythm, biases token transitions linked to punctuation patterns, and suppresses lexical clusters disfavored by the lexicon rules.

This is analogous to feature activation steering in controllable generation, but with externalized and interpretable features.

---

## D.3 Constraint-Induced Token Distribution Shaping

At each decoding step $t$, the base LLM produces logits:

$$z_t = f_\theta(y_{<t}, x)$$

The fingerprint modifies the *effective sampling distribution* implicitly via instruction-conditioned logits:

$$z'_t = f_\theta(y_{<t}, x, \mathcal{I})$$

Although we do not explicitly reweight logits, empirical evidence from instruction-following models suggests that:

- sentence length constraints bias the probability mass toward early punctuation tokens  
- punctuation density constraints bias selection among delimiter tokens  
- lexicon constraints suppress token clusters through negative conditioning  

Formally, we may conceptualize an implicit reweighting:

$$p'_\theta(y_t \mid \cdot) \propto p_\theta(y_t \mid \cdot) \cdot \exp\left(-\sum_j \lambda_j \cdot \Delta_j(y_{1:t})\right)$$

where $\Delta_j$ approximates the incremental contribution of token $y_t$ toward violating constraint $j$.

This view aligns with **energy-based constrained decoding**, though implemented through natural-language conditioning rather than token-level modification.

---

## D.4 Emergence of Structural Style from Local Constraints

A central observation is that **global style emerges from the aggregation of local decisions**:

### Sentence Rhythm

Sentence length distributions arise from:

- early or delayed emission of terminal punctuation  
- preference for conjunction vs clause boundaries  
- tolerance for subordinate clauses  

The histogram constraint does not enforce exact lengths, but biases the *distribution of stopping times*.

### Paragraph Structure

Paragraph rhythm emerges from:

- probability of emitting newline tokens  
- probability of single-sentence termination  
- continuation bias under discourse coherence  

### Lexical Tone

Preferred phrases and avoided words act as **local energy barriers** in lexical space, steering generation toward author-characteristic idioms.

Thus, although constraints are global, **their enforcement is distributed across thousands of local token decisions**.

---

## D.5 Audit as Measurement in the Style Feature Space

After generation, the output $y$ is mapped back into feature space:

$$\phi(y) = (\psi_1(y), \dots, \psi_J(y))$$

This constitutes a **measurement operator**:

$$\mathcal{M}: y \mapsto \phi(y)$$

We compare $\phi(y)$ to target distributions and ranges, yielding a constraint violation vector:

$$v(y) = \big(g_1(\psi_1(y)), \dots, g_J(\psi_J(y))\big)$$

This transforms latent stylistic behavior into **observable geometric deviation** in feature space.

---

## D.6 Iterative Projection as Approximate Feasible Set Mapping

Let $y^{(0)}$ be the initial rewrite. The audit identifies a violation vector $v^{(0)}$.

The repair step constructs a new prompt containing:

- the original input $x$  
- the candidate $y^{(t)}$  
- the violation report $v^{(t)}$  
- the fingerprint $\mathcal{F}$  

The LLM then samples:

$$y^{(t+1)} \sim p_\theta(\cdot \mid x, \mathcal{I}, y^{(t)}, v^{(t)})$$

We interpret this as an approximate **projected gradient step** in style space:

$$y^{(t+1)} \approx \Pi_{\mathcal{C}}(y^{(t)})$$

where $\Pi_{\mathcal{C}}$ is the projection onto the admissible constraint set.

Convergence is detected when:

$$S(y^{(t)};\mathcal{F}) \ge S_{pass} \quad \land \quad H_t = 0$$

---

## D.7 Why This Works Without Fine-Tuning

Three properties of modern instruction-tuned LLMs make this approach viable:

### (I) Rich Latent Disentanglement

Although not perfectly disentangled, latent spaces exhibit partial separability between:

- syntactic rhythm  
- punctuation usage  
- discourse structure  
- lexical tone  

Stylometric constraints align with axes the model already encodes.

### (Ii) Strong Instruction Adherence

Instruction-tuned models approximate constrained decoding by:

- maintaining long-range control variables in attention  
- preserving global objectives across paragraphs  
- rebalancing generation probabilities dynamically  

### (Iii) Redundancy in Stylistic Signals

Writing style is **overdetermined**:

- sentence rhythm, punctuation, and lexicon co-vary  
- enforcing a subset typically induces the remainder  
- violations are sparse and repairable  

Thus the fingerprint need not be complete to be effective.

---

## D.8 Failure Modes and Their Detection

The system explicitly models and detects the principal failure classes.

### D.8.1 Semantic Drift

Detected by:
- entity mismatch  
- numeric mismatch  
- sentence insertion/deletion  

Policy:
- semantic constraints are hard  
- drift triggers rejection and repair  

### D.8.2 Over-Fitting to Sonstraints

Symptoms:
- unnaturally uniform sentence lengths  
- mechanical punctuation placement  
- lexical repetition  

Detected by:
- low entropy in histograms  
- excessive KL divergence collapse  

Mitigation:
- histogram tolerances  
- stochastic sampling  
- soft penalties  

### D.8.3 Constraint Incompatibility

Occurs when:
- input content structure is incompatible with author rhythm  
- rhetorical form cannot be preserved  

Detected by:
- repeated repair failure  
- persistent violation vector  

Policy:
- emit deviation report  
- relax low-priority constraints  

---

## D.9 Interpretation: Fingerprints as External Style Embeddings

We may view the fingerprint as an **explicit, interpretable surrogate for a latent style embedding**.

Instead of:

$$z_{style} \in \mathbb{R}^d \quad \text{(opaque)}$$

we construct:

$$\mathcal{F} = \{ (\psi_j, \mathcal{C}_j, w_j) \}_{j=1}^J$$

which defines a **style manifold**:

$$\mathcal{M}_{style} = \{ y : \forall j,\; \psi_j(y) \in \mathcal{C}_j \}$$

Rewriting becomes:

$$\hat y = \arg\max_{y \in \mathcal{M}_{style}} p_\theta(y \mid x)$$

This formulation clarifies that the fingerprint is not merely descriptive, but **defines the admissible region of stylistic space**.

---

## D.10 Summary

The fingerprint guides rewriting by:

1. Translating stylometric statistics into control instructions  
2. Steering latent generation via instruction-conditioned logits  
3. Measuring output in explicit feature space  
4. Iteratively projecting candidates toward the admissible manifold  
5. Enforcing semantic invariants as hard constraints  

This yields a controllable, auditable, and theoretically grounded mechanism for **author-conditioned style transfer without fine-tuning**.

---

---
   
   ## Appendix E. Existence and Feasibility of Stylometric Rewrites
   
   This appendix establishes sufficient conditions under which a **meaning-preserving rewrite satisfying a stylometric fingerprint exists**, and clarifies the role of tolerance bounds and constraint compatibility.
   
   ---
   
   ## E.1 Problem Restatement
   
   Given:
   
   - Input text $x \in \Sigma^*$  
   - Style fingerprint $\mathcal{F} = \{(\psi_j, \mathcal{C}_j, w_j)\}_{j=1}^J$  
   - Hard constraint set $\mathcal{H}\subseteq\{1,\dots,J\}$
   
   We seek an output $y$ such that:
   
   $$\begin{aligned}
   & y \in \mathcal{Y}_{hard}(x,\mathcal{F}) \\
   & \psi_j(y) \in \mathcal{C}_j \quad \forall j \notin \mathcal{H} \\
   & \text{Sem}(y) = \text{Sem}(x)
   \end{aligned}$$
   
   where $\text{Sem}(\cdot)$ denotes semantic equivalence up to admissible paraphrase.
   
   ---
   
   ## E.2 Feasible Stylometric Region
   
   Define the **stylometric manifold**:
   
   $$\mathcal{M}_{style} = \bigcap_{j=1}^J \{y : \psi_j(y) \in \mathcal{C}_j\}$$
   
   and the **semantic equivalence class**:
   
   $$\mathcal{E}(x) = \{y : \text{Sem}(y) = \text{Sem}(x)\}$$
   
   The feasible set is:
   
   $$\mathcal{F}(x,\mathcal{F}) = \mathcal{M}_{style} \cap \mathcal{E}(x)$$
   
   A rewrite exists iff:
   
   $$\mathcal{F}(x,\mathcal{F}) \neq \varnothing$$
   
   ---
   
   ## E.3 Sufficient Conditions for Non-Emptiness
   
   We provide sufficient (not necessary) conditions guaranteeing feasibility.
   
   ### Theorem 1 (Tolerance-Bound Feasibility)
   
   Assume:
   
   1. All histogram constraints satisfy:
   $$\tau_j \ge \epsilon_j$$
   where $\epsilon_j$ is the intrinsic sampling variance of feature $\psi_j$ over paraphrases of $x$
   
   2. All range constraints satisfy:
   $$[a_j, b_j] \supseteq [\psi_j(x) - \delta_j, \psi_j(x) + \delta_j]$$
   for some $\delta_j > 0$

3. Hard constraints do not contradict semantic invariants

Then: $$\mathcal{F}(x,\mathcal{F}) \neq \varnothing$$

#### Sketch of proof

Let $\mathcal{P}(x)$ be the set of meaning-preserving paraphrases of $x$, which is non-empty for any non-degenerate $x$. Feature maps $\psi_j$ are continuous, or piecewise continuous, under paraphrase operations. By assumption, tolerance intervals contain a neighbourhood around $\psi_j(x)$. It follows that there exists $y \in \mathcal{P}(x)$ such that $\psi_j(y)\in\mathcal{C}_j$ for all $j$. ∎

---

## E.4 Constraint Compatibility and Conflict Graphs

A constraint compatibility graph is defined as follows:

- Nodes represent constraints $j$.
- An edge connects $j$ and $k$ if $\mathcal{C}_j \cap \mathcal{C}_k = \varnothing$ under semantic invariants.

A necessary condition for feasibility:

$$\text{Graph}(\mathcal{F}) \text{ is bipartite with respect to hard constraints}$$

In practice, some constraints are compatible, while others may conflict:

- Sentence rhythm and paragraph rhythm are compatible.
- Lexicon and semantic invariants may conflict.
- Template and rhythm may conflict in short texts.

The system imposes a partial order:

$$\text{meaning} \succ \text{lexicon} \succ \text{structure} \succ \text{punctuation} \succ \text{templates}$$

This ensures that, in the event of conflict, feasibility is prioritized.

---

## E.5 Minimal Tolerance Bounds

Let $\sigma_j$ be the empirical standard deviation of feature $\psi_j$ across the author corpus.

Recommended sufficient tolerances:

- For range constraints:
$$[a_j, b_j] = [\mu_j - 2\sigma_j,\; \mu_j + 2\sigma_j]$$

- For histogram constraints:
$$\tau_j \ge 2 \cdot \mathbb{E}[W_1(\mathbf{h},\mathbf{h}')],$$

where $\mathbf{h}$ and $\mathbf{h}'$ are histograms from independent corpus samples.

These tolerances ensure that intra-author variation is admissible and that projection steps remain contractive.

---

## E.6 Convergence of Iterative Repair

Let $S(y;\mathcal{F})$ be the compliance score.

Suppose:

1. Each repair step reduces total violation:
$$\mathbb{E}[S(y^{(t+1)})] \ge S(y^{(t)}) + \eta$$
for some $\eta > 0$.

2. $S$ is bounded above by 1.

Then,

$$\exists T < \infty : S(y^{(T)}) \ge S_{pass}$$

i.e., finite-step convergence in expectation.

Empirically, rapid convergence (within one to three iterations) is observed in most rewrites.

---

## E.7 Degenerate and Infeasible Cases

Feasibility may fail in several situations:

1. Extremely short texts: There are insufficient degrees of freedom for histogram control.
2. Highly constrained technical content: Semantic invariants dominate stylistic degrees of freedom.
3. Overly tight tolerances: $\tau_j < \epsilon_j$.

In these cases, Stylometric-Transfer reports deviation, relaxes the lowest-priority constraints, and guarantees semantic correctness.

---

## E.8 Interpretation

The fingerprint does not specify a single point in style space, but rather a convex (or approximately convex) admissible region.

Rewriting succeeds when

$$\mathcal{E}(x) \cap \mathcal{M}_{style} \neq \varnothing$$

This perspective clarifies the necessity of tolerances, the ill-posed nature of strict imitation, and the rationale for deviation reporting.

---

## Appendix F. Comparison with Fine-Tuning and Lora

This section situates Stylometric-Transfer amongst existing approaches to author-style modelling and controlled generation, with attention to transparency and editorial control.

---

## F.1 Taxonomy of Style Modeling Approaches

Four principal paradigms can be distinguished:

| Paradigm | Representation | Training | Interpretability | Editability |
|----------|----------------|----------|------------------|-------------|
| Fine-tuning | Model weights | Required | None | None |
| LoRA / adapters | Low-rank deltas | Required | None | None |
| Latent embeddings | Vectors $z_{style}$ | Required | Low | None |
| **Stylometric-Transfer** | Explicit constraints | None | **High** | **Full** |

---

## F.2 Fine-Tuning Approaches

### Mechanism

Fine-tuning learns

$$p_{\theta'}(y\mid x) \approx p(y\mid x,\text{author})$$

by adjusting base parameters $\theta \to \theta'$.

### Limitations

- Style representation is entirely implicit.
- Learnt stylistic features cannot be inspected.
- No partial control; sentence rhythm and lexicon cannot be weighted separately.
- Catastrophic forgetting is a risk.
- Retraining is expensive for each author.

### Contrast

Stylometric-Transfer instead solves

$$\max_{y \in \mathcal{M}_{style}} p_\theta(y\mid x)$$

with no parameter updates, an explicit admissible region, and post-hoc auditing.

---

## F.3 LoRA / Adapter-Based Style Conditioning

### Mechanism

Low-rank matrices $\Delta W$ are learnt so that

$$h' = h + \Delta W h$$

encode author-specific modulation.

### Advantages

- Efficient.
- Modular.

### Limitations

- Style is encoded in a latent linear subspace.
- Not interpretable.
- No direct mapping to stylometric features.
- Combining multiple styles is difficult.

### Contrast

Stylometric-Transfer exposes every control dimension, allows continuous interpolation via tolerances, and supports manual editing and versioning.

---

## F.4 Latent Style Embedding Methods

### Mechanism

A vector

$$z_{style} \in \mathbb{R}^d$$

is learnt, and generation is conditioned as

$$p(y\mid x,z_{style})$$

using cross-alignment, VAEs, or conditional decoders.

### Advantages

- Compact.
- Differentiable.

### Limitations

- Dimensions are entangled.
- Coordinates lack semantic interpretation.
- No guarantee that $z_{style}$ corresponds to human-meaningful features.
- No auditability.

### Contrast

Stylometric-Transfer replaces

$$z_{style} \quad \longrightarrow \quad \mathcal{F} = \{(\psi_j,\mathcal{C}_j,w_j)\}$$

yielding explicit axes of variation, measurable compliance, and verifiable reproduction.

---

## F.5 Control Granularity and Editorial Authority

A central distinction is who controls style.

| Property | Fine-tune | LoRA | Embedding | Stylometric-Transfer |
|----------|-----------|------|-----------|----------------------|
| Human-readable model | [X] | [X] | [X] | **** |
| Partial constraint weighting | [X] | [X] | [X] | **** |
| Manual editing | [X] | [X] | [X] | **** |
| Version control | [X] | [X] | [X] | **** |
| Deviation reporting | [X] | [X] | [X] | **** |

Stylometric-Transfer treats style as an editorial object, not a byproduct of training.

---

## F.6 Data Efficiency

Fine-tuning and embedding methods require

$$N \gg 10^4 \text{ tokens}$$

to stabilize latent style representations.

Stylometric-Transfer requires only enough data to estimate low-variance statistics, often $N \approx 10^3-10^4$ tokens, and remains robust on heterogeneous corpora.

---

## F.7 Transferability and Compositionality

Latent methods encounter difficulties when combining multiple authors, interpolating interpretable features, or transferring style across domains. Stylometric-Transfer supports convex combinations of fingerprints, selective inheritance of features, and domain-specific constraint relaxation as central mechanisms.

Formally, fingerprints compose as

$$\mathcal{F}_\lambda = \lambda \mathcal{F}_1 + (1-\lambda)\mathcal{F}_2$$

`controls.humanizer_variance`

At the level of histogram mixtures, range interpolation, and lexicon unions, Stylometric-Transfer operates by making style constraints explicit.

---

## F.8 Interpretability and Scientific Value

From a scientific perspective, fine-tuning uncovers unknown features, embeddings encode unlabelled dimensions, and Stylometric-Transfer recovers measurable linguistic variables. This approach enables hypothesis testing, ablation studies, stylistic causality analysis, and reproducible experiments.

---

## F.9 Summary

Stylometric-Transfer departs from prevailing methods by externalizing style as a set of explicit constraints. This approach avoids reliance on training or latent embeddings, enabling auditability and editorial oversight, and supporting theoretical analysis of feasibility and convergence. Rather than inferring what style is, it delineates where style may reside within feature space.

---

## F.10 Comparison to Practical Pipelines

In practice, teams rarely deploy a single, unmodified paradigm. Systems considered "best-of-class" typically combine several components: an instruction-tuned large language model, guardrails for structure and meaning, some form of evaluation (both automatic and human), and, when resources allow, training-time specialization (such as fine-tuning or adapters).

The table below contrasts Stylometric-Transfer with common high-performing alternatives as they are typically implemented. The intention is not to assert superiority, but to clarify the relevant trade-offs.

| Approach (typical) | Strengths | Weaknesses | When it tends to win |
| --- | --- | --- | --- |
| Prompt-only style steering (LLM + "write like X") | Lowest engineering cost, rapid iteration, can be effective for broad register shifts | Non-auditable, fragile to prompt drift, difficult to reproduce, style may collapse into generic LLM voice, meaning drift is common without additional checks | Casual rewrites, low-stakes editing, early prototyping |
| Agentic / iterative editing (LLM + critique loops) | Improved compliance via self-critique, can identify formatting or consistency errors, flexible to new constraints | Greater latency and cost, risk of over-editing and drift, "improvement" can be subjective and unstable | Long documents where structure is important, when a human reviewer is involved |
| Fine-tuned author or domain model | Highest imitation fidelity in-domain, reduced prompt overhead, can be efficient at inference once trained | Opaque, costly to train and update, difficult to verify changes, risky for impersonation, challenging to parameterize partial style controls | Narrow, stable domains with large datasets, high-volume generation, strict internal style guides |
| LoRA/adapters for style or domain | Less expensive than full fine-tuning, modular, adapters can be switched | Still opaque, style dimensions may be entangled, multiple adapters can conflict, auditability remains limited | Medium-scale specialization, internal domain conditioning |
| Latent style embeddings + conditional generation | Compact conditioning, can interpolate styles, integrates with learnt pipelines | Difficult to interpret, evaluation is challenging, risk of content leakage into style vector | Research settings, controlled datasets, style mixing experiments |
| Detector-guided "humanization" optimization | Can target a specific detector or proxy, objective is easily defined | Highly susceptible to Goodhart's law, may produce adversarial artifacts, can impair meaning and readability | When the explicit objective is to reduce detector score (not recommended for general writing) |
| **Stylometric-Transfer (this work)** | Explicit, versionable constraints, local grounding, stable reproducibility, clear deviation reporting, integrates humanization as auditable mechanisms | Limited ceiling for subtle stylistic phenomena, depends on LLM compliance, chunking introduces boundary effects, heuristics may mis-segment | When auditability is required, when tunable controls are needed, when per-author JSON artifacts are preferred over opaque weights |

### F.10.1 Explicit Constraints in Practice

Two practical observations explain why explicit constraints can compete with more elaborate learnt approaches in real deployments. First, human-in-the-loop editing benefits from inspectability. When a rewrite is unsatisfactory, a JSON fingerprint and a deviations report indicate which parameter requires adjustment. By contrast, a fine-tuned model offers few actionable levers beyond retraining or prompt modification. Second, many style differences are low-dimensional and stable.

In technical fields, the most noticeable improvements come from a narrow set of stable signals: sentence rhythm, punctuation, discourse markers, function-word balance, and a handful of lexical preferences. Making these elements explicit allows for predictable control, even if some subtleties are missed.

### F.10.2 Where Learnt Approaches Retain an Advantage

Learnt methods maintain an edge when style depends on high-dimensional features that lightweight metrics cannot easily capture. These include nuanced syntactic alternations, idiomatic collocations, long-range narrative structure, and pragmatic implicature. In such cases, explicit constraints may approximate the surface of the voice - cadence and visible markers - without fully reproducing the deeper texture.

---

## Appendix G. JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://stylometric-transfer/schema/style_fingerprint.schema.json",
  "title": "Stylometric Style Fingerprint Schema",
  "description": "Explicit, interpretable stylometric style model for author-conditioned style transfer",
  "type": "object",
  "required": [
    "schema_version",
    "profile_id",
    "metadata",
    "measurements",
    "targets",
    "lexicon",
    "templates",
    "controls",
    "validators",
    "derived_instructions"
  ],
  "properties": {
    "schema_version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"
    },
    "profile_id": {
      "type": "string",
      "description": "Unique identifier for this fingerprint"
    },
    "metadata": {
      "type": "object",
      "required": ["author", "corpus", "extraction"],
      "properties": {
        "author": {
          "type": "object",
          "required": ["name", "is_self"],
          "properties": {
            "name": { "type": "string" },
            "is_self": { "type": "boolean" }
          }
        },
        "corpus": {
          "type": "object",
          "required": ["document_count", "documents", "size", "sampling"],
          "properties": {
            "document_count": { "type": "integer", "minimum": 1 },
            "documents": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["path", "size"],
                "properties": {
                  "path": { "type": "string" },
                  "name": { "type": "string" },
                  "title": { "type": ["string", "null"] },
                  "description": { "type": ["string", "null"] },
                  "language": { "type": ["string", "null"] },
                  "locale": { "type": ["string", "null"] },
                  "genres": { "type": "array", "items": { "type": "string" } },
                  "time_range": {
                    "type": "object",
                    "properties": {
                      "start": { "type": ["string", "null"] },
                      "end": { "type": ["string", "null"] }
                    }
                  },
                  "size": {
                    "type": "object",
                    "required": ["words_est"],
                    "properties": {
                      "words_est": { "type": "integer", "minimum": 0 },
                      "pages_est": { "type": ["integer", "null"] },
                      "sentences_est": { "type": "integer", "minimum": 0 },
                      "paragraphs_est": { "type": "integer", "minimum": 0 },
                      "chars": { "type": "integer", "minimum": 0 }
                    }
                  }
                }
              }
            },
            "size": {
              "type": "object",
              "properties": {
                "words_est": { "type": "integer", "minimum": 0 },
                "pages_est": { "type": ["integer", "null"] }
              }
            },
            "sampling": {
              "type": "object",
              "properties": {
                "method": { "type": "string" },
                "notes": { "type": "string" }
              }
            }
          }
        },
        "extraction": {
          "type": "object",
          "required": ["model", "date", "methods", "confidence"],
          "properties": {
            "model": { "type": "string" },
            "date": { "type": "string" },
            "methods": { "type": "array", "items": { "type": "string" } },
            "confidence": { "type": "string" },
            "limitations": { "type": "array", "items": { "type": "string" } },
            "tunables_snapshot": {
              "type": ["object", "null"],
              "description": "Optional snapshot of config.tunables.json used during fingerprinting for auditability."
            }
          }
        }
      }
    },
    "measurements": {
      "type": "object",
      "description": "Raw stylometric measurements extracted from corpus",
      "additionalProperties": true,
      "properties": {
        "totals": {
          "type": "object",
          "properties": {
            "documents_used": { "type": "integer", "minimum": 0 },
            "total_words_est": { "type": "integer", "minimum": 0 },
            "total_sentences_est": { "type": "integer", "minimum": 0 },
            "total_paragraphs_est": { "type": "integer", "minimum": 0 }
          }
        },
        "sentence": {
          "type": "object",
          "properties": {
            "length_words": {
              "type": "object",
              "properties": {
                "mean": { "type": "number" },
                "stdev": { "type": "number" },
                "histogram_bins": { "type": "array", "items": { "type": "string" } },
                "histogram_p": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 1 } }
              }
            }
          }
        },
        "paragraph": {
          "type": "object",
          "properties": {
            "length_sentences_histogram_bins": { "type": "array", "items": { "type": "string" } },
            "length_sentences_histogram_p": { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 1 } },
            "one_sentence_paragraph_rate": { "type": "number" }
          }
        },
        "punctuation": {
          "type": "object",
          "properties": {
            "counts": { "type": "object", "additionalProperties": { "type": "number" } },
            "rates_per_1000w": { "type": "object", "additionalProperties": { "type": "number" } },
            "comma_density_per_100w": { "type": "number" }
          }
        },
        "orthography_signals": {
          "type": "object",
          "description": "Orthography and spelling heuristics",
          "properties": {
            "contractions_rate": { "type": "number" },
            "oxford_comma_signal": { "type": "number" },
            "spelling_variant": { "$ref": "#/definitions/spelling_variant" }
          }
        },
        "function_words": {
          "type": "object",
          "properties": {
            "rates_per_1000w": { "type": "object", "additionalProperties": { "type": "number" } },
            "top": { "type": "array", "items": { "$ref": "#/definitions/word_count" } }
          }
        },
        "stance_signals": {
          "type": "object",
          "properties": {
            "hedge_rate": { "type": "number" },
            "booster_rate": { "type": "number" },
            "directive_rate": { "type": "number" },
            "first_person_rate": { "type": "number" },
            "second_person_rate": { "type": "number" },
            "third_person_rate": { "type": "number" }
          }
        },
        "templates_signals": {
          "type": "object",
          "properties": {
            "sentence_openers_top": { "type": "array", "items": { "$ref": "#/definitions/phrase_count" } },
            "transition_openers_top": { "type": "array", "items": { "$ref": "#/definitions/phrase_count" } },
            "transition_marker_positions": {
              "type": "object",
              "properties": {
                "start_rate_per_1000w": { "type": "number" },
                "mid_rate_per_1000w": { "type": "number" }
              }
            }
          }
        },
        "rhetoric_moves": {
          "type": "object",
          "properties": {
            "claim_rate": { "type": "number" },
            "evidence_rate": { "type": "number" },
            "counterpoint_rate": { "type": "number" },
            "concession_rate": { "type": "number" },
            "synthesis_rate": { "type": "number" },
            "claim_evidence_ratio": { "type": "number" }
          }
        },
        "paragraph_cadence": {
          "type": "object",
          "properties": {
            "opening_sentence_length_mean": { "type": "number" },
            "opening_sentence_length_stdev": { "type": "number" },
            "closing_sentence_length_mean": { "type": "number" },
            "closing_sentence_length_stdev": { "type": "number" }
          }
        },
        "epistemic_profile": {
          "type": "object",
          "properties": {
            "speculative_rate": { "type": "number" },
            "probabilistic_rate": { "type": "number" },
            "assertive_rate": { "type": "number" },
            "directive_rate": { "type": "number" }
          }
        },
        "syntax_texture": {
          "type": "object",
          "properties": {
            "subordinate_clause_rate": { "type": "number" },
            "parenthetical_rate": { "type": "number" },
            "appositive_rate": { "type": "number" }
          }
        },
        "lexical_signals": {
          "type": "object",
          "description": "Lexical statistics such as rare-word lists",
          "properties": {
            "rare_words": { "type": "array", "items": { "$ref": "#/definitions/word_count" } },
            "rare_word_max_count": { "type": "integer" },
            "rare_word_min_length": { "type": "integer" }
          }
        },
        "lexical_avoidance": {
          "type": "object",
          "properties": {
            "category_rates_per_1000w": { "type": "object", "additionalProperties": { "type": "number" } },
            "rare_words": { "type": "array", "items": { "$ref": "#/definitions/word_count" } }
          }
        },
        "repetition": {
          "type": "object",
          "properties": {
            "bigram_repeat_rate": { "type": "number" },
            "trigram_repeat_rate": { "type": "number" },
            "min_repeat_count": { "type": "integer" }
          }
        },
        "common_phrases": {
          "type": "object",
          "properties": {
            "bigrams_top": { "type": "array", "items": { "$ref": "#/definitions/phrase_count" } },
            "trigrams_top": { "type": "array", "items": { "$ref": "#/definitions/phrase_count" } }
          }
        },
        "common_phrases_validation": {
          "type": "object",
          "properties": {
            "validated": {
              "type": "object",
              "properties": {
                "bigrams_top": { "type": "array", "items": { "$ref": "#/definitions/phrase_count" } },
                "trigrams_top": { "type": "array", "items": { "$ref": "#/definitions/phrase_count" } }
              }
            },
            "dropped": { "type": "array", "items": { "$ref": "#/definitions/phrase_count" } },
            "notes": { "type": "array", "items": { "type": "string" } }
          }
        }
      }
    },
    "targets": {
      "type": "object",
      "description": "Target ranges and distributions for rewriting",
      "additionalProperties": true,
      "properties": {
        "persona": {
          "type": "object",
          "description": "Persona and stance constraints",
          "properties": {
            "pronoun_preferences": { "$ref": "#/definitions/pronoun_preferences" }
          }
        },
        "rhetoric_moves": {
          "type": "object",
          "properties": {
            "claim_rate": { "$ref": "#/definitions/range" },
            "evidence_rate": { "$ref": "#/definitions/range" },
            "counterpoint_rate": { "$ref": "#/definitions/range" },
            "concession_rate": { "$ref": "#/definitions/range" },
            "synthesis_rate": { "$ref": "#/definitions/range" },
            "claim_evidence_ratio": { "$ref": "#/definitions/range" }
          }
        },
        "epistemic_profile": {
          "type": "object",
          "properties": {
            "speculative_rate": { "$ref": "#/definitions/range" },
            "probabilistic_rate": { "$ref": "#/definitions/range" },
            "assertive_rate": { "$ref": "#/definitions/range" },
            "directive_rate": { "$ref": "#/definitions/range" }
          }
        },
        "paragraph_cadence": {
          "type": "object",
          "properties": {
            "opening_sentence_length_mean": { "$ref": "#/definitions/range" },
            "opening_sentence_length_stdev": { "$ref": "#/definitions/range" },
            "closing_sentence_length_mean": { "$ref": "#/definitions/range" },
            "closing_sentence_length_stdev": { "$ref": "#/definitions/range" }
          }
        },
        "syntax_texture": {
          "type": "object",
          "properties": {
            "subordinate_clause_rate": { "$ref": "#/definitions/range" },
            "parenthetical_rate": { "$ref": "#/definitions/range" },
            "appositive_rate": { "$ref": "#/definitions/range" }
          }
        },
        "discourse_markers": {
          "type": "object",
          "properties": {
            "start_rate_per_1000w": { "$ref": "#/definitions/range" },
            "mid_rate_per_1000w": { "$ref": "#/definitions/range" }
          }
        },
        "repetition": {
          "type": "object",
          "properties": {
            "bigram_repeat_rate": { "$ref": "#/definitions/range" },
            "trigram_repeat_rate": { "$ref": "#/definitions/range" }
          }
        }
      }
    },
    "lexicon": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "preferred_words": { "type": "array", "items": { "type": "string" } },
        "preferred_phrases": { "type": "array", "items": { "type": "string" } },
        "avoid_words": { "type": "array", "items": { "type": "string" } },
        "avoid_words_soft": { "type": "array", "items": { "type": "string" } },
        "avoid_phrases": { "type": "array", "items": { "type": "string" } },
        "synonym_preferences": { "type": "object", "additionalProperties": { "type": "string" } },
        "notes": { "type": "string" }
      }
    },
    "templates": {
      "type": "object",
      "description": "Rhetorical and syntactic templates",
      "additionalProperties": true,
      "properties": {
        "sentence_openers": { "type": "array", "items": { "type": "string" } },
        "paragraph_openers": { "type": "array", "items": { "type": "string" } },
        "preferred_transitions": { "type": "array", "items": { "type": "string" } },
        "syntactic_patterns": { "type": "array", "items": { "type": "string" } },
        "paragraph_moves": { "type": "array", "items": { "type": "string" } },
        "rhetorical_moves": { "type": "array", "items": { "type": "string" } }
      }
    },
    "controls": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "priority_order": { "type": "array", "items": { "type": "string" } },
        "strictness": { "type": ["string", "object"] },
        "rewrite_policy": { "type": ["string", "object"] },
        "humanizer_variance": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean" },
            "seed": { "type": "integer" },
            "max_ops_per_1000w": { "type": "number" },
            "allowed_ops": { "type": "array", "items": { "type": "string" } }
          }
        }
      }
    },
    "validators": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "weights": { "type": "object", "additionalProperties": { "type": "number", "minimum": 0 } },
        "scoring_weights": { "type": "object", "additionalProperties": { "type": "number", "minimum": 0 } },
        "checks": { "type": "array", "items": { "type": ["string", "object"] } },
        "scoring": { "type": "object" }
      }
    },
    "derived_instructions": {
      "type": "object",
      "description": "Compiled prompts and guidance for LLM",
      "additionalProperties": true,
      "properties": {
        "system_style": { "type": ["string", "array"] },
        "rewrite_prompt": { "type": "string" },
        "generation_prompt": { "type": "string" }
      }
    }
  },
  "definitions": {
    "range": {
      "type": "object",
      "required": ["min", "max"],
      "properties": {
        "min": { "type": "number" },
        "max": { "type": "number" }
      }
    },
    "histogram": {
      "type": "object",
      "required": ["bins", "values"],
      "properties": {
        "bins": {
          "type": "array",
          "description": "Ordinal bin boundaries",
          "items": { "type": "string" }
        },
        "values": {
          "type": "array",
          "description": "Probability mass per bin",
          "items": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      }
    },
    "spelling_variant": {
      "type": "object",
      "description": "Heuristic detection of US vs Canadian spelling in English text",
      "properties": {
        "language": { "type": "string" },
        "variant": { "type": "string", "enum": ["us", "canadian", "unknown"] },
        "confidence": { "type": "string", "enum": ["low", "medium", "high"] },
        "us_hits": { "type": "integer", "minimum": 0 },
        "canadian_hits": { "type": "integer", "minimum": 0 },
        "examples": {
          "type": "object",
          "properties": {
            "us": { "type": "array", "items": { "type": "string" } },
            "canadian": { "type": "array", "items": { "type": "string" } }
          }
        },
        "note": { "type": "string" }
      }
    },
    "pronoun_preferences": {
      "type": "object",
      "description": "Preferred pronoun sets when writing in English",
      "properties": {
        "default_set": { "type": "string" },
        "allowed_sets": { "type": "array", "items": { "type": "string" } },
        "avoid_sets": { "type": "array", "items": { "type": "string" } },
        "strictness": { "type": "string", "enum": ["soft", "hard"] },
        "notes": { "type": "string" }
      }
    },
    "phrase_count": {
      "type": "object",
      "properties": {
        "phrase": { "type": "string" },
        "count": { "type": "integer", "minimum": 0 },
        "ngram": { "type": "integer", "minimum": 1 },
        "reason": { "type": "string" }
      }
    },
    "word_count": {
      "type": "object",
      "properties": {
        "word": { "type": "string" },
        "count": { "type": "integer", "minimum": 0 },
        "rate_per_1000w": { "type": "number" }
      }
    }
}
```

---

## Appendix H. Tunables Schema

The tunables schema defines deterministic parameters for humanizer conflict thresholds and basic validation, including line-count change warnings, during style application. When a fingerprint is generated, the current tunables may be embedded under the tunables_snapshot key to document the exact settings used for provenance. The schema below specifies the supported keys and types:

`config.tunables.json`
`metadata.extraction.tunables_snapshot`

``` json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://stylometric-transfer/schema/config.tunables.schema.json",
  "title": "Stylometric-Transfer Tunables Schema",
  "description": "Optional tuning parameters for humanizer conflict thresholds",
  "type": "object",
  "properties": {
    "humanizer_conflicts": {
      "type": "object",
      "properties": {
        "em_dash_keep_rate": { "type": "number", "minimum": 0 },
        "hedge_keep_rate": { "type": "number", "minimum": 0 },
        "first_person_keep_rate": { "type": "number", "minimum": 0 },
        "contractions_avoid_threshold": { "type": "number", "minimum": 0 },
        "contractions_use_threshold": { "type": "number", "minimum": 0 },
        "heading_title_case_keep_rate": { "type": "number", "minimum": 0 },
        "boldface_keep_per_1000w": { "type": "number", "minimum": 0 },
        "inline_header_list_keep_rate": { "type": "number", "minimum": 0 }
      },
      "additionalProperties": false
    },
    "humanizer_mandatory": {
      "type": "object",
      "properties": {
        "avoid_em_dashes": { "type": "boolean" },
        "normalize_double_quotes": { "type": "boolean" },
        "emoji_policy": {
          "type": "string",
          "enum": ["remove", "replace", "none"]
        },
        "force_local_spelling": {
          "type": "string",
          "enum": ["none", "canadian", "australian", "british", "us"]
        }
      },
      "additionalProperties": false
    },
    "humanizer_variance": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "seed": { "type": "integer" },
        "max_ops_per_1000w": { "type": "number", "minimum": 0 },
        "allowed_ops": {
          "type": "array",
          "items": { "type": "string" }
        }
      },
      "additionalProperties": false
    },
    "humanization_metrics": {
      "type": "object",
      "properties": {
        "weights": {
          "type": "object",
          "additionalProperties": { "type": "number", "minimum": 0 }
        }
      },
      "additionalProperties": false
    },
    "humanization_baseline": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "window_words": { "type": "integer", "minimum": 50 },
        "stride_words": { "type": "integer", "minimum": 25 },
        "min_window_words": { "type": "integer", "minimum": 50 },
        "max_windows": { "type": "integer", "minimum": 1 }
      },
      "additionalProperties": false
    },
    "humanization_controller": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "seed": { "type": "integer" },
        "quantiles": {
          "type": "array",
          "items": { "type": "number", "minimum": 0, "maximum": 1 }
        },
        "range_pct": { "type": "number", "minimum": 0 },
        "min_width": { "type": "number", "minimum": 0 },
        "max_width": { "type": "number", "minimum": 0 },
        "allowed_metrics": {
          "type": "array",
          "items": { "type": "string" }
        },
        "feedback_enabled": { "type": "boolean" },
        "feedback_tolerance": { "type": "number", "minimum": 0 },
        "max_feedback_retries": { "type": "integer", "minimum": 0 }
      },
      "additionalProperties": false
    },
    "lexical_signals": {
      "type": "object",
      "properties": {
        "rare_words_limit": { "type": "integer", "minimum": 1 }
      },
      "additionalProperties": false
    },
    "lexical_avoidance": {
      "type": "object",
      "properties": {
        "rare_words_limit": { "type": "integer", "minimum": 1 }
      },
      "additionalProperties": false
    },
    "controls_normalization": {
      "type": "object",
      "properties": {
        "rewrite_policy": {
          "type": "object",
          "properties": {
            "jaccard_threshold": { "type": "number", "minimum": 0, "maximum": 1 },
            "dedupe_on_subset": { "type": "boolean" },
            "prefer_more_specific": { "type": "boolean" },
            "compress_directives": { "type": "boolean" },
            "directive_verbs": {
              "type": "array",
              "items": { "type": "string" }
            },
            "stopwords": {
              "type": "array",
              "items": { "type": "string" }
            }
          },
          "additionalProperties": false
        },
        "priority_order": {
          "type": "object",
          "properties": {
            "token_pattern": { "type": "string" },
            "dedupe_case_insensitive": { "type": "boolean" },
            "exclude_tokens": {
              "type": "array",
              "items": { "type": "string" }
            }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },
    "fiction_detection": {
      "type": "object",
      "properties": {
        "quote_span_min": { "type": "integer", "minimum": 0 },
        "quoted_ratio_min": { "type": "number", "minimum": 0, "maximum": 1 },
        "quote_para_ratio_min": { "type": "number", "minimum": 0, "maximum": 1 },
        "quoted_ratio_force": { "type": "number", "minimum": 0, "maximum": 1 }
      },
      "additionalProperties": false
    },
    "chunking": {
      "type": "object",
      "properties": {
        "max_input_tokens": { "type": "integer", "minimum": 200 },
        "chunk_split_on": {
          "type": "string",
          "enum": ["word", "sentence", "paragraph"]
        },
        "chunk_summary": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean" },
            "summary_words": { "type": "integer", "minimum": 5 }
          },
          "additionalProperties": false
        },
        "min_chunks_when_perturbing": { "type": "integer", "minimum": 1 },
        "recovery_split_max_depth": { "type": "integer", "minimum": 0 },
        "recovery_split_min_chars": { "type": "integer", "minimum": 0 },
        "variance_aware": {
          "type": "object",
          "properties": {
            "enabled": { "type": "boolean" },
            "sentence_stdev_ref": { "type": "number", "minimum": 0 },
            "paragraph_burst_ref": { "type": "number", "minimum": 0 },
            "min_factor": { "type": "number", "minimum": 0 },
            "max_factor": { "type": "number", "minimum": 0 }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },
    "style_retry": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "threshold": { "type": "number", "minimum": 0, "maximum": 1 },
        "max_retries": { "type": "integer", "minimum": 0 }
      },
      "additionalProperties": false
    },
    "section_restore": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "max_restore_sections": { "type": "integer", "minimum": 0 },
        "heading_similarity_threshold": { "type": "number", "minimum": 0, "maximum": 1 },
        "signature_similarity_threshold": { "type": "number", "minimum": 0, "maximum": 1 },
        "signature_min_overlap": { "type": "integer", "minimum": 0 }
      },
      "additionalProperties": false
    },
    "sanity_checks": {
      "type": "object",
      "properties": {
        "line_count_warn_pct": { "type": "number", "minimum": 0 },
        "word_count_warn_pct": { "type": "number", "minimum": 0 },
        "paragraph_count_warn_pct": { "type": "number", "minimum": 0 }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

Humanization guidelines are adapted from the humanizer skill in softaworks/agent-toolkit by @leonardocouy.

### H.1 Tunable Definitions

- `em_dash_keep_rate`: If the fingerprint’s em dash rate (per 1,000 words) meets or exceeds this value, the rule to avoid em dashes is set aside as conflicting.
- `hedge_keep_rate`: If the fingerprint’s hedging rate (per 1,000 words) meets or exceeds this value, rules discouraging hedging are set aside.
- `first_person_keep_rate`: If the fingerprint’s first person rate (per 1,000 words) is below this value (or pronoun preferences avoid first person), rules requiring first person are set aside.
- `contractions_avoid_threshold`: If the fingerprint’s contraction rate (per 1,000 words) meets or exceeds this value, the rule to avoid contractions is set aside.
- `contractions_use_threshold`: If the fingerprint’s contraction rate (per 1,000 words) is below this value, the rule to use contractions is set aside.
- `heading_title_case_keep_rate`: If the input Markdown’s heading Title Case ratio meets or exceeds this value, the rule to avoid Title Case headings is set aside.
- `boldface_keep_per_1000w`: If boldface density (per 1,000 words) meets or exceeds this value, the rule to avoid boldface is set aside.
- `inline_header_list_keep_rate`: If the ratio of inline-header list items (such as `- **Label:**`) meets or exceeds this value, the rule to avoid inline-header lists is set aside.
- `avoid_em_dashes`: When true, em dashes are always removed in the final output (mandatory humanizer control).
- `normalize_double_quotes`: When true, curly double quotes are normalized to straight quotes after rewriting.
- `emoji_policy`: `remove`, `replace`, or `none`. `replace` swaps emojis with conventional monochrome symbols when possible, otherwise removes them.
- `force_local_spelling`: Locale override for spelling normalization applied after rewriting (`none`, `canadian`, `australian`, `british`, `us`). Rules in `config.local_spelling_rules.json` are applied.
- `humanizer_variance.enabled`: Enables bounded stochastic micro-variation during application.
- `humanizer_variance.seed`: RNG seed for deterministic runs.
- `humanizer_variance.max_ops_per_1000w`: Maximum number of micro-operations per 1,000 words. The recommended starting point is `0.5`; `0.5–1.5` is generally safe. Values above `2.0` can introduce noise unless the input is highly repetitive.
- `humanizer_variance.allowed_ops`: Allowed micro-operations (for example, `swap_transition`, `drop_filler`). It is advisable to begin with `["swap_transition", "drop_filler"]`, add operations gradually, and keep the list short to avoid compounding randomness.
- `humanization_metrics.weights`: Optional weighting for the 0–100 aggregate humanization score. Any metric assigned a weight of 0 is excluded.
- `lexical_signals.rare_words_limit`: Maximum number of rare words included in `measurements.lexical_signals.rare_words`.
- `lexical_avoidance.rare_words_limit`: Maximum number of rare words included in `measurements.lexical_avoidance.rare_words`.
- `chunking.max_input_tokens`: Hard cap on input tokens per chunk (after prompt overhead). Lower values increase chunk count but reduce per-request latency and timeouts.
- `chunking.chunk_split_on`: Primary chunking unit (`word`, `sentence`, or `paragraph`). If a paragraph exceeds the budget, it falls back to sentence splitting for that chunk; if a sentence is still oversized, it falls back to word splitting for that chunk. Bullet or numbered list lines are treated as sentence units.
- `chunking.chunk_summary.enabled`: When true, each chunk requests a short rolling summary (not included in the final output) and passes it to the next chunk for semantic continuity.
- `chunking.chunk_summary.summary_words`: Target word count for the rolling summary (default 25). Keep small to minimize token overhead.
- `style_retry.enabled`: Enable or disable the delta-feedback retry pass after measuring style compliance.
- `style_retry.threshold`: Retry when compliance score is below this threshold (default `0.75`). Lower values trigger fewer retries (more permissive); higher values trigger more retries (stricter). `0.0` effectively disables threshold-based retries, while `1.0` retries unless the output is nearly perfect.
- `style_retry.max_retries`: Maximum number of retry passes (default `1`).
- `section_restore.enabled`: Enable or disable restoring missing sections after rewrite.
- `section_restore.max_restore_sections`: Maximum number of missing sections to restore (0 disables restoration).
- `section_restore.heading_similarity_threshold`: Fuzzy heading match threshold for considering a rewritten heading present.
- `section_restore.signature_similarity_threshold`: Content-signature similarity threshold for matching a section by its opening content.
- `section_restore.signature_min_overlap`: Minimum number of overlapping signature tokens required for a content match.
- `line_count_warn_pct`: If the output line count changes by this percentage or more, a warning is issued for possible missing or expanded content.
- `word_count_warn_pct`: If the output word count changes by this percentage or more, a warning is issued for possible missing or expanded content.
- `paragraph_count_warn_pct`: If the output paragraph count changes by this percentage or more, a warning is issued for possible missing or expanded content.

---

### Appendix I. Stylometry and Humanization: FAQ and Glossary

This appendix is intentionally detailed and is intended for readers seeking clarification between textbook definitions and code.

### I.1 Glossary

**Author-voice text** The portion of a document assumed to represent the author’s own prose rather than quoted material, references, footnotes, or boilerplate. The project filters non-author-voice regions before measuring style and before requesting a rewrite.

**Token / type** A *token* refers to one occurrence of a word; a *type* is a unique word form. If the text is "cats chase cats", then $N=3$ tokens and $V=2$ types.

**Rate per 1,000 words** A normalised count: $r = 1000 \cdot C / W$. This rescales event counts (commas, semicolons, hedges) so that texts of different lengths become comparable.

**Histogram (probability distribution)** A vector of bin probabilities $\mathbf{h}$ such that $\sum_b h_b = 1$. Histograms are used for sentence-length and paragraph-length distributions because the "shape" matters (variance and tails) even when means match.

**Entropy** A measure of spread. If probabilities are $p_i$, then $H=-\sum_i p_i\log p_i$. High entropy indicates that many categories are used fairly evenly; low entropy indicates that mass is concentrated in a few categories. In this project, entropy is used as a proxy for variety (punctuation variety, function-word variety, character-trigram texture).

**KL divergence** If $P$ is an observed distribution and $Q$ is a reference distribution, then: $$D_{\\mathrm{KL}}(P\\|Q)=\\sum_i P_i\\log\\frac{P_i}{Q_i}.$$ It is asymmetric. In operational terms, it penalises mass in $P$ that is surprising under $Q$. The project uses KL-derived measures to quantify how far a rewrite drifts from the fingerprint in distributional features.

**Jensen–Shannon divergence (JSD)** A symmetric, smoothed divergence derived from KL: $$\\mathrm{JSD}(P,Q)=\\tfrac{1}{2}D_{\\mathrm{KL}}(P\\|M)+\\tfrac{1}{2}D_{\\mathrm{KL}}(Q\\|M), \\quad M=\\tfrac{1}{2}(P+Q).$$ JSD is bounded and tends to be easier to interpret as a distance-like score for histograms.

**Burstiness (coefficient of variation)** For a length sequence $\\ell$, burstiness is $B = \\sigma(\\ell)/\\mu(\\ell)$. It measures variability relative to the mean. LLM outputs often have artificially low burstiness; many human authors do not.

**Self-echo** Repeated n-grams within a text (often beyond what the topic justifies). This paper uses repeated bigram/trigram rates as a proxy.

**Humanization metric** A quantitative proxy intended to detect and reduce AI-typical artefacts. In this system, metrics are for engineering feedback, not forensic detection.

**Humanization baseline** Intra-corpus variability extracted from the author’s corpus and used to produce per-chunk controller overlays (small target nudges that encourage natural dispersion).

**Controller overlay** A per-chunk adjustment to target ranges (for example, nudging sentence-length mean towards a sampled quantile of the corpus distribution) intended to introduce author-like variability without random drift. It is deterministic given a seed and tunables.

**Bounded stochastic variance** Seeded micro-edits (small operations) applied under strict caps (operations per 1,000 words, allowed operation set). The word "bounded" is key: it is designed to be auditable and to prevent excessive randomness.

**Chunking** Splitting input Markdown into smaller parts so each LLM call remains within a token budget. Chunking is not solely a performance measure; it affects distributional control. Smaller chunks reduce per-call timeouts and allow more local variability, but can increase the risk of global inconsistencies if constraints are not managed carefully.

**Deviation report** A structured record of constraint conflicts, style drifts, and any deterministic fixups applied after the LLM output. Deviations constitute the audit trail.

### I.2 FAQ

**Q: Why does stylometry often focus on function words?** Function words are frequent, relatively topic-invariant, and difficult to consciously control. If two texts match on many topic words but diverge strongly on function-word balance and connective habits, the style is plausibly different even if the topic is the same.

**Q: Why use per-1,000-word rates rather than raw counts?** Raw counts scale with length and can be misleading. Normalisation makes style density comparable. It also makes a rewrite audit meaningful: if a 2,000-word text has 120 commas, the comma density is 60 per 1,000 words. If the rewrite has 30 commas, density is 15 per 1,000 words, indicating compression or a shift to simpler sentence structure.

**Q: Why store histograms instead of just mean and standard deviation?** The distribution’s tails often carry stylistic meaning. Many authors occasionally produce very short or very long sentences. LLMs tend to regress to the middle. A histogram preserves where the mass sits across bins, which is a more direct representation of rhythm than a single mean.

**Q: If the corpus contains quotations, why filter them at all?** Quotations can represent other voices and content. For non-fiction, long quoted passages are typically the speech of sources, not the author’s own style. For fiction, quoted dialogue is part of the author’s craft and should usually be included. The project therefore detects fiction versus non-fiction and changes the quote-handling policy accordingly, with an explicit message and manual overrides.

**Q: Why not use a full syntactic parser or a transformer embedding for deeper style?** Such approaches are possible, but they entail costs: dependencies, latency, fragility, and reduced auditability. The philosophy here is to measure what can be defended. A smaller set of stable, interpretable features often provides more engineering leverage than a high-dimensional embedding whose drift is difficult to diagnose.

**Q: Is the humanization score a detector?** No. It is a dashboard metric. It is intended to indicate whether the rewrite became more uniform, more repetitive, or more distributionally distant from the fingerprint. It should be interpreted like a unit test: a failing unit test does not prove a program is wrong in every way, but it is a useful signal that something is off.

**Q: Why do you sometimes want more chunks than token limits require?** Variability is easier to express locally. If a high sentence-length standard deviation is desired, rewriting an entire document as one chunk can push the model towards a smooth compromise style. Smaller chunks allow the controller overlay to sample different quantiles across chunks (deterministically), producing natural dispersion while still keeping within global constraints.

**Q: Does chunking risk losing coherence or dropping content?** Yes, and the project treats this as a primary failure mode. It provides: (i) deterministic preservation of protected regions (blockquotes, references, footnotes, citations), (ii) line, word, and paragraph change warnings, and (iii) optional section restoration when headings are missing. Chunking is therefore paired with post-rewrite checks rather than treated as a blind splitting strategy.

**Q: Why keep an explicit JSON fingerprint rather than fine-tune a model?** An explicit fingerprint is inspectable, editable, and versionable. It provides editorial authority: one can see what the system believes about the author’s style and change it.

Fine-tuning and latent embeddings offer considerable potential; however, they present challenges for auditability and interpretability, often conflating content, stylistic features, and safety-related behaviours.

The most common stylometric error is topic leakage - signals attributed to "style" may, in fact, reflect the author's preferred subject matter. Over-confidence presents another risk: stylometric scores may be mistaken for definitive proof, rather than recognized as probabilistic and context-dependent indicators. The approach outlined here attempts to address both pitfalls by filtering out proper-name phrases from lexical signals and focusing on ranges and distributions rather than single thresholds.

Significant changes in word count following a rewrite should be regarded as a potential threat to meaning preservation. A marked reduction may suggest summarisation, while substantial expansion could imply the introduction of extraneous detail. In such cases, the deviations report and count-change warnings should prompt manual review or the tightening of constraints (for instance, by adjusting chunk size, retry thresholds, or imposing stricter requirements to preserve structure).

---


## Licence Notice

This work is licensed under the PolyForm Noncommercial Licence 1.0.0. Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com). See `LICENSE.md` for the full licence text and terms.
