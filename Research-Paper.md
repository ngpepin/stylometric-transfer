 
**Repository:** `stylometric-transfer`  
**Keywords:** stylometry, computational stylistics, authorship attribution, controllable text generation, text style transfer, interpretability

(c) 2026 Nicolas Pepin

---

## Abstract

Stylometric-Transfer provides a practical method for (i) stylometric profiling of an author's writing corpus into an explicit, interpretable JSON artefact (a style fingerprint) and (ii) meaning-preserving style transfer that rewrites new text to conform to the fingerprint using a large language model (LLM). The approach combines classic stylometric measurement - such as punctuation rates and sentence-length distributions - with LLM-mediated synthesis into human-editable constraints, including ranges, histograms, lexicon rules, and rhetorical templates. The fingerprint is formalized as a constraint set, and a constraint-satisfaction decoding view is provided for LLM rewriting, together with compliance scoring based on distributional divergences. Notably, the framework unifies stylometric transfer and humanization by quantifying a conflict‑resolution layer that filters humanization guidelines against fingerprint constraints, and by supporting bounded stochastic variance under explicit controls. This hybrid design allows for an auditable alternative to latent style embeddings, while remaining consistent with established stylometry and text style transfer research.

**Reader’s guide:** Sections 1–3 outline the system’s purpose and rationale. Section 4 explains the measurements with simple examples. Sections 5–6 describe how those measurements become constraints for rewriting. Section 2.5 provides the detailed humanization math and chunk‑sizing logic. Section 7 connects the ideas to the code. The appendices provide further mathematical and algorithmic detail for expert readers.

---

## 1. Introduction

Stylometry examines quantitative signals of writing style for tasks such as authorship attribution and author profiling. A well-known example is the Federalist Papers analysis, where frequent-word statistics support Bayesian inference over disputed authorship ([press.uchicago.edu](https://press.uchicago.edu/ucp/books/book/distributed/I/bo5667096.html?utm_source=chatgpt.com)).

Text style transfer (TST), by contrast, seeks to transform text so that stylistic properties match a target style, while preserving content. A persistent challenge is separating content from style in the absence of parallel data, which has led to cross-alignment approaches and ongoing debate about evaluation and ethics ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com)).

In practical terms, stylometry treats an author’s style as a set of measurable habits: sentence length, punctuation frequency, recurring transitions, and words that are rarely used. Style transfer attempts to generate new text as if those habits were followed, without altering meaning. The difficulty lies in ensuring that a model does not introduce stylistic artefacts or factual drift. This paper addresses how to make that tension measurable and manageable.

A hybrid approach is proposed: style is represented explicitly as a stylometric style fingerprint (in JSON), and an LLM acts as a constrained rewriter, guided by both the fingerprint and locally measured statistics from the author corpus and candidate text. Humanization guidelines are integrated by means of a conflict‑resolution layer, which deterministically filters guideline rules when they contradict fingerprint signals or the input’s stylistic structure. Where enabled, bounded stochastic variance applies a small, seeded number of micro‑edits to reduce AI‑typical uniformity while preserving meaning and constraints.

### 1.1 Stylometry in plain language (what “style” means here)

Stylometry is easiest to understand as a family of *comparisons* rather than a single technique. You measure the same set of signals on two texts (or on a text and a corpus baseline) and ask: do the numbers look like they came from the same distribution of writing habits?

In this project, “style” is not treated as an ineffable vibe. It is operationalized as:

- **Habitual choices**: punctuation density, sentence length dispersion, paragraph cadence, transition habits.
- **Low‑salience lexical defaults**: function‑word balance, common collocations, preferred connectives.
- **Structural preferences**: how often the author uses one‑sentence paragraphs, how headings are formed, how lists are punctuated.

And just as importantly, style is *not* treated as:

- **Topic** (“spacecraft”, “inflation”, “ancient Athens”) or proper‑name density (people/places/brands).
- **Facts** (entities, dates, quantities) - these are meaning and must not be fabricated or “smoothed”.
- **Genre constraints** that come from the medium rather than the author (Markdown boilerplate, license notices).

This distinction matters because many naïve “style models” accidentally learn topic proxies. A useful mental check is: if you swap all named entities (or delete them) but keep the grammar and rhythm, would the author still sound like themselves? If yes, you are probably measuring style rather than topic.

### 1.2 A gentle taxonomy of stylometric features (why these work)

Stylometric features span multiple levels. The system focuses on what is both (a) measurable with low complexity and (b) stable enough to be useful.

1) **Orthography and typography (surface layer)**  
Spelling variants (e.g., Canadian vs US), contractions, dash/ellipsis conventions, heading case, and typographic habits. These are often “easy wins” because they are consistent and have clear editorial relevance.

2) **Lexical statistics (word choice layer)**  
Rare words (words that appear but are not repeated) and lexical avoidance (common words that are absent). These are treated as *soft signals*: they guide the rewrite but should not become brittle prohibitions.

3) **Function words and stance (cognitive layer)**  
Function words (the, of, to, and, but, although, ...) and stance markers (hedges, boosters, directives) are historically strong stylometric signals because authors use them without much conscious control, and they survive topic changes better than content words.

4) **Syntax texture and discourse structure (compositional layer)**  
Paragraph rhythm, sentence‑opener patterns, transition placement, and rhetorical move signals (claim/evidence/counterpoint/concession/synthesis). These are particularly useful in style transfer because they shape the reader’s experience without needing new facts.

The important lesson is not that any one feature “identifies the author”, but that *many weak, interpretable features* can jointly constrain the rewrite in a way that feels coherent while remaining auditable.

### 1.3 Why distributions and ranges beat single numbers

Averages are fragile. Two writers can share the same mean sentence length while having very different distributions:

- Writer A: mostly 12–18 words, few long sentences (low variance).
- Writer B: mixes 6‑word stingers with 45‑word periodic sentences (high variance).

If you only encode the mean, you can accidentally force Writer B into Writer A’s smoothness - an AI‑typical regularity. This is why the fingerprint stores:

- **Histograms** (shape, not just central tendency),
- **Ranges** (tolerance for within‑author variability),
- **Rates** (per‑1000‑word normalizations) rather than raw counts.

In practice, these distributions do double duty: they are (i) a measurement baseline for extracting style and (ii) a post‑rewrite audit that detects compression/expansion and other artefacts.

### 1.4 Humanization, defined carefully (what problem it solves)

“Humanization” is a loaded term. In this paper, it is used narrowly to mean:

> **Reducing AI‑typical artefacts** (over‑regular rhythm, self‑echo, template‑like transitions, suspiciously uniform paragraphing) **without adding facts, changing claims, or hiding provenance**.

It is not an attempt to defeat detectors or to obscure authorship. In fact, the design constraint is the opposite: any humanization mechanism must be explicit, bounded, and observable in a metrics report.

This definition leads to a practical workflow:

1) Extract an author’s *natural variability* (distributions, not just averages).  
2) Rewrite with constraints.  
3) Measure the result.  
4) If the rewrite is “too smooth” or self‑echoing, apply bounded, logged mechanisms (controller overlays; stochastic micro‑ops) that nudge variability *toward* the author’s baseline rather than toward arbitrary randomness.

### 1.5 Business Strategy and Opportunity (why stylometry + humanization matter now)

The diffusion of LLM writing systems changes not only how text is produced but also the economics of editorial work. As text becomes cheaper to generate, the binding constraints shift: attention, trust, compliance, and organisational coherence become scarcer and therefore more valuable. Stylometry and quantitative humanization can be read as managerial responses to that shift - they provide instruments for *governance of writing at scale*.

This section frames the opportunity in management terms: (i) macro drivers, (ii) value creation and capture, (iii) adoption barriers and complements, and (iv) strategic risks and ethics.

#### 1.5.1 Macro drivers: why “writing operations” becomes a strategic capability

Several converging forces make explicit style control and human‑likeness measurement increasingly salient:

- **Supply shock in text production.** Organisations can now generate large volumes of drafts, variants, and personalised messages. This makes *quality assurance* and *voice coherence* limiting factors.
- **Rising cost of trust failures.** Hallucinated facts, inconsistent policy phrasing, and tone drift carry reputational and legal risk. The managerial problem becomes: how do we control and audit what gets published?
- **Regulatory and contractual pressure.** Sectors such as finance, health, and government are increasingly constrained by record‑keeping, disclosure obligations, and controlled language. A measurable style layer supports internal controls and external defensibility.
- **Channel fragmentation.** Brands communicate across web, email, support, social, and internal documentation. A consistent voice is a coordination device; without it, the organisation sounds like many organisations.

In this context, stylometry is not merely an academic method for attribution. It becomes a tool for **operationalising “voice”** into measurable signals that can be monitored, tuned, and enforced.

#### 1.5.2 Value creation mechanisms: what stylometric transfer enables

From a resource‑based view (RBV), an organisation’s distinctive voice can be treated as an intangible asset. LLMs, however, make imitation cheap at the surface level, which increases the value of *systems* that preserve and audit the underlying asset. Stylometric fingerprints support three value‑creating mechanisms:

1) **Editorial productivity without identity dilution.** Faster drafting and rewriting, but with explicit constraints that preserve voice and meaning.
2) **Quality assurance via measurable proxies.** Distributional audits (function words, punctuation densities, rhythm) catch “too‑smooth” or overly templated artefacts that human editors frequently flag but cannot easily quantify.
3) **Traceable governance.** A versioned JSON fingerprint is an auditable policy object: it documents what the system believed the style to be, what constraints were prioritised, and where conflicts occurred (via deviations).

Humanization layers add a complementary capability: they do not aim to “make text undetectable”, but to reduce systematic artefacts (over‑regularity, self‑echo) that degrade perceived quality. In managerial terms, this is **variance management**: introducing bounded, author‑consistent dispersion so that outputs do not converge to the same generic, model‑average cadence.

#### 1.5.3 Strategic use cases and buyer value (where this pays off)

The opportunity spans multiple organisational functions:

- **Brand and comms:** preserve brand voice across campaigns, regions, and agencies; reduce “tone drift” when producing variants.
- **Knowledge management:** standardise internal documentation and onboarding materials while preserving domain precision; keep technical detail intact.
- **Customer support:** rewrite responses for clarity and empathy within bounded tone constraints; reduce template‑like repetition that triggers user distrust.
- **Legal and compliance:** enforce hard avoids and controlled phrasing; maintain consistent hedging levels; log deviations for audit.
- **Publishing and media:** accelerate copy‑editing, house‑style conformity, and localisation while keeping author identity legible.

Across these domains, the managerial value proposition is: *increase throughput while reducing variance in the wrong dimensions* (facts, policy wording, tone) and *increasing variance in the right dimensions* (natural rhythm, non‑templated phrasing).

#### 1.5.4 Competitive landscape: why explicit fingerprints are a differentiator

Many “best‑of‑class” pipelines for generation rely on latent representations (fine‑tuning, adapters, embeddings) that can be powerful but are difficult to inspect. For organisations, this creates governance friction: a latent style vector cannot easily be reviewed by an editor or a compliance officer.

An explicit fingerprint positions differently:

- **Interpretability as a product feature.** Editors can read the rules, modify them, and reason about expected effects.
- **Versionability and change control.** Fingerprints can be reviewed like policy documents: diffed, approved, rolled back.
- **Auditability and blame assignment.** When something goes wrong, deviations and measurements provide a partial causal trail.

In economic terms, the explicit‑constraint approach trades some peak imitation fidelity for lower coordination costs and lower risk costs - a trade many organisations rationally prefer.

#### 1.5.5 Adoption barriers and complements (what must be true to succeed)

The main barriers are not algorithmic; they are organisational:

- **Defining “voice” operationally.** Teams must agree on what matters (clarity, warmth, formality) and accept that some elements are measurable proxies.
- **Domain drift and genre shift.** A fingerprint built on op‑eds may not transfer cleanly to technical manuals. Governance requires multiple fingerprints or conditional policies.
- **Human‑in‑the‑loop processes.** High‑stakes publishing still benefits from editorial review. The system is best seen as an amplifier and consistency engine, not a replacement.
- **Infrastructure reliability.** Chunking, retries, and timeouts are not implementation details; they shape the feasible operational envelope.

The most important complement is **measurement literacy**: users must understand what a metric movement means (e.g., a 25% word‑count drop suggests summarisation risk). Without that literacy, a dashboard becomes decoration rather than control.

#### 1.5.6 Strategic risks and ethics (why “humanization” must be bounded)

Finally, the strategic opportunity comes with governance risks:

- **Misuse risk.** Systems that mimic authorial signatures can be used for impersonation. Guardrails and intended‑use policies matter.
- **Over‑optimisation.** If organisations optimise for superficial “human” metrics, they can harm clarity or accuracy. Metrics must remain subordinate to meaning preservation.
- **Arms‑race framing.** Treating humanization as “evading detectors” is both ethically fraught and strategically brittle. A better framing is editorial quality: avoid artefacts that readers dislike and that editors reject.

For these reasons, this project makes humanization explicit, deterministic by default, bounded when stochastic, and always logged. From a management perspective, this is the key design choice: it turns a vague aspiration (“make it sound human”) into an auditable operational capability.

---

## 2. Related Work

This section provides a brief overview for non-specialists and references for specialists. The central point is that measurements are kept interpretable, and the LLM is tasked with following constraints rather than relying on hidden style embeddings.

### 2.1 Stylometry and Distance-Based Measures

Stylometric authorship attribution typically relies on robust, interpretable features such as word frequency profiles and distance measures. Burrows's Delta and its variants remain widely used; recent work explains the decomposition of feature selection, scaling (for example, z-transformation), and distance metrics, clarifying the effectiveness of Delta-style measures ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com)).

For those new to the field, these methods are intentionally simple, relying on counts and distributions rather than opaque neural features. This makes them suitable for interpretable fingerprints.

### 2.2 Text Style Transfer and Evaluation

Non-parallel TST methods, such as cross-alignment, demonstrate that certain stylistic attributes can be changed without parallel sentence pairs ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com)). Recent surveys discuss broad applications alongside challenges in evaluation and ethical risk, including concerns about misuse for impersonation, and support explicit safeguards and transparency in TST pipelines ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com)).

The literature indicates that style is difficult to define precisely. The approach here is to make style explicit and auditable, rather than learned and hidden.

### 2.3 Corpus Size and Diminishing Returns

In practice, stylometric signals stabilize as corpus size grows. A pragmatic rule of thumb is that **~20–50k words** often yields a stable fingerprint for a single author/genre, while **~100k words** typically captures most steady signals. If the key rates (sentence/paragraph distributions, punctuation per 1k words, function‑word profile, stance rates) drift by **<1–2%** after adding another 10–20k words, you are likely in a diminishing‑returns regime. Additional data is still valuable when the corpus spans multiple genres or eras, or when capturing sparse phenomena such as rare rhetorical moves or infrequent lexical patterns.

### 2.4 Humanization-Aware Stylometric Transfer

Most style-transfer pipelines treat humanization as a separate editing step. Stylometric-Transfer incorporates humanization directly into constraint-guided rewriting by formalizing a conflict-resolution layer: humanization guidelines are applied only when they do not violate fingerprint-derived constraints or the input's structural features (such as heading case or inline-header lists). The guideline list is parsed into structured rules by an LLM (with deterministic fallback), then filtered against fingerprint signals before any rewrite prompt is constructed. This produces a single, auditable framework that balances stylistic fidelity with the removal of AI artefacts, rather than relying on post-hoc edits that may diverge from the author’s voice.

### 2.5 Humanization Mechanisms and Benefits

Humanization in this system is not a vague “make it sound human” instruction. It is a set of explicit, inspectable mechanisms designed to target known LLM artefacts while preserving the author’s voice. These mechanisms include:

- **Conflict‑filtered guidelines**: generic humanization rules are only applied when they do not contradict the fingerprint’s statistical baselines (for example, avoiding “don’t use em‑dashes” when the author’s corpus uses them frequently).
- **Mandatory hygiene rules**: optional hard constraints (such as removing em‑dashes or replacing emojis) are enforced deterministically and recorded in the deviations log.
- **Structural preservation**: blockquotes, citations, footnotes, code spans, and (for non‑fiction) multi‑word quotations are shielded from stylistic edits to prevent false “humanization” changes in non‑authorial content.
- **Bounded stochastic variance**: when enabled, a small, seeded number of micro‑edits (e.g., swapping transition words or dropping filler terms) introduce controlled irregularity without semantic drift.

The practical benefit is a *measurable reduction in AI‑typical uniformity* while keeping the transformation aligned to the author’s measurable habits. Because the humanization layer is explicit, deterministic by default, and logged, it can be audited and tuned without introducing opaque behaviour. In short, humanization is treated as a constrained post‑processing step integrated into the same interpretability framework as stylometric profiling itself.

#### 2.5.1 Quantitative humanization metrics (with math)

When `--metrics` is enabled, a compact quantitative report is emitted for **both the input and the output**. Let a text contain $N$ word tokens and $V$ unique types. Let $f_i$ be the frequency of type $i$.

It is worth stating the intent plainly: these metrics are not an attempt to “detect AI” in a forensic sense. They are operational proxies for a handful of failure modes that practitioners repeatedly observe in LLM rewrites:

- **Over‑regularity** (too even a rhythm, too consistent sentence length, too little local fluctuation),
- **Self‑echo** (repeating the same phrases or n‑grams),
- **Distributional drift** (function‑word and punctuation patterns that stop looking like the target author),
- **Over‑compression** (dropping detail, merging sentences, or collapsing paragraphs in a way that changes pacing).

Because the system’s primary invariants are meaning preservation and interpretability, each metric is deliberately simple and inspectable. The best way to use the report is *comparatively*: look at which metrics move between input and output, and whether they move in a plausible direction for the author and genre. A high aggregate score is not “proof of humanness”; a large, repeated regression in one dimension is a useful debugging signal.

**Lexical diversity family**

$$\mathrm{TTR}=\frac{V}{N}, \qquad C=\frac{\log V}{\log N}, \qquad R=\frac{V}{\sqrt{N}}.$$

The Maas index is computed as

$$a^2=\frac{\log N-\log V}{(\log N)^2},$$

and the reported score uses its inverse (smaller $a^2$ = more diverse).

**Repetition and distributional diversity**

Let $m_1 = N$ and $m_2=\sum_i f_i^2$. Yule’s $K$ and Simpson’s $D$ are:

$$K = 10^4 \cdot \frac{m_2-m_1}{m_1^2}, \qquad D=\frac{\sum_i f_i(f_i-1)}{N(N-1)}.$$

Both are inverted for scoring (lower repetition $\Rightarrow$ higher score).

Self‑echo repetition uses repeated n‑grams. For bigrams and trigrams:

$$r_n=\frac{\sum_{g: c_g\ge 3} c_g}{|G_n|}, \qquad r=\frac{r_2+r_3}{2}.$$

Intuitively, the TTR/Herdan/Guiraud/Maas group tries to capture “how many distinct words are being used”, while the Yule/Simpson/repeat‑rate group tries to capture “how concentrated is the vocabulary and phrasing”. In practice, a rewrite can increase diversity simply by swapping synonyms, so these signals are most valuable when considered alongside meaning‑preservation checks and the author’s baseline.

**Burstiness and rhythm**

For sentence lengths $\ell_s$ (words) and paragraph lengths $\ell_p$ (sentences):

$$B_s=\frac{\sigma(\ell_s)}{\mu(\ell_s)}, \qquad B_p=\frac{\sigma(\ell_p)}{\mu(\ell_p)}.$$

These values characterize rhythmic variability rather than average length.

**Punctuation and function‑word entropy**

Let $c_k$ be counts over punctuation types, and $p_k=c_k/\sum c_k$:

$$H_{\text{punct}}=-\sum_k p_k\log_2 p_k.$$

Function‑word entropy is computed analogously using function‑word counts. If $P$ is the output function‑word distribution and $Q$ is the fingerprint’s distribution, then:

$$D_{\mathrm{KL}}(P\|Q)=\sum_i P_i \log_2\frac{P_i}{Q_i}.$$

The score uses a clipped inverse of $D_{\mathrm{KL}}$ (smaller divergence = higher score).

This family is especially useful because function words are (a) hard to consciously control at scale, and (b) disproportionately stable within an author and genre. When the KL divergence is consistently high across runs, it often indicates one of: the rewrite is too aggressive, the input is out‑of‑domain relative to the fingerprint, or the chunking/constraints encourage paraphrases that inadvertently shift function‑word balance.

**Sentence length divergence vs fingerprint**

Let $P$ and $Q$ be sentence‑length histograms (bins from the fingerprint), and $M=\tfrac{1}{2}(P+Q)$:

$$\mathrm{JSD}(P,Q)=\tfrac{1}{2}D_{\mathrm{KL}}(P\|M)+\tfrac{1}{2}D_{\mathrm{KL}}(Q\|M).$$

The score uses $1-\mathrm{JSD}$ (clipped).

**Character trigram entropy**

Let $t$ be character trigrams over letters (non‑letters removed). With $p_t$ as the trigram distribution:

$$H_{3\text{-gram}}=-\sum_t p_t \log_2 p_t.$$

Character‑level entropy is a deliberately coarse proxy for orthographic “texture”. It reacts to repeated patterns such as boilerplate phrasing, excessive reuse of the same suffixes, and certain forms of templated text. It should be interpreted gently: technical writing can legitimately lower trigram entropy (lots of repeated terminology), while literary writing can raise it.

**Average word length**

$$\bar{\ell}=\frac{1}{N}\sum_{j=1}^{N}\ell(w_j),$$

scored by distance from a neutral reference (e.g., 5 characters).

**Metric summary (symbols and intent)**

| Metric | Symbol / definition | Intent |
| --- | --- | --- |
| Type–token ratio | $\mathrm{TTR}=V/N$ | Surface lexical diversity (length‑sensitive). |
| Herdan’s $C$ | $C=\log V/\log N$ | Length‑normalized diversity. |
| Guiraud’s $R$ | $R=V/\sqrt{N}$ | Length‑normalized diversity. |
| Maas index (inverse) | $a^2=(\log N-\log V)/(\log N)^2$ | Diversity via inverse of $a^2$. |
| Yule’s $K$ (inverse) | $K=10^4(m_2-m_1)/m_1^2$ | Penalizes repetition. |
| Simpson’s $D$ (inverse) | $D=\sum f_i(f_i-1)/(N(N-1))$ | Penalizes repetition. |
| Repeat rate | $r=\tfrac{1}{2}(r_2+r_3)$ | Self‑echo (reused n‑grams). |
| Sentence burstiness | $B_s=\sigma(\ell_s)/\mu(\ell_s)$ | Rhythm variability at sentence level. |
| Paragraph burstiness | $B_p=\sigma(\ell_p)/\mu(\ell_p)$ | Rhythm variability at paragraph level. |
| Punctuation entropy | $H_{\text{punct}}=-\sum p_k\log_2 p_k$ | Variety and balance of punctuation. |
| Punctuation variety | $\#\{k: c_k>0\}$ | Breadth of punctuation usage. |
| Function‑word entropy | $H_{\text{fw}}=-\sum p_i\log_2 p_i$ | Balance of function words. |
| Function‑word KL (inverse) | $D_{\mathrm{KL}}(P\|Q)$ | Divergence from fingerprint profile. |
| Sentence‑length JSD (inverse) | $\mathrm{JSD}(P,Q)$ | Divergence from fingerprint histogram. |
| Char trigram entropy | $H_{3\text{-gram}}$ | Low‑level orthographic diversity. |
| Avg word length | $\bar{\ell}$ | Neutrality vs extreme short/long bias. |

#### 2.5.2 Aggregate 0–100 humanization score

Each metric is normalized into $[0,1]$ by a simple clipping rule. For metrics where “more is better”:

$$s_i=\min\left(1,\frac{x_i}{c_i}\right),$$

and for inverse metrics (e.g., repetition or divergence):

$$s_i=\max\left(0, 1-\frac{x_i}{c_i}\right).$$

Given weights $w_i$ (from `humanization_metrics.weights`), the aggregate score is:

$$S=100\cdot \frac{\sum_i w_i s_i}{\sum_i w_i}.$$

This yields a compact, interpretable 0–100 score while preserving the underlying per‑metric diagnostics for auditability.

Pedagogically, the aggregate score is best treated like a “dashboard needle” rather than a scientific measurement. It is useful for regression testing (did a change systematically degrade outputs?) and for quickly spotting suspicious runs (e.g., an output that is unusually repetitive or unusually flat). It is not designed to compare unrelated authors, and it should not be used to optimize outputs at the expense of meaning preservation.

**How to read the metrics report (practical checklist)**

1) **Start with invariants:** scan the deviations for meaning‑preservation issues (missing sections, altered numerals, broken citations/quotes). If those fail, ignore the score and fix preservation first.
2) **Look for compression artefacts:** large drops in word/paragraph counts often indicate over‑summarization. Confirm whether chunk sizing or overly aggressive rewrite policies are pushing the LLM to compress.
3) **Inspect repetition and rhythm:** if `repetition_inverse` drops or burstiness collapses, the output is likely becoming templated. Consider enabling controller overlays or modest stochastic variance, or reducing chunk size.
4) **Check distributional drift:** if function‑word KL divergence rises and sentence‑length JS divergence rises simultaneously, the output is drifting away from the author’s measurable profile (often an out‑of‑domain input or insufficient constraints).
5) **Use the aggregate score last:** treat it as a regression indicator. Compare runs with identical inputs and tunables; do not compare unrelated authors or genres.

#### 2.5.3 Corpus baselines and controller overlays

To avoid “over‑normalizing” away an author’s natural variability, the fingerprint can embed corpus‑derived **humanization baselines** (rolling windows). For each metric $m$, the corpus is scanned in windows of size $W$ with stride $S$ and summarized into quantiles $\{p10,p25,p50,p75,p90\}$ plus mean. These baseline distributions are stored in `measurements.humanization_baseline` and *withheld from the LLM prompt*.

The key idea is that human writing is not stationary: within a single author, local patches of prose can legitimately be more clipped, more discursive, more list‑heavy, or more punctuated than the global average. A single global target can therefore create the very uniformity we are trying to avoid. Windowed baselines make that variability explicit and measurable.

During rewriting, a **controller overlay** can apply chunk‑level targets sampled from these baselines. For a metric $m$ and quantile $q$, let $v=\mathcal{Q}_m(q)$ and define a symmetric target band:

$$w=\min(\mathrm{max\_width},\ \max(\mathrm{min\_width},\ |v|\cdot \mathrm{range\_pct})), \qquad [v-w,\ v+w].$$

Ratios (e.g., one‑sentence paragraph rate) are clamped to $[0,1]$. These per‑chunk ranges are merged into the fingerprint **only for that chunk**, and the overlay is logged for auditability. Optionally, if the observed metric $o$ falls outside the band by more than a tolerance $\tau$, the next retry receives targeted feedback:

$$\text{deviation} = \frac{|o - \mathrm{clip}(o,[v-w,v+w])|}{\max(\epsilon,2w)} > \tau.$$

Practically, the overlay behaves like a gentle “nudge” rather than a hard constraint: it broadens the set of acceptable outputs by shifting targets between plausible regions of the author’s own distribution. The overlay is computed locally and logged, making it auditable and reversible. The additional feedback loop is intentionally bounded to avoid infinite refinement.

#### 2.5.4 Bounded stochastic variance

A small, seeded perturbation layer introduces controlled irregularity without semantic drift. If $\mathrm{ops}_{1000}$ is the maximum micro‑operations per 1000 words, then for a chunk of $N$ tokens:

$$n_{\text{ops}} = \left\lfloor \mathrm{ops}_{1000}\cdot \frac{N}{1000}\right\rfloor.$$

Allowed operations (e.g., `swap_transition`, `drop_filler`) are sampled deterministically from the seed; all applied operations are logged as deviations. This layer is explicit and bounded by design.

This mechanism is intentionally conservative: it aims to reduce obvious templating and repetition, not to invent stylistic novelty. In terms of engineering tradeoffs, it provides a deterministic “salt” that helps break repeated local patterns while remaining easy to audit (the applied ops are explicit).

#### 2.5.5 Chunk sizing and variance-aware splitting (with math)

Chunking is computed **before** any LLM call so the chunk count is deterministic and fully known. Let $T_{\text{max}}$ be the model’s max prompt token budget, and let $T_{\text{base}}$ be the prompt overhead (system + scaffold + fingerprint). The base budget is:

$$T_{\text{in}}=\max(400,\ T_{\text{max}}-T_{\text{base}}).$$

If a tunable cap $T_{\text{cap}}$ is set, then:

$$T_{\text{in}}=\min(T_{\text{in}},\ T_{\text{cap}}).$$

When variance‑aware chunking is enabled, a scaling factor $f\in[f_{\min},f_{\max}]$ derived from baseline variability is applied:

$$T_{\text{in}}=\max(200,\ \lfloor f\cdot T_{\text{in}}\rfloor).$$

A rough character budget follows $C_{\text{max}}\approx 4\cdot T_{\text{in}}$.

The two competing goals of chunking are:

1) **Reliability:** stay comfortably under the endpoint’s practical limits (timeouts, rate limits, response truncation).
2) **Coherence:** keep enough local context that the LLM can preserve meaning and Markdown structure without over‑summarizing.

Smaller chunks tend to improve reliability (especially on flaky endpoints) but can increase total runtime and can increase the risk of “compression” artefacts at boundaries. Larger chunks improve coherence but can amplify timeouts and make retries more expensive.

**Split strategy.** The tunable `chunk_split_on` chooses the primary unit: paragraph, sentence, or word. If a paragraph exceeds $C_{\text{max}}$, the algorithm falls back to sentence splitting **for that paragraph**; if a single sentence still exceeds the limit, it falls back to word splitting **for that sentence**. Bullet and numbered list lines are treated as sentence units even without terminal punctuation. This avoids oversize chunks while preserving the highest‑level structure possible.

**Perturbation guardrail.** When perturbations are enabled (controller overlay or stochastic variance), the system enforces a minimum chunk count $K_{\min}$ so variability has room to express; the largest chunks are split until $K_{\min}$ is reached or further splitting would be meaningless.

**Worked example.** Suppose $T_{\text{max}}=16000$ and the fingerprint+prompt overhead is $T_{\text{base}}=9000$. Then $T_{\text{in}}=\max(400,7000)=7000$. If `chunking.max_input_tokens=6000`, then $T_{\text{in}}=6000$ and $C_{\text{max}}\approx 24000$ characters. If variance‑aware scaling yields $f=0.85$, then $T_{\text{in}}=\lfloor 0.85\cdot 6000\rfloor=5100$ tokens ($C_{\text{max}}\approx 20400$ chars). At that point, the number of chunks is determined entirely by how much text can be packed under $C_{\text{max}}$, plus any minimum‑chunk enforcement for perturbations.

After rewriting, the system emits the **input** and **output** metric profiles (with the aggregate score) for quick inspection and regression analysis.

**Genre‑aware quotation handling**: the pipeline auto‑detects fiction vs non‑fiction using quote‑density signals (multi‑word quote spans, quoted‑word ratio, and quote‑paragraph ratio). In non‑fiction, multi‑word quotations are excluded from profiling and preserved verbatim during rewriting; in fiction they remain part of the author’s voice. These thresholds are tunable via `config.tunables.json` (see `fiction_detection.*`) and can be overridden explicitly (`--fiction` / `--non‑fiction`).

---

## 3. Problem Setup

Let an author corpus be a set of documents

$$\mathcal{D} = \{d_1,\dots,d_N\}, \quad d_i \in \Sigma^*$$

where $\Sigma$ is a character alphabet and $\Sigma^*$ denotes the set of all finite strings over that alphabet. In plain terms, each document $d_i$ is treated as a sequence of characters (later tokenized into words and sentences by the measurement layer). The index $N$ is simply the number of documents in the corpus.

Define:

- An interpretable feature extractor $\phi: \Sigma^* \to \mathbb{R}^K$ that produces measurable statistics (rates, histograms, counts).
- A style fingerprint $\mathcal{F}$ that stores target statistics, distributions, and discrete constraints (lexicon rules, templates).
- A rewriter $\mathcal{R}_\theta$ (LLM with parameters $\theta$) mapping input text $x$ to output $y$:

$$y = \mathcal{R}_\theta(x \mid \mathcal{F}).$$

Here the notation “$\mid \mathcal{F}$” means that rewriting is *conditioned* on the fingerprint: the same input $x$ can yield different outputs depending on the constraints encoded in $\mathcal{F}$. This makes the fingerprint the explicit, versionable “control surface” for the LLM.

The primary constraint is meaning preservation: no new facts, claims, or examples; entities and numerals are preserved unless explicitly permitted.

In essence, the author’s writing habits are measured and compressed into a JSON fingerprint. The LLM is then asked to rewrite new text so that these habits are respected, while the underlying meaning remains unchanged.

---

## 4. Stylometric Measurements

This section describes the simple, interpretable statistics measured by the system. The aim is not linguistic perfection, but stability, explainability, and auditability. These measurements constitute the ground truth the LLM must follow.

Before describing individual features, it helps to be explicit about *what* is being counted and *what is excluded*.

**Tokens, types, and normalisation.** The code uses lightweight tokenisation (whitespace and punctuation heuristics) to obtain approximate word tokens. For many stylometric signals, this is sufficient because the goal is comparative stability across runs, not linguistic annotation. The key step is normalisation:

- Counts are scaled per 1000 words (or per 100 words for certain densities) so that a short and a long document become comparable.
- Histograms are normalised to probability distributions so that divergence measures are meaningful.

This choice has an interpretability benefit: every reported number can be traced back to “how many events happened” and “how many opportunities there were for the event”.

**Author‑voice filtering.** Not all text in a corpus represents the author’s voice. Blockquotes, reference sections, footnotes, inline citations, and boilerplate notices (copyright/terms/privacy) are filtered out of measurement and excerpt selection. In addition, the pipeline detects whether a text is likely fiction or non‑fiction. In non‑fiction, multi‑word quoted passages are treated as quotations and are excluded from profiling (and preserved during rewriting), whereas in fiction, quoted dialogue is treated as part of the author’s style.

**Why these heuristics are acceptable.** A reader might reasonably ask: are these measurements “scientifically pure”? The answer is: they are engineered for a particular purpose. For constraint‑guided rewriting, the best measurement is the one that is (i) stable, (ii) cheap to compute, (iii) easy to reason about, and (iv) aligned with what editors recognise as stylistic drift. If a metric is too fragile or too expensive, it will not be usable in practice.

### 4.1 Rate and Density Features

Let $W(d)$ be an approximate word-token count and $C_e(d)$ the count of an event $e$ (such as commas). Define per-1000-word rates:

$$r_e(d) = 1000 \cdot \frac{C_e(d)}{\max(1, W(d))}.$$

The $\max(1,\cdot)$ guard prevents division by zero on degenerate or heavily filtered text. The “per 1000 words” normalization makes rates comparable across documents and corpora of different sizes.

**Worked example (rate normalization).** Suppose a document has $W(d)=2000$ words and $C_{\text{comma}}(d)=120$ commas. Then:

$$r_{\text{comma}}(d)=1000\cdot\frac{120}{2000}=60\ \text{commas per 1000 words}.$$

The fingerprint stores targets as tolerance intervals:

$$r_e \in [\underline{r}_e, \overline{r}_e],$$

reflecting intra-author variability across topics and subgenres. Intuitively, the interval says: “the author’s comma rate is usually between these bounds, so a rewrite that falls far outside is likely drifting stylistically (or compressing/expanding the prose).”

### 4.2 Histogram Features

For sentence lengths $\ell_1,\dots,\ell_m$ (in words), define a binned histogram

$$\mathbf{h} \in \Delta^{B-1}, \quad h_b = \frac{1}{m}\sum_{i=1}^m \mathbf{1}[\ell_i \in \text{bin}(b)],$$

where $\mathbf{1}[\cdot]$ is an indicator function (1 if the condition holds, 0 otherwise), $\Delta^{B-1}$ is the probability simplex (all nonnegative vectors that sum to 1), and bins are ordinal intervals (for example, $<10$, 10–17, 18–25, ...). In words: we count how many sentences fall into each length bin and then normalize by the total number of sentences $m$.

**Worked example (histogram).** Suppose we observe $m=5$ sentence lengths (in words): $\ell = [8, 12, 12, 23, 41]$, and bins are $<10$, 10–17, 18–25, 26–40, $>40$. Then the histogram mass is:

- $h_{<10}=1/5$ (one sentence of length 8),
- $h_{10-17}=2/5$ (two sentences of length 12),
- $h_{18-25}=1/5$ (one sentence of length 23),
- $h_{26-40}=0$,
- $h_{>40}=1/5$ (one sentence of length 41).

Paragraph rhythm is also captured with a one-sentence paragraph rate:

$$\rho_{1}(d) = \frac{\#\{\text{paragraphs with exactly one sentence}\}}{\max(1,\ \#\{\text{paragraphs}\})}.$$

This rate serves as a stylistic baseline. Excessive one-sentence paragraphs are flagged as an AI artefact only if they exceed the author’s $\rho_1$ range. The key pedagogical point is that “one-sentence paragraphs” are not inherently wrong; they are a stylistic choice. The fingerprint makes that choice measurable so the system can distinguish “authorial habit” from “LLM overuse”.

**Worked example (one-sentence paragraph rate).** If a document has 10 paragraphs and 4 contain exactly one sentence, then $\rho_1=4/10=0.4$.

### 4.3 Rare-Word Signals

Let $f(w)$ be the corpus frequency of a token $w$ after filtering stopwords, numerals, and short tokens. A rare-word list is recorded:

$$\mathcal{R} = \{w : f(w) \le c_{\max}\},$$

where $c_{\max}$ is a small threshold (for example, 2–5 occurrences). This set is not a “ban list”; it is a cue. If a word appears only once in a large corpus, overusing it in a rewrite can sound unlike the author even if it is semantically acceptable. Conversely, domain-specific corpora can legitimately contain rare technical terms; this is why the system treats the list as a hint and allows overrides.

### 4.4 Rhetorical and Epistemic Signals

Beyond surface statistics, the system tracks interpretable rhetorical moves and certainty bands. Let $\mathcal{S}$ denote sentences in the corpus. For a rhetorical marker set $\mathcal{M}_k$ (e.g., claim or concession indicators), define:

$$r_k = 1000 \cdot \frac{\#\{s \in \mathcal{S} : s \text{ contains any marker in } \mathcal{M}_k\}}{\max(1, W)}.$$

Here $W$ is the corpus (or chunk) word count used for normalization. We compute rates for claim, evidence, counterpoint, concession, and synthesis markers. Epistemic stance bands (speculative, probabilistic, assertive, directive) are computed using simple token lists. These signals are intentionally approximate; they are used to set tolerances, not to classify sentences perfectly. The pedagogical takeaway is that we are measuring *tendencies* (how often a move appears), not doing discourse parsing.

### 4.5 Paragraph Cadence and Discourse Marker Position

Let $s_1$ and $s_n$ be the opening and closing sentences of a paragraph. The system records distributions of opening/closing sentence lengths (means and standard deviations) to capture cadence. It also tracks the position of discourse markers (e.g., “however”, “therefore”) as start‑of‑sentence versus mid‑sentence rates:

$$r_{\text{start}} = 1000 \cdot \frac{\#\{\text{markers at sentence start}\}}{\max(1, W)}, \quad r_{\text{mid}} = 1000 \cdot \frac{\#\{\text{markers mid‑sentence}\}}{\max(1, W)}.$$

These features help preserve where transitions tend to appear in the author’s voice. Many writers have strong positional habits (“However,” at the start vs “..., however, ...” mid-sentence). Preserving these positions often matters more than preserving the exact marker choice.

### 4.6 Repetition Signals (Self‑Echo)

AI‑generated text often repeats phrases locally. To detect this, we measure repetition rates for bigrams and trigrams:

$$\rho_n = \frac{\sum_{g \in \mathcal{G}_n} \mathbf{1}[c(g) \ge c_{\min}] \cdot c(g)}{\max(1, |\mathcal{G}_n|)},$$

where $\mathcal{G}_n$ is the multiset of n‑grams and $c_{\min}$ is a small repeat threshold (default 3). The numerator counts repeated n-gram *occurrences* beyond the threshold, and the denominator normalizes by the total number of n-grams. These rates define ceilings for acceptable self‑echo. In words: if an output repeats the same bigram/trigram too often, it becomes perceptibly templated.

### 4.7 Delta-Style Diagnostics (Optional)

Although Stylometric-Transfer is not an authorship attribution system, Delta-style distances can serve as diagnostic measures of stylistic proximity. Following standardization and Manhattan-style aggregation:

$$\Delta(d,d') = \frac{1}{K}\sum_{k=1}^K \left|z_k(d) - z_k(d')\right|,$$

where $z_k$ is the z-transformed version of feature $k$ (subtract the corpus mean and divide by the corpus standard deviation). The absolute difference is then averaged across features, producing a distance-like score: smaller values mean “more similar under these normalized features”. Detailed explanations of Delta variants motivate this approach ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com)).

---

## 5. The Style Fingerprint as a Constraint Model

The fingerprint is treated as a set of weighted constraints:

$$\mathcal{F} = \{(\psi_j, \mathcal{C}_j, w_j)\}_{j=1}^J,$$

where:
- $\psi_j(y)$ is a measurable statistic of output text (for example, comma rate or histogram vector).
- $\mathcal{C}_j$ is an admissible set (range, divergence tolerance, forbidden list).
- $w_j$ is a weight (priority).

This notation is meant to make the engineering idea explicit: each constraint is “a measurement you can compute” ($\psi_j$), plus “what is allowed” ($\mathcal{C}_j$), plus “how important it is” ($w_j$). The system is successful when it can find an output $y$ whose measurements land inside many admissible regions, with the highest-weight regions satisfied most reliably.

Typical constraint types include:

1. Range constraints: $\psi_j(y) \in [a,b]$  
2. Histogram constraints: $D(\mathbf{h}^*, \mathbf{h}(y)) \le \tau$  
3. Lexicon constraints: forbidden phrases or words; preferred synonyms; avoid-rare words $\mathcal{R}$  
4. Template constraints: rhetorical move frequency bounds

The JSON format introduces practical control fields, such as `priority_order` and `strictness`, to specify constraint precedence.

**Plain-language explanation:** The fingerprint functions as a weighted checklist. Certain items are strict (for example, “never use em-dashes”); others are soft (such as “prefer shorter sentences”). The system tracks how closely the output adheres to each requirement.

**Constraint normalization and controller overlays:** To keep constraints compact and interpretable, the system deterministically de‑duplicates `rewrite_policy` clauses and filters `priority_order` to short, token‑like entries. In addition, a corpus‑derived variability baseline can drive per‑chunk target overlays during rewriting: a small, deterministic controller samples quantiles of within‑author variability (e.g., sentence length or punctuation density) and nudges chunk‑level targets so the output reflects natural intra‑author dispersion without changing the core fingerprint. When perturbations are enabled (controller overlays or stochastic variance), the pipeline enforces a minimum chunk count so variability can be expressed. These overlays are applied locally and logged for auditability.

---

## 6. Constraint Satisfaction Decoding and Compliance Scoring

During generation, the language model produces a candidate rewrite, which is immediately measured using the same metrics applied to the author’s corpus. This closes the loop: if the output diverges, the model receives precise feedback on which metrics have drifted and can be prompted for correction.

This section develops the mathematical framing of rewriting as a constraint satisfaction problem.

### 6.1 Soft-Constrained Objective

Let $p_\theta(y\mid x)$ denote the model’s conditional probability of output $y$ given input $x$. The soft-constrained objective is defined as:

$$\max_{y \in \mathcal{Y}} \; \log p_\theta(y \mid x) - \lambda\, \mathcal{L}_{style}(y;\mathcal{F}) - \mu\,\mathcal{L}_{sem}(y;x),$$

where:
- $\mathcal{L}_{style}$ penalizes deviation from the style fingerprint.
- $\mathcal{L}_{sem}$ penalizes semantic drift (estimated conservatively via invariants or, optionally, semantic similarity models).

Pedagogically: $\log p_\theta(y\mid x)$ is the model’s “fluency” preference (how likely the model thinks the rewrite is), while $\mathcal{L}_{style}$ and $\mathcal{L}_{sem}$ are penalties that pull the output back toward the desired style and meaning. The hyperparameters $\lambda$ and $\mu$ control how strongly we trade off “what the model would naturally say” against “what we require”.

A typical decomposition is:

$$\mathcal{L}_{style}(y;\mathcal{F}) = \sum_{j=1}^J w_j\, \ell_j(\psi_j(y), \mathcal{C}_j).$$

This expresses style loss as a weighted sum of per-constraint penalties. If a particular constraint is crucial (e.g., orthography, hard avoids, or a tight sentence-length range), it receives higher weight and therefore dominates the loss.

Examples of penalties include:

**Range penalty** for $\mathcal{C}_j=[a,b]$:

$$\ell_j(v,[a,b]) = \big(\max(0,a-v)\big)^2 + \big(\max(0,v-b)\big)^2.$$

This penalty is zero when $v$ lies inside the interval $[a,b]$. If $v$ falls below $a$, it grows quadratically with the gap $(a-v)$; if it exceeds $b$, it grows quadratically with $(v-b)$. Quadratic growth is a common choice because it increasingly discourages large violations while being gentle near the boundary.

**Worked example (range penalty).** If the target interval is $[a,b]=[5,8]$ and the observed value is $v=3$, then:

$$\ell(v,[a,b])=(5-3)^2+0=4.$$

If instead $v=9$, then:

$$\ell(v,[a,b])=0+(9-8)^2=1.$$

**Histogram penalty** using KL divergence:

$$\ell_j(\mathbf{h},\mathbf{h}^{*}) = D_{KL}(\mathbf{h}^{*}\|\mathbf{h}) = \sum_{b=1}^B h_b^{*} \log \frac{h_b^{*}}{\max(\epsilon,h_b)}.$$

Here $\mathbf{h}^*$ is the target histogram from the fingerprint and $\mathbf{h}(y)$ is the output histogram. KL divergence is asymmetric; it answers the question “how surprising is the target distribution under the output distribution?”. The $\epsilon$ guard prevents division by zero when a bin is empty. Other divergences (e.g., Jensen–Shannon or Wasserstein) can be substituted depending on whether ordinal bin geometry matters.

**Worked example (KL on histograms).** Let the target be $\mathbf{h}^*=[0.5,0.5]$ and the output be $\mathbf{h}=[0.9,0.1]$ over two bins. Then:

$$D_{KL}(\mathbf{h}^*\|\mathbf{h}) = 0.5\log\frac{0.5}{0.9} + 0.5\log\frac{0.5}{0.1}.$$

The first term is negative (the output over-allocates mass to bin 1) while the second is strongly positive (the output under-allocates bin 2). The sum is positive, reflecting that the output distribution does not match the target well.

For ordinal bins, the Wasserstein distance $W_1$ may be preferable; the implementation can support either approach.

### 6.2 Hard Constraints (Feasibility)

Some constraints are best enforced as hard feasibility requirements:

- Entity and number preservation constraints must be satisfied unless explicitly overridden.
- Hard forbidden lexicon constraints (for instance, terms that must not appear).

The feasible set is defined as:

$$\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y\in \mathcal{Y} : \forall j\in \mathcal{H},\; \psi_j(y)\in \mathcal{C}_j\},$$

where $\mathcal{H}\subseteq\{1,\dots,J\}$ indexes the hard constraints.

In words, $\mathcal{Y}_{hard}$ is the subset of all possible rewrites that satisfy every non-negotiable rule. If meaning preservation forbids changing a quoted passage, then any rewrite that alters it is outside the feasible set. This is the mathematical way of expressing “some rules are deal-breakers.”

Decoding then becomes:

$$\max_{y\in \mathcal{Y}_{hard}(x,\mathcal{F})} \log p_\theta(y\mid x) - \lambda\sum_{j\notin \mathcal{H}} w_j\,\ell_j(\psi_j(y),\mathcal{C}_j) - \mu\,\mathcal{L}_{sem}(y;x).$$

This is the same soft-constrained objective as before, but restricted to feasible outputs. In implementations that cannot perform true constrained decoding, the system approximates this restriction via prompt instructions, frozen placeholders, and post-hoc repairs.

### 6.3 Practical Constraint-Satisfaction Decoding Procedure

In practice, exact constrained decoding over $\mathcal{Y}_{hard}$ is rarely available in production language models. Stylometric-Transfer approximates constraint satisfaction through instruction prompting, self-audit, and repair.

A practical approximation proceeds as follows:

1. Generate a candidate rewrite $y^{(0)}$ using explicit instructions that encode $\mathcal{F}$.
2. Compute local measurements $\phi(y^{(t)})$ and audit for constraint violations.
3. If violations are detected, re-prompt the model with a structured report to obtain $y^{(t+1)}$.
4. Stop when compliance exceeds a threshold or an iteration limit is reached.

### 6.4 Compliance Scoring

A normalised compliance score $S(y;\mathcal{F})\in[0,1]$ aggregates constraint satisfaction:

$$S(y;\mathcal{F}) = \sigma\Big(\sum_{j=1}^J w_j\, s_j(y)\Big), \quad \sum_j w_j = 1,$$

where $\sigma$ is a squashing function (such as the identity clipped to $[0,1]$ or a logistic function), and $s_j(y)\in[0,1]$ is a per-constraint score.

The normalization $\sum_j w_j=1$ makes the weighted sum interpretable: it becomes an average of per-constraint scores. The squashing function $\sigma$ is optional; it can be used to make the score more sensitive to low compliance (logistic) or simply to clip numerical noise (identity + clipping).

**Worked example (weighted compliance).** Suppose we track three constraint scores $s=[0.9, 0.7, 0.8]$ with weights $w=[0.5, 0.2, 0.3]$. Then the weighted sum is:

$$\sum_j w_js_j = 0.5\cdot 0.9 + 0.2\cdot 0.7 + 0.3\cdot 0.8 = 0.83.$$

If $\sigma$ is identity-with-clipping, then $S=0.83$. This reads as “83% compliant under these weighted checks”.

Examples include:

- **Range score**:
$$s_j(y) = 1 - \min\left(1, \frac{\ell_j(\psi_j(y),[a,b])}{\kappa_j}\right)$$
for a scaling constant $\kappa_j>0$.

Here $\kappa_j$ turns a raw penalty into a unitless score: if the penalty equals $\kappa_j$, the score reaches 0. This allows different constraint families (rates, divergences, boolean checks) to be combined into a single compliance number without one metric dominating purely due to scale.

- **Histogram score** (KL):
$$s_j(y) = \exp\big(-\alpha_j\, D_{KL}(\mathbf{h}^*\|\mathbf{h}(y))\big).$$

The exponential mapping is a standard way to convert a divergence into a score: small divergences produce scores near 1, while large divergences decay smoothly toward 0. The parameter $\alpha_j$ controls how sharply the score drops.

- **Lexicon hard constraint score**:
$$s_j(y)=\mathbf{1}[\text{no forbidden term appears}].$$

This is the simplest possible score: either the forbidden term appears (0) or it does not (1). Hard constraints are often treated as boolean because any violation is unacceptable.
This compliance score supports reporting (via `validators.weights` and `checks` in JSON), iterative repair thresholds, and regression tests for stability.

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

These design choices reflect stylometric traditions that favour interpretable features, as well as concerns about evaluation and ethical risk in text style transfer. ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))

---

## 8. Ethical Considerations

Transparency and non-impersonation are central. The system is intended for personal writing, editing support, and self-modelling, not for imitating living authors without consent.

Stylometric-Transfer could be misapplied for impersonation; recent surveys highlight ethical risks and the need for safeguards. ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com))

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

This project is intentionally pragmatic: it aims for *auditable control* more than maximum imitation fidelity. That design choice yields predictable strengths (interpretability, editability, local grounding) and equally predictable limitations. This section summarizes the important ones, with an emphasis on what can go wrong in real usage.

### 9.1 Stylometry limitations

**Genre and domain dependence.** Many stylometric signals are not “author essence”; they are *genre constraints*. A newspaper leader, a technical manual, and a literary short story produce different punctuation rates, sentence distributions, and function‑word profiles even for the same writer. If the input is out‑of‑domain relative to the corpus used to build the fingerprint, the system can only satisfy some constraints without threatening meaning preservation. In practice this manifests as more deviations, more retries, or an overly conservative rewrite.

**Topic and named‑entity leakage.** Frequent n‑grams and “common phrases” are especially vulnerable to being dominated by topical entities (people, places, organizations). Deterministic filtering and blacklist strategies reduce this, but cannot eliminate the underlying issue: a corpus can be stylistically consistent yet topically narrow. The solution is not more filtering alone; it is also broader corpora (within the same genre) or a higher‑level representation of lexical preference (e.g., conceptual rather than named entities).

**Sparse features and small corpora.** Rare rhetorical moves, rare punctuation events (e.g., semicolons), and “rare words” require enough data to stabilize. For small corpora, estimates are noisy and can lead to brittle targets. The project mitigates this by favoring ranges and histograms over point targets, but sparse phenomena remain hard.

**Language and tokenization assumptions.** The measurement layer is intentionally lightweight; it uses heuristic sentence/paragraph splitting and simple token counting. This is robust for plain English prose but less reliable for mixed scripts, heavy math, tables, code‑dense documents, or languages with different punctuation conventions. Where segmentation fails, all downstream metrics (and therefore targets) inherit the error.

**Interpretability vs completeness.** “Explicit and simple” measurements leave out many subtle signals: syntax trees, discourse relations, register shifts, pragmatic implicatures, and long‑range narrative structure. Those omissions are deliberate for auditability, but they cap the ceiling of what can be captured quantitatively.

### 9.2 Humanization limitations

**Humanization is not a single axis.** “Human‑like” prose is not a stable target: authors differ, genres differ, and even within one document the distribution shifts (opening vs closing, dialogue vs exposition, summary vs detail). Any single scalar score necessarily compresses nuance and should be treated as a monitoring signal, not an optimization objective.

**Metric gaming and Goodhart’s law.** If an operator tunes prompts or post‑processing to maximize the aggregate score $S$, the system may learn to “game” the proxies (e.g., add synonym churn to boost diversity, or inject punctuation to raise entropy) without improving the text. The project defends against this by (a) keeping meaning preservation and structural invariants primary, (b) logging per‑metric diagnostics, and (c) bounding perturbations. But the risk is intrinsic to proxy scoring.

**Local edits can cause global drift.** Micro‑operations and controller overlays are designed to be small, but they can interact: a swap of transition words can change sentence boundaries; punctuation edits can change clause structure; a change in paragraphing can affect cadence metrics. The system therefore keeps the perturbation budget low and uses chunk‑level overlays rather than global optimization, but complex interactions remain possible.

**Chunk boundary artefacts.** Chunking is essential for reliability, but it can introduce boundary effects: repeated openers at chunk starts, inconsistent local voice, or over‑compression in smaller chunks. Variance‑aware chunking and minimum‑chunk enforcement mitigate this, but there is no free lunch: smaller chunks improve reliability, larger chunks improve coherence.

**Quotations are ambiguous.** The fiction vs non‑fiction heuristic treats multi‑word quotes as quotations in non‑fiction, but real documents contain mixed modes (quoted slogans, reported speech, epigraphs, and stylized “air quotes”). The system allows explicit overrides, but automatic classification will sometimes be wrong.

**Hard hygiene rules can be stylistically incorrect.** Deterministic bans (e.g., “no em dashes”) increase consistency across outputs, but they can conflict with the author’s authentic style. The project treats these as optional tunables because they are editorial choices, not stylometric truths.

### 9.3 LLM and systems limitations

**Model variability and endpoint reliability.** Even with fixed prompts, LLMs can be nondeterministic (sampling, backend variation), and endpoints can be slow or flaky (timeouts, transient 5xx). The pipeline uses retries and bounded refinement loops, but it cannot guarantee that every request succeeds within a fixed time budget.

**Meaning preservation is not formally verified.** The system enforces meaning preservation primarily through prompting, structural freezing, and deviation reporting. It does not yet provide a formal semantic equivalence proof. As a result, human review remains necessary for high‑stakes text.

**Evaluation remains multi‑objective.** The system optimizes for multiple goals: meaning preservation, adherence to explicit style constraints, reduced AI‑typical artefacts, and Markdown validity. These goals sometimes conflict. The project’s design makes those conflicts explicit (via priorities and deviations), but it cannot eliminate them.

**Comparison context.** Many production-grade pipelines combine LLMs with training-time specialization, agentic critique loops, or detector-guided objectives. Stylometric-Transfer deliberately prioritizes auditability and explicit control; Appendix F provides a structured comparison against common "best-of-class" alternatives and where each tends to win.

---

## 10. Conclusion

For those new to the field, the main point is that classic stylometry and modern language models can be combined without sacrificing interpretability. For experts, the contribution is a concrete, auditable constraint model and a measurable conflict-resolution layer that unifies stylometric transfer and humanisation.

Stylometric-Transfer connects classic stylometry and LLM-based rewriting by pairing interpretable, versionable style models with constraint-guided generation. The explicit JSON fingerprint enhances auditability and editorial control, drawing on established stylometric measurement and style transfer research. ([press.uchicago.edu](https://press.uchicago.edu/ucp/books/book/distributed/I/bo5667096.html?utm_source=chatgpt.com))

---

## References

- Mosteller, F., & Wallace, D. L. *Inference and Disputed Authorship: The Federalist.* Addison-Wesley (1964). ([archive.org](https://archive.org/details/inferencedispute00most?utm_source=chatgpt.com))
- Evert, S., et al. "Understanding and explaining Delta measures for authorship attribution." *Digital Scholarship in the Humanities* (2017). ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))
- Shen, T., Lei, T., Barzilay, R., & Jaakkola, T. "Style Transfer from Non-Parallel Text by Cross-Alignment." (2017). ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com))
- Mukherjee, S., et al. "A Survey of Text Style Transfer: Applications and Ethical Implications." (2024). ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com))
- Hu, Z., et al. "Text Style Transfer: A Review and Experimental Evaluation." *KDD Explorations* (PDF). ([kdd.org](https://www.kdd.org/exploration_files/vol24issue1_2._Text_Style_Transfer__A_Review_and_Experimental_Evaluation.pdf?utm_source=chatgpt.com))

---

## Appendix A. Methods (Pseudocode)

This appendix presents pseudocode for the fingerprinter (extractor) and rewriter stages. It is intended to be accessible to those new to stylometry and serves as a procedural summary of the system.

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
        t ← filter_non_voice(t)  # blockquotes, references, footnotes, inline citations, boilerplate; drop multi‑word quotes if non‑fiction
        if length(t) ≥ MIN_LEN:
            texts.append(t)

    M ← compute_measurements(texts)
        # includes: sentence/paragraph histograms, punctuation rates,
        # contractions/oxford comma, function words, stance signals,
        # sentence-openers/templates, n-grams

    if phrase_validation_enabled:
        V ← prefilter_proper_names(M.common_phrases)
        V ← validate_common_phrases(V, llm=C)
        M.common_phrases_validation ← V

    E ← pick_representative_excerpts(files, char_budget=B, voice_scoring=on)
    L ← load_lexicon_hints(optional)

    prompt ← build_fingerprint_prompt(schema=S, measurements=M, excerpts=E,
                                      lexicon_hints=L, model=C.model)
    if prompt_too_large(prompt, C.max_prompt_tokens):
        batches ← chunk_excerpts(E, budget=C.max_prompt_tokens)
        partials ← []
        for b in batches:
            prompt_b ← build_fingerprint_prompt(schema=S, measurements=M, excerpts=b,
                                                lexicon_hints=L, model=C.model)
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
    # enforce invariants
    F.schema_version ← default_if_missing(F.schema_version, "1.0.0")
    F.measurements ← M  # embed verbatim

    write_json(out, F)
    return F
end procedure
```

Common‑phrase validation includes a deterministic prefilter (honorifics + capitalization‑ratio heuristics) and an LLM phrase‑ranking step that drop likely proper‑name phrases (e.g., person/place/org names) before final selection.

Rare‑word selection can be optionally ranked by the same LLM validation call (shared with common‑phrase filtering), using capitalization ratios to de‑prioritize proper names before truncation.

### A.2 Rewrite (Fingerprint + Draft → Styled Draft)

**Inputs:** Fingerprint $\mathcal{F}$, input Markdown $x$, language model $\mathcal{R}_\theta$  
**Output:** Rewritten Markdown $y$ and deviations report

```text
procedure APPLY_FINGERPRINT(fingerprint F, markdown_path in, output_path out, llm_config C):
    x ← read_text(in)
    x ← strip_base64_images(x)
    x ← mask_non_voice_blocks(x)  # blockquotes, references, footnotes (and multi‑word quotes if non‑fiction)
    x ← mask_inline_citations(x)
    Mx ← compute_measurements(filter_non_voice(x))  # non‑fiction excludes multi‑word quotes
    Hraw ← load_humanizer_guidelines(optional)
    H ← parse_humanizer_rules_llm(Hraw)  # default
    if H is empty:
        H ← parse_humanizer_rules_regex(Hraw)
    H ← filter_conflicting(H, F.measurements, F.targets)

    style_feedback ← null
    for r in 0..R:
        prompt ← build_rewrite_prompt(fingerprint=F, input_measurements=Mx,
                                      input_text=x, style_feedback=style_feedback,
                                      humanizer_guidelines=H)
        raw ← call_llm_chat_completions(prompt, C)

        if is_valid_json(raw):
            obj ← parse_json(raw)
        else:
            obj ← parse_json(call_llm_chat_completions(build_json_repair_prompt(raw), C))

        y ← obj.final_markdown
        y ← restore_placeholders(y)  # non-voice blocks, citations, base64
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

### A.3 Optional: Audit-and-Repair Loop (Constraint Satisfaction Approximation)

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

### A.4 Worked micro-examples (numbers you can compute by hand)

This subsection provides a few small examples that mirror what the code does. They are intentionally tiny so readers can verify the arithmetic.

**A.4.1 Punctuation rate per 1000 words.** If a filtered text segment contains 500 words and 18 semicolons, the semicolon rate is:

$$r_{\text{semicolon}} = 1000\cdot\frac{18}{500} = 36\ \text{per 1000 words}.$$

**A.4.2 One-sentence paragraph rate.** If there are 20 paragraphs and 7 have exactly one sentence:

$$\rho_1 = 7/20 = 0.35.$$

**A.4.3 Controller overlay band.** Suppose a baseline quantile yields $v=5.0$ for comma density per 100 words (commas/100w). If `range_pct=0.15`, `min_width=0.05`, `max_width=6.0`, then:

$$w=\max(0.05, |5.0|\cdot 0.15)=0.75,$$

so the target band becomes $[4.25,5.75]$ for that chunk.

**A.4.4 Burstiness.** If sentence lengths are $\ell_s=[10, 10, 30]$, then $\mu=16.67$ and (population) $\sigma\approx 9.43$, so:

$$B_s=\sigma/\mu \approx 0.57.$$

---

## Appendix B. Formal Constrained Decoding Framing

This appendix reframes the decoding process as a standard constrained optimisation or constrained Markov decision process. For those less familiar with the formalism, the essential point is that the LLM is directed by measurable constraints, not hidden embeddings.

### B.1 Constrained Maximum A Posteriori Decoding

Let $p_\theta(y\mid x)$ represent the base LLM distribution. Constraints are indexed by $j=1,\dots,J$, each with statistics $\psi_j(y)$ and admissible sets $\mathcal{C}_j$.

The feasible set of hard constraints is defined as:

$$\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y : \forall j \in \mathcal{H},\; \psi_j(y) \in \mathcal{C}_j\}$$

The constrained MAP problem becomes:

$$\hat y = \arg\max_{y \in \mathcal{Y}_{hard}(x,\mathcal{F})} \; \log p_\theta(y\mid x)$$

In practice, $\mathcal{Y}_{hard}$ cannot be enumerated directly. The problem is therefore relaxed using a Lagrangian penalty formulation:

$$\hat y = \arg\max_{y \in \mathcal{Y}} \; \log p_\theta(y\mid x)
- \sum_{j=1}^J \lambda_j \cdot g_j(\psi_j(y))
- \mu \cdot \mathcal{L}_{sem}(y;x),$$

where:

- $g_j(\cdot)$ is a non-negative violation function, with $g_j(v)=0$ if and only if $v \in \mathcal{C}_j$
- $\lambda_j \ge 0$ are Lagrange multipliers derived from `validators.weights`
- $\mathcal{L}_{sem}$ enforces preservation of meaning

This approach aligns with the standard soft-constrained decoding paradigm in controllable generation and lexically constrained decoding.

---

### B.2 Projection View

Alternatively, rewriting may be seen as projecting an unconstrained sample $y^{(0)} \sim p_\theta(\cdot \mid x)$ onto the admissible region:

$$\hat y = \Pi_{\mathcal{C}}(y^{(0)}) = \arg\min_{y} \; d(y, y^{(0)}) + \sum_j \lambda_j g_j(\psi_j(y)),$$

where $d(\cdot,\cdot)$ measures edit or semantic divergence. In practice, $\Pi_{\mathcal{C}}$ is approximated by LLM self-repair passes, each guided by explicit audit reports.

---

### B.3 Constrained Markov Decision Process (CMDP) Interpretation

Token generation can be framed as a CMDP:

- States: $s_t = y_{1:t}$
- Actions: $a_t = y_{t+1}$
- Reward: $r_t = \log p_\theta(a_t\mid s_t,x)$
- Costs: $c_{j,t}$, which accumulate toward $\psi_j(y)$

with terminal constraints:

$$\mathbb{E}\Big[ \sum_t c_{j,t} \Big] \le \tau_j$$

This formulation clarifies that the system approximates policy optimisation under global style budgets, implemented through instruction-guided generation and post-hoc repair.

---

## Appendix C. Evaluation and Acceptance Criteria

This appendix specifies divergence metrics and acceptance thresholds, each mapped to a corresponding fingerprint JSON field. For a summary, note how each metric aligns with a specific JSON field.

### C.1 Metric Families

#### (1) Rate Constraints (scalar)

Given a target interval $[a,b]$ and observed value $v$:

$$\text{viol}_r(v) = \max(0,a-v) + \max(0,v-b)$$

Score:

$$s_r(v) = \exp(-\alpha_r \cdot \text{viol}_r(v))$$

Mapped JSON paths:
- `/targets/punctuation/comma_density_per_100w`
- `/targets/orthography/contractions_rate`

---

#### (2) Histogram Constraints (sentence / paragraph)

The primary metric is the L1 histogram distance:

$$d_h(\mathbf{h}^*, \mathbf{h}) = \\frac{1}{2} \sum_{b=1}^{B} |h_b - h_b^*|$$

Score:

$$s_h = 1 - \min(1, d_h)$$

Mapped JSON:
- `/targets/sentence/length_words/distribution`
- `/targets/paragraph/length_sentences`

---

#### (3) Lexicon Constraints

Hard constraints:

$$s_{lex}^{hard} = \mathbf{1}[\text{no forbidden term appears}]$$

Soft constraints:

$$s_{lex}^{soft} = \exp(-\alpha_{lex} \cdot |f_y - f^*|)$$

Mapped JSON:
- `/lexicon/avoid_words`
- `/lexicon/avoid_phrases`
- `/lexicon/preferred_phrases`

---

#### (4) Function‑Word and Stance Signals

For each rate signal (such as `hedge_rate`, `first_person_rate`), the relative deviation is defined as:

$$d_s(v, v^*) = \\frac{|v - v^*|}{\\max(|v^*|, 1)}$$

Score:

$$s_s = 1 - \min(1, d_s)$$

Mapped JSON:
- `/measurements/function_words`
- `/measurements/stance_signals`

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

Mapped JSON:
- `/validators/scoring/overall_threshold/pass`
- `/validators/scoring/overall_threshold/warn`

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

---

### C.4 Iterative Repair Stopping Rule

Let $S_t$ denote the score at iteration $t$. The process stops when:

$$S_t \ge S_{pass} \quad \text{and} \quad H_t = 0$$

Otherwise, continue for up to $T_{max}$ repair passes.

---

## Appendix D. Mechanism of Fingerprint-Conditioned Rewriting

This appendix details how an explicit stylometric fingerprint guides an LLM to rewrite text in the target author style, despite the LLM's internal representations being latent and opaque. The process is formalised as external style conditioning through instruction embedding, constraint activation, and iterative projection.

For non‑experts, the essential idea is straightforward: the fingerprint functions as a checklist, and the audit loop enforces adherence to that checklist.

---

## D.1 From Stylometric Profile to Control Signals

The style fingerprint $\mathcal{F}$ is not provided to the LLM as raw statistics, but as a compiled control representation comprising:

1. Numeric constraints (ranges, histograms, tolerances)
2. Discrete symbolic constraints (lexicon rules, rhetorical templates, structural policies)
3. Priority and strictness controls (ordering, hard versus soft constraints)
4. Derived natural-language instructions (compiled in `derived_instructions.*`)
5. Optional bounded humanizer variance (seeded micro-variations within constraints)

The compiled instruction set is denoted:

$$\mathcal{I}(\mathcal{F}) = \text{Compile}(\mathcal{F})$$

where $\mathcal{I}(\mathcal{F})$ is a structured textual representation injected into the LLM prompt.

This compilation step involves three main transformations:

### (i) Constraint verbalisation

Numeric constraints are rendered as qualitative instructions:

- "Use short-to-medium sentences (10-18 words typical)"
- "Favour one-sentence paragraphs occasionally (~15%)"
- "Avoid heavy semicolon usage; commas preferred"

This converts $\psi_j(y)\in\mathcal{C}_j$ into behavioural descriptors.

### (ii) Salience weighting

Constraint weights $w_j$ are reflected in:

- prompt ordering
- emphasis (phrasing, repetition)
- explicit language such as "must" or "prefer"

### (iii) Conflict resolution policy

The `controls.priority_order` field induces a partial order:

$$\text{meaning preservation} \succ \text{lexicon} \succ \text{sentence rhythm} \succ \text{punctuation} \succ \text{templates}$$

This ordering is verbalised to ensure that stylistic fidelity does not override semantic fidelity.

**Bounded stochastic variance.** When enabled, `controls.humanizer_variance` allows a small number of seeded micro‑operations (e.g., transition swaps, filler drops) per 1000 words. These edits are constrained, logged, and subordinate to the fingerprint, introducing human‑like irregularity without semantic drift.

---

## D.2 Conditioning as External Latent Space Steering

Let $h(x)$ denote the latent representation of the input text under the LLM, and $c(\mathcal{I})$ the latent encoding of the instruction set.

The model samples from:

$$p_\theta(y \mid x, \mathcal{I}) = p_\theta(y \mid h(x), c(\mathcal{I}))$$

Here, $c(\mathcal{I})$ induces a soft bias over stylistic manifolds in latent space.

Rather than learning a new style embedding, the fingerprint:

- activates latent regions associated with sentence rhythm
- biases token transitions linked to punctuation patterns
- suppresses lexical clusters disfavoured by the lexicon rules

This is analogous to feature activation steering in controllable generation, but with externalised and interpretable features.

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

### Sentence rhythm

Sentence length distributions arise from:

- early or delayed emission of terminal punctuation  
- preference for conjunction vs clause boundaries  
- tolerance for subordinate clauses  

The histogram constraint does not enforce exact lengths, but biases the *distribution of stopping times*.

### Paragraph structure

Paragraph rhythm emerges from:

- probability of emitting newline tokens  
- probability of single-sentence termination  
- continuation bias under discourse coherence  

### Lexical tone

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

### (i) Rich latent disentanglement

Although not perfectly disentangled, latent spaces exhibit partial separability between:

- syntactic rhythm  
- punctuation usage  
- discourse structure  
- lexical tone  

Stylometric constraints align with axes the model already encodes.

### (ii) Strong instruction adherence

Instruction-tuned models approximate constrained decoding by:

- maintaining long-range control variables in attention  
- preserving global objectives across paragraphs  
- rebalancing generation probabilities dynamically  

### (iii) Redundancy in stylistic signals

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

### D.8.2 Over-fitting to Constraints

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

   Then:
   $$\mathcal{F}(x,\mathcal{F}) \neq \varnothing$$

   #### Sketch of proof

- Let $\mathcal{P}(x)$ denote the set of meaning-preserving paraphrases of $x$, which is non-empty for any non-degenerate $x$.
- Feature maps $\psi_j$ are continuous, or piecewise continuous, under paraphrase operations.
- By assumption, tolerance intervals contain a neighbourhood around $\psi_j(x)$.
- It follows that there exists $y \in \mathcal{P}(x)$ such that $\psi_j(y)\in\mathcal{C}_j$ for all $j$. ∎

---

## E.4 Constraint Compatibility and Conflict Graphs

A constraint compatibility graph is defined as follows:

- Nodes represent constraints $j$.
- An edge connects $j$ and $k$ if $\mathcal{C}_j \cap \mathcal{C}_k = \varnothing$ under semantic invariants.

A necessary condition for feasibility is:

$$\text{Graph}(\mathcal{F}) \text{ is bipartite with respect to hard constraints}$$

In practice, some constraints are compatible, while others may conflict:

- Sentence rhythm and paragraph rhythm are compatible.
- Lexicon and semantic invariants may conflict.
- Template and rhythm may conflict in short texts.

The system imposes a partial order:

$$\text{meaning} \succ \text{lexicon} \succ \text{structure} \succ \text{punctuation} \succ \text{templates}$$

This ensures that, in the event of conflict, feasibility is prioritised.

---

## E.5 Minimal Tolerance Bounds

Let $\sigma_j$ denote the empirical standard deviation of feature $\psi_j$ across the author corpus.

Recommended sufficient tolerances are:

- For range constraints:
  $$[a_j, b_j] = [\mu_j - 2\sigma_j,\; \mu_j + 2\sigma_j]$$

- For histogram constraints:
  $$\tau_j \ge 2 \cdot \mathbb{E}[W_1(\mathbf{h},\mathbf{h}')]$$

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

1. **Extremely short texts**: There are insufficient degrees of freedom for histogram control.
2. **Highly constrained technical content**: Semantic invariants dominate stylistic degrees of freedom.
3. **Overly tight tolerances**: $\tau_j < \epsilon_j$.

In such cases, Stylometric-Transfer:

- reports deviation,
- relaxes the lowest-priority constraints,
- guarantees semantic correctness.

---

## E.8 Interpretation

The fingerprint does not specify a single point in style space, but rather a convex (or approximately convex) admissible region.

Rewriting succeeds when

$$\mathcal{E}(x) \cap \mathcal{M}_{style} \neq \varnothing$$

This perspective clarifies the necessity of tolerances, the ill-posed nature of strict imitation, and the rationale for deviation reporting.

---

## Appendix F. Comparison with Fine-Tuning, LoRA, and Latent Style Embedding Approaches

This appendix places Stylometric-Transfer in the context of existing approaches to author-style modelling and controlled generation, with emphasis on transparency and editorial control.

---

## F.1 Taxonomy of Style Modelling Approaches

Four main paradigms can be distinguished:

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
- Learned stylistic features cannot be inspected.
- No partial control; sentence rhythm and lexicon cannot be weighted separately.
- Catastrophic forgetting is a risk.
- Retraining is expensive for each author.

### Contrast

Stylometric-Transfer instead solves

$$\max_{y \in \mathcal{M}_{style}} p_\theta(y\mid x)$$

with

- no parameter updates,
- an explicit admissible region,
- post-hoc auditing.

---

## F.3 LoRA / Adapter-Based Style Conditioning

### Mechanism

Low-rank matrices $\Delta W$ are learned so that

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

Stylometric-Transfer

- exposes every control dimension,
- allows continuous interpolation via tolerances,
- supports manual editing and versioning.

---

## F.4 Latent Style Embedding Methods

### Mechanism

A vector

$$z_{style} \in \mathbb{R}^d$$

is learned, and generation is conditioned as

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

$$z_{style}
\quad \longrightarrow \quad
\mathcal{F} = \{(\psi_j,\mathcal{C}_j,w_j)\}$$

yielding

- explicit axes of variation,
- measurable compliance,
- verifiable reproduction.

---

## F.5 Control Granularity and Editorial Authority

A central distinction is who controls style.

| Property | Fine-tune | LoRA | Embedding | Stylometric-Transfer |
|----------|-----------|------|-----------|----------------------|
| Human-readable model | ✗ | ✗ | ✗ | **✓** |
| Partial constraint weighting | ✗ | ✗ | ✗ | **✓** |
| Manual editing | ✗ | ✗ | ✗ | **✓** |
| Version control | ✗ | ✗ | ✗ | **✓** |
| Deviation reporting | ✗ | ✗ | ✗ | **✓** |

Stylometric-Transfer treats style as an editorial object, rather than a byproduct of training.

---

## F.6 Data Efficiency

Fine-tuning and embedding methods require

$$N \gg 10^4 \text{ tokens}$$

to stabilise latent style representations.

Stylometric-Transfer requires only enough data to estimate low-variance statistics, often $N \approx 10^3-10^4$ tokens, and remains robust on heterogeneous corpora.

---

## F.7 Transferability and Compositionality

Latent methods encounter difficulties when

- combining multiple authors,
- interpolating interpretable features,
- transferring style across domains.

Stylometric-Transfer supports

- convex combinations of fingerprints,
- selective inheritance of features,
- domain-specific constraint relaxation.

Formally, fingerprints compose as

$$\mathcal{F}_\lambda = \lambda \mathcal{F}_1 + (1-\lambda)\mathcal{F}_2$$

at the level of histogram mixtures, range interpolation, and lexicon unions.

---

## F.8 Interpretability and Scientific Value

From a scientific perspective:

- Fine-tuning learns unknown features.
- Embeddings encode unlabeled dimensions.
- Stylometric-Transfer recovers measurable linguistic variables.

This enables hypothesis testing, ablation studies, stylistic causality analysis, and reproducible experiments.

---

## F.9 Summary

Stylometric-Transfer differs fundamentally from existing approaches by

1. externalising style as explicit constraints,
2. avoiding training and latent embeddings,
3. enabling auditability and editorial control,
4. supporting theoretical analysis of feasibility and convergence.

Rather than learning what style is, it defines where style may reside in feature space.

---

## F.10 Comparison to "best-of-class" practical pipelines

In practice, teams rarely deploy a single pure paradigm. "Best-of-class" systems typically combine several ingredients:

- a powerful instruction-tuned LLM,
- guardrails for structure and meaning,
- some form of evaluation (automatic + human),
- and, when budgets allow, training-time specialization (fine-tuning or adapters).

The table below compares Stylometric-Transfer to common high-performing alternatives as they are typically used in the field. The goal is not to claim superiority, but to clarify tradeoffs.

| Approach (typical) | Strengths | Weaknesses | When it tends to win |
| --- | --- | --- | --- |
| Prompt-only style steering (LLM + "write like X") | Lowest engineering cost; fast iteration; can be surprisingly good for broad register shifts | Non-auditable; fragile to prompt drift; hard to reproduce; style can collapse into generic LLM voice; meaning drift is common without extra checks | Casual rewrites; low-stakes editing; early prototyping |
| Agentic / iterative editing (LLM + critique loops) | Better compliance via self-critique; can catch formatting/consistency errors; flexible to new constraints | More latency/cost; can over-edit and introduce drift; "improvement" can be subjective and unstable | Long documents where structure matters; when a human reviewer is in the loop |
| Fine-tuned author or domain model | Highest imitation fidelity in-domain; less prompt overhead; can be fast at inference once trained | Opaque; expensive to train and update; difficult to verify what changed; risky for impersonation; hard to parameterize partial style knobs | Narrow, stable domain with large data; high-volume generation; strict internal style guides |
| LoRA/adapters for style or domain | Cheaper than full fine-tune; modular; can switch adapters | Still opaque; style dimensions entangled; multiple adapters can conflict; auditability remains limited | Medium-scale specialization; internal domain conditioning |
| Latent style embeddings + conditional generation | Compact conditioning; can interpolate styles; integrates with learned pipelines | Hard to interpret; evaluation difficult; risk of content leakage into style vector | Research settings; controlled datasets; style mixing experiments |
| Detector-guided "humanization" optimization | Can target a specific detector or proxy; easy to define an objective | Highly vulnerable to Goodhart's law; may produce adversarial artifacts; can harm meaning and readability | When the objective is explicitly "reduce detector score" (not recommended for general writing) |
| **Stylometric-Transfer (this work)** | Explicit, versionable constraints; local grounding; stable reproducibility; clear deviation reporting; integrates humanization as auditable mechanisms | Limited ceiling on subtle stylistic phenomena; depends on LLM compliance; chunking introduces boundary effects; heuristics can mis-segment | When auditability matters; when you need tunable controls; when you want per-author JSON artifacts rather than opaque weights |

### F.10.1 Why explicit constraints can be competitive

Two practical observations explain why explicit constraints can compete with more "powerful" learned approaches in real deployments:

1) **Human-in-the-loop editing benefits from inspectability.** When a rewrite is not quite right, a JSON fingerprint and a deviations report make it clear *which knob to turn*. In contrast, a fine-tuned model offers few actionable levers beyond "retrain" or "prompt harder".

2) **Many style differences are low-dimensional and stable.** In many domains, the biggest wins come from a handful of stable signals (sentence rhythm, punctuation, discourse markers, function-word balance, and a small set of lexical preferences). Making those explicit yields predictable control even if it does not capture every subtlety.

### F.10.2 Where best-of-class learned approaches remain ahead

Learned approaches retain an advantage when style is expressed in high-dimensional ways that are hard to capture with lightweight metrics: nuanced syntactic alternations, idiomatic collocations, long-range narrative structure, and pragmatic implicature. In such regimes, explicit constraints may approximate the "outer shell" of the voice (cadence and surface markers) without fully reproducing the deeper texture.

---

## Appendix G. JSON schema

``` json
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
}

```

---

## Appendix H. Tunables schema (config.tunables.json)

`config.tunables.json` provides deterministic control over humanizer conflict thresholds and basic sanity checks, such as line-count change warnings, during style application. When a fingerprint is generated, the current tunables are optionally embedded under `metadata.extraction.tunables_snapshot` to preserve the exact settings used for provenance. The following schema outlines the supported keys and types:

```json
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
        "em_dash_keep_rate": {
          "type": "number",
          "minimum": 0
        },
        "hedge_keep_rate": {
          "type": "number",
          "minimum": 0
        },
        "first_person_keep_rate": {
          "type": "number",
          "minimum": 0
        },
        "contractions_avoid_threshold": {
          "type": "number",
          "minimum": 0
        },
        "contractions_use_threshold": {
          "type": "number",
          "minimum": 0
        },
        "heading_title_case_keep_rate": {
          "type": "number",
          "minimum": 0
        },
        "boldface_keep_per_1000w": {
          "type": "number",
          "minimum": 0
        },
        "inline_header_list_keep_rate": {
          "type": "number",
          "minimum": 0
        }
      },
      "additionalProperties": false
    },
    "humanizer_mandatory": {
      "type": "object",
      "properties": {
        "avoid_em_dashes": {
          "type": "boolean"
        },
        "emoji_policy": {
          "type": "string",
          "enum": [
            "remove",
            "replace",
            "none"
          ]
        }
      },
      "additionalProperties": false
    },
    "humanizer_variance": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean"
        },
        "seed": {
          "type": "integer"
        },
        "max_ops_per_1000w": {
          "type": "number",
          "minimum": 0
        },
        "allowed_ops": {
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      },
      "additionalProperties": false
    },
    "humanization_metrics": {
      "type": "object",
      "properties": {
        "weights": {
          "type": "object",
          "additionalProperties": {
            "type": "number",
            "minimum": 0
          }
        }
      },
      "additionalProperties": false
    },
    "humanization_baseline": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean"
        },
        "window_words": {
          "type": "integer",
          "minimum": 50
        },
        "stride_words": {
          "type": "integer",
          "minimum": 25
        },
        "min_window_words": {
          "type": "integer",
          "minimum": 50
        },
        "max_windows": {
          "type": "integer",
          "minimum": 1
        }
      },
      "additionalProperties": false
    },
    "humanization_controller": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean"
        },
        "seed": {
          "type": "integer"
        },
        "quantiles": {
          "type": "array",
          "items": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          }
        },
        "range_pct": {
          "type": "number",
          "minimum": 0
        },
        "min_width": {
          "type": "number",
          "minimum": 0
        },
        "max_width": {
          "type": "number",
          "minimum": 0
        },
        "allowed_metrics": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "feedback_enabled": {
          "type": "boolean"
        },
        "feedback_tolerance": {
          "type": "number",
          "minimum": 0
        },
        "max_feedback_retries": {
          "type": "integer",
          "minimum": 0
        }
      },
      "additionalProperties": false
    },
    "lexical_signals": {
      "type": "object",
      "properties": {
        "rare_words_limit": {
          "type": "integer",
          "minimum": 1
        }
      },
      "additionalProperties": false
    },
    "lexical_avoidance": {
      "type": "object",
      "properties": {
        "rare_words_limit": {
          "type": "integer",
          "minimum": 1
        }
      },
      "additionalProperties": false
    },
    "controls_normalization": {
      "type": "object",
      "properties": {
        "rewrite_policy": {
          "type": "object",
          "properties": {
            "jaccard_threshold": {
              "type": "number",
              "minimum": 0,
              "maximum": 1
            },
            "dedupe_on_subset": {
              "type": "boolean"
            },
            "prefer_more_specific": {
              "type": "boolean"
            },
            "compress_directives": {
              "type": "boolean"
            },
            "directive_verbs": {
              "type": "array",
              "items": {
                "type": "string"
              }
            },
            "stopwords": {
              "type": "array",
              "items": {
                "type": "string"
              }
            }
          },
          "additionalProperties": false
        },
        "priority_order": {
          "type": "object",
          "properties": {
            "token_pattern": {
              "type": "string"
            },
            "dedupe_case_insensitive": {
              "type": "boolean"
            },
            "exclude_tokens": {
              "type": "array",
              "items": {
                "type": "string"
              }
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
        "quote_span_min": {
          "type": "integer",
          "minimum": 0
        },
        "quoted_ratio_min": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "quote_para_ratio_min": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "quoted_ratio_force": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      },
      "additionalProperties": false
    },
    "chunking": {
      "type": "object",
      "properties": {
        "max_input_tokens": {
          "type": "integer",
          "minimum": 200
        },
        "chunk_split_on": {
          "type": "string",
          "enum": [
            "word",
            "sentence",
            "paragraph"
          ]
        },
        "min_chunks_when_perturbing": {
          "type": "integer",
          "minimum": 1
        },
        "recovery_split_max_depth": {
          "type": "integer",
          "minimum": 0
        },
        "recovery_split_min_chars": {
          "type": "integer",
          "minimum": 0
        },
        "variance_aware": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean"
            },
            "sentence_stdev_ref": {
              "type": "number",
              "minimum": 0
            },
            "paragraph_burst_ref": {
              "type": "number",
              "minimum": 0
            },
            "min_factor": {
              "type": "number",
              "minimum": 0
            },
            "max_factor": {
              "type": "number",
              "minimum": 0
            }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },
    "style_retry": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean"
        },
        "threshold": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "max_retries": {
          "type": "integer",
          "minimum": 0
        }
      },
      "additionalProperties": false
    },
    "section_restore": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean"
        },
        "max_restore_sections": {
          "type": "integer",
          "minimum": 0
        },
        "heading_similarity_threshold": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "signature_similarity_threshold": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "signature_min_overlap": {
          "type": "integer",
          "minimum": 0
        }
      },
      "additionalProperties": false
    },
    "sanity_checks": {
      "type": "object",
      "properties": {
        "line_count_warn_pct": {
          "type": "number",
          "minimum": 0
        },
        "word_count_warn_pct": {
          "type": "number",
          "minimum": 0
        },
        "paragraph_count_warn_pct": {
          "type": "number",
          "minimum": 0
        }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

**Attribution:** Humanization guidelines are adapted from the humanizer skill in softaworks/agent-toolkit by @leonardocouy.

### H.1 Tunable definitions (interpretation)

- `em_dash_keep_rate`: if the fingerprint’s em dash rate (per 1000 words) meets or exceeds this value, the rule to avoid em dashes is set aside as conflicting.
- `hedge_keep_rate`: if the fingerprint’s hedging rate (per 1000 words) meets or exceeds this value, rules discouraging hedging are set aside.
- `first_person_keep_rate`: if the fingerprint’s first person rate (per 1000 words) is below this value (or pronoun preferences avoid first person), rules requiring first person are set aside.
- `contractions_avoid_threshold`: if the fingerprint’s contraction rate (per 1000 words) meets or exceeds this value, the rule to avoid contractions is set aside.
- `contractions_use_threshold`: if the fingerprint’s contraction rate (per 1000 words) is below this value, the rule to use contractions is set aside.
- `heading_title_case_keep_rate`: if the input Markdown’s heading Title Case ratio meets or exceeds this value, the rule to avoid Title Case headings is set aside.
- `boldface_keep_per_1000w`: if boldface density (per 1000 words) meets or exceeds this value, the rule to avoid boldface is set aside.
- `inline_header_list_keep_rate`: if the ratio of inline-header list items (such as `- **Label:**`) meets or exceeds this value, the rule to avoid inline-header lists is set aside.
- `avoid_em_dashes`: when true, em dashes are always removed in the final output (mandatory humanizer control).
- `emoji_policy`: `remove`, `replace`, or `none`. `replace` swaps emojis with conventional monochrome symbols when possible, otherwise removes them.
- `humanizer_variance.enabled`: enables bounded stochastic micro‑variation during application.
- `humanizer_variance.seed`: RNG seed for deterministic runs.
- `humanizer_variance.max_ops_per_1000w`: maximum number of micro‑operations per 1000 words. **Recommendation:** start at `0.5`; `0.5–1.5` is usually safe. Values above `2.0` can begin to feel noisy unless the input is highly repetitive.
- `humanizer_variance.allowed_ops`: allowed micro‑operations (e.g., `swap_transition`, `drop_filler`). **Recommendation:** begin with `["swap_transition", "drop_filler"]`, add ops gradually, and keep the list short to avoid compounding randomness.
- `humanization_metrics.weights`: optional weighting for the 0–100 aggregate humanization score. Any metric with a weight of 0 is excluded.
- `lexical_signals.rare_words_limit`: maximum number of rare words included in `measurements.lexical_signals.rare_words`.
- `lexical_avoidance.rare_words_limit`: maximum number of rare words included in `measurements.lexical_avoidance.rare_words`.
- `chunking.max_input_tokens`: hard cap on input tokens per chunk (after prompt overhead). Lower values increase chunk count but reduce per‑request latency and timeouts.
- `chunking.chunk_split_on`: primary chunking unit (`word`, `sentence`, or `paragraph`). If a paragraph exceeds the budget, it falls back to sentence splitting for that chunk; if a sentence is still oversized, it falls back to word splitting for that chunk. Bullet/numbered list lines are treated as sentence units.
- `style_retry.enabled`: enable/disable the delta‑feedback retry pass after measuring style compliance.
- `style_retry.threshold`: retry when compliance score is below this threshold (default `0.75`). Lower values trigger fewer retries (more permissive); higher values trigger more retries (stricter). `0.0` effectively disables threshold‑based retries, while `1.0` retries unless the output is nearly perfect.
- `style_retry.max_retries`: maximum number of retry passes (default `1`).
- `section_restore.enabled`: enable/disable restoring missing sections after rewrite.
- `section_restore.max_restore_sections`: maximum number of missing sections to restore (0 disables restoration).
- `section_restore.heading_similarity_threshold`: fuzzy heading match threshold for considering a rewritten heading “present”.
- `section_restore.signature_similarity_threshold`: content‑signature similarity threshold for matching a section by its opening content.
- `section_restore.signature_min_overlap`: minimum number of overlapping signature tokens required for a content match.
- `line_count_warn_pct`: if the output line count changes by this percentage or more, a warning is issued for possible missing or expanded content.
- `word_count_warn_pct`: if the output word count changes by this percentage or more, a warning is issued for possible missing or expanded content.
- `paragraph_count_warn_pct`: if the output paragraph count changes by this percentage or more, a warning is issued for possible missing or expanded content.

---

## Appendix I. Stylometry + Humanization FAQ and Glossary (a practical one-stop reference)

This appendix is intentionally verbose. It is written for readers who want the “missing middle” between textbook definitions and code.

### I.1 Glossary (terms used throughout the paper and the code)

**Author‑voice text**  
The portion of a document assumed to represent the author’s own prose rather than quoted material, references, footnotes, or boilerplate. The project filters non‑author‑voice regions before measuring style and before asking the LLM to rewrite.

**Token / type**  
A *token* is one occurrence of a word; a *type* is a unique word form. If the text is “cats chase cats”, then $N=3$ tokens and $V=2$ types.

**Rate per 1000 words**  
A normalised count: $r = 1000 \cdot C / W$. This rescales event counts (commas, semicolons, hedges) so that texts of different lengths become comparable.

**Histogram (probability distribution)**  
A vector of bin probabilities $\mathbf{h}$ such that $\sum_b h_b = 1$. Histograms are used for sentence‑length distributions and paragraph‑length distributions because “shape” matters (variance and tails) even when means match.

**Entropy**  
A measure of spread. If probabilities are $p_i$, then $H=-\sum_i p_i\log p_i$. High entropy means “many categories are used fairly evenly”; low entropy means “mass is concentrated in a few categories”. In this project, entropy is used as a simple, interpretable proxy for variety (punctuation variety; function‑word variety; character‑trigram texture).

**KL divergence**  
If $P$ is an observed distribution and $Q$ is a reference distribution, then:
$$D_{\\mathrm{KL}}(P\\|Q)=\\sum_i P_i\\log\\frac{P_i}{Q_i}.$$
It is asymmetric. In operational terms: it penalises “surprising under $Q$” mass in $P$. The project uses KL‑derived measures to quantify how far a rewrite drifts from the fingerprint in distributional features.

**Jensen–Shannon divergence (JSD)**  
A symmetric, smoothed divergence derived from KL:
$$\\mathrm{JSD}(P,Q)=\\tfrac{1}{2}D_{\\mathrm{KL}}(P\\|M)+\\tfrac{1}{2}D_{\\mathrm{KL}}(Q\\|M), \\quad M=\\tfrac{1}{2}(P+Q).$$
JSD is bounded and tends to be easier to interpret as a “distance‑like” score for histograms.

**Burstiness (coefficient of variation)**  
For a length sequence $\\ell$, burstiness is $B = \\sigma(\\ell)/\\mu(\\ell)$. It measures variability relative to the mean. LLM outputs often have artificially low burstiness; many human authors do not.

**Self‑echo**  
Repeated n‑grams within a text (often beyond what the topic justifies). This paper uses repeated bigram/trigram rates as a lightweight proxy.

**Humanization metric**  
A quantitative proxy intended to detect and reduce AI‑typical artefacts. In this system, metrics are for *engineering feedback*, not forensic detection.

**Humanization baseline**  
Intra‑corpus variability extracted from the author’s corpus and used to produce per‑chunk “controller overlays” (small target nudges that encourage natural dispersion).

**Controller overlay**  
A per‑chunk adjustment to target ranges (e.g., nudging sentence‑length mean toward a sampled quantile of the corpus distribution) intended to introduce author‑like variability without random drift. It is deterministic given a seed and tunables.

**Bounded stochastic variance**  
Seeded micro‑edits (small operations) applied under strict caps (operations per 1000 words, allowed operation set). The word “bounded” is the key: it is designed to be auditable and prevent runaway randomness.

**Chunking**  
Splitting input Markdown into smaller parts so each LLM call stays within a token budget. Chunking is not just a performance hack: it affects distributional control. Smaller chunks reduce per‑call timeouts and allow more local variability, but can increase risk of global inconsistencies if constraints are not managed carefully.

**Deviation report**  
A structured record of constraint conflicts, style drifts, and any deterministic “fixups” applied after the LLM output. Deviations are the audit trail.

### I.2 FAQ (questions a careful reader is likely to ask)

**Q: Why does stylometry often focus on function words?**  
Because function words are frequent, relatively topic‑invariant, and difficult to consciously control. If two texts match on many topic words but diverge strongly on function‑word balance and connective habits, the “style” is plausibly different even if the “topic” is the same.

**Q: Why use per‑1000‑word rates rather than raw counts?**  
Raw counts scale with length and can be misleading. Normalisation makes “style density” comparable. It also makes a rewrite audit meaningful: if a 2,000‑word text has 120 commas, the comma density is 60 per 1000 words. If the rewrite has 30 commas, density is 15 per 1000 words, signalling compression or a shift to simpler sentence structure.

**Q: Why store histograms instead of just mean and standard deviation?**  
Because the distribution’s tails often carry stylistic meaning. Many authors occasionally produce very short or very long sentences. LLMs tend to “regress to the middle”. A histogram preserves where the mass sits across bins, which is a more direct representation of rhythm than a single mean.

**Q: If the corpus contains quotations, why filter them at all?**  
Because quotations can represent other voices and content. For non‑fiction, long quoted passages are typically the speech of sources, not the author’s own style. For fiction, quoted dialogue is part of the author’s craft and should usually be included. The project therefore detects fiction vs non‑fiction and changes the quote‑handling policy accordingly, with an explicit message and manual overrides.

**Q: Why not use a full syntactic parser or a transformer embedding for deeper style?**  
You can, but you pay for it: dependencies, latency, fragility, and reduced auditability. The philosophy here is “measure what you can defend”. A smaller set of stable, interpretable features often provides more engineering leverage than a high‑dimensional embedding whose drift is hard to diagnose.

**Q: Is the humanization score a detector?**  
No. It is a dashboard metric. It is meant to tell you whether the rewrite became more uniform, more repetitive, or more distributionally distant from the fingerprint. You should interpret it like a unit test: a failing unit test does not prove your program is wrong in every way, but it is a useful signal that something is off.

**Q: Why do you sometimes want more chunks than “token limits require”?**  
Because variability is easier to express locally. If you want a high sentence‑length standard deviation, rewriting an entire document as one chunk can push the model toward a smooth “compromise” style. Smaller chunks allow the controller overlay to sample different quantiles across chunks (deterministically), producing natural dispersion while still keeping within global constraints.

**Q: Doesn’t chunking risk losing coherence or dropping content?**  
Yes, and the project treats this as a first‑class failure mode. It provides: (i) deterministic preservation of protected regions (blockquotes, references, footnotes, citations), (ii) line/word/paragraph change warnings, and (iii) optional section restoration when headings go missing. Chunking is therefore paired with post‑rewrite checks rather than treated as a blind splitting strategy.

**Q: Why keep an explicit JSON fingerprint rather than fine‑tune a model?**  
Because an explicit fingerprint is inspectable, editable, and versionable. It gives you editorial authority: you can see what the system believes about the author’s style and change it. Fine‑tuning and latent embeddings can be powerful, but they are harder to audit and to explain, and they often entangle content, style, and safety behaviours.

**Q: What are the easiest ways to misuse stylometry?**  
The classic failure is topic leakage: measuring “style” signals that are really just “the author often writes about X”. Another failure is over‑confidence: treating a stylometric score as proof rather than as a probabilistic, context‑dependent signal. This paper tries to mitigate both by (i) filtering proper‑name phrases from lexical signals and (ii) emphasising ranges and distributions, not single thresholds.

**Q: What should I do when the rewrite shrinks or expands a lot?**  
Treat it as a meaning‑preservation risk. Large word‑count drops can indicate summarisation; large expansions can indicate hallucinated detail. Use the deviations report and the count‑change warnings as triggers for manual review or for tightening constraints (e.g., chunk sizing, retry thresholds, or stricter “preserve structure” instructions).

---

## Licence Notice

This work is licensed under the PolyForm Noncommercial Licence 1.0.0.  
Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).  
See `LICENSE.md` for the full licence text and terms.
