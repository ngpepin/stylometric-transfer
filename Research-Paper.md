 
**Repository:** `stylometric-transfer`  
**Keywords:** stylometry, computational stylistics, authorship attribution, controllable text generation, text style transfer, interpretability

(c) 2026 Nicolas Pepin

---

## Abstract

Stylometric-Transfer provides a practical method for (i) stylometric profiling of an author's writing corpus into an explicit, interpretable JSON artefact (a style fingerprint) and (ii) meaning-preserving style transfer that rewrites new text to conform to the fingerprint using a large language model (LLM). The approach combines classic stylometric measurement - such as punctuation rates and sentence-length distributions - with LLM-mediated synthesis into human-editable constraints, including ranges, histograms, lexicon rules, and rhetorical templates. The fingerprint is formalized as a constraint set, and a constraint-satisfaction decoding view is provided for LLM rewriting, together with compliance scoring based on distributional divergences. Notably, the framework unifies stylometric transfer and humanization by quantifying a conflict‑resolution layer that filters humanization guidelines against fingerprint constraints, and by supporting bounded stochastic variance under explicit controls. This hybrid design allows for an auditable alternative to latent style embeddings, while remaining consistent with established stylometry and text style transfer research.

**Reader’s guide:** Sections 1–3 outline the system’s purpose and rationale. Section 4 explains the measurements with simple examples. Sections 5–6 describe how those measurements become constraints for rewriting. Section 7 connects the ideas to the code. The appendices provide further mathematical and algorithmic detail for expert readers.

---

## 1. Introduction

Stylometry examines quantitative signals of writing style for tasks such as authorship attribution and author profiling. A well-known example is the Federalist Papers analysis, where frequent-word statistics support Bayesian inference over disputed authorship ([press.uchicago.edu](https://press.uchicago.edu/ucp/books/book/distributed/I/bo5667096.html?utm_source=chatgpt.com)).

Text style transfer (TST), by contrast, seeks to transform text so that stylistic properties match a target style, while preserving content. A persistent challenge is separating content from style in the absence of parallel data, which has led to cross-alignment approaches and ongoing debate about evaluation and ethics ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com)).

In practical terms, stylometry treats an author’s style as a set of measurable habits: sentence length, punctuation frequency, recurring transitions, and words that are rarely used. Style transfer attempts to generate new text as if those habits were followed, without altering meaning. The difficulty lies in ensuring that a model does not introduce stylistic artefacts or factual drift. This paper addresses how to make that tension measurable and manageable.

A hybrid approach is proposed: style is represented explicitly as a stylometric style fingerprint (in JSON), and an LLM acts as a constrained rewriter, guided by both the fingerprint and locally measured statistics from the author corpus and candidate text. Humanization guidelines are integrated by means of a conflict‑resolution layer, which deterministically filters guideline rules when they contradict fingerprint signals or the input’s stylistic structure. Where enabled, bounded stochastic variance applies a small, seeded number of micro‑edits to reduce AI‑typical uniformity while preserving meaning and constraints.

---

## 2. Related Work

This section provides a brief overview for non-specialists and references for specialists. The central point is that measurements are kept interpretable, and the LLM is tasked with following constraints rather than relying on hidden style embeddings.

### 2.1 Stylometry and Distance-Based Measures

Stylometric authorship attribution typically relies on robust, interpretable features such as word frequency profiles and distance measures. Burrows's Delta and its variants remain widely used; recent work explains the decomposition of feature selection, scaling (for example, z-transformation), and distance metrics, clarifying the effectiveness of Delta-style measures ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com)).

For those new to the field, these methods are intentionally simple, relying on counts and distributions rather than opaque neural features. This makes them suitable for interpretable fingerprints.

### 2.2 Text Style Transfer and Evaluation

Non-parallel TST methods, such as cross-alignment, demonstrate that certain stylistic attributes can be changed without parallel sentence pairs ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com)). Recent surveys discuss broad applications alongside challenges in evaluation and ethical risk, including concerns about misuse for impersonation, and support explicit safeguards and transparency in TST pipelines ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com)).

The literature indicates that style is difficult to define precisely. The approach here is to make style explicit and auditable, rather than learned and hidden.

### 2.3 Humanization-Aware Stylometric Transfer

Most style-transfer pipelines treat humanization as a separate editing step. Stylometric-Transfer incorporates humanization directly into constraint-guided rewriting by formalizing a conflict-resolution layer: humanization guidelines are applied only when they do not violate fingerprint-derived constraints or the input's structural features (such as heading case or inline-header lists). The guideline list is parsed into structured rules by an LLM (with deterministic fallback), then filtered against fingerprint signals before any rewrite prompt is constructed. This produces a single, auditable framework that balances stylistic fidelity with the removal of AI artefacts, rather than relying on post-hoc edits that may diverge from the author’s voice.

### 2.4 Humanization Mechanisms and Benefits

Humanization in this system is not a vague “make it sound human” instruction. It is a set of explicit, inspectable mechanisms designed to target known LLM artefacts while preserving the author’s voice. These mechanisms include:

- **Conflict‑filtered guidelines**: generic humanization rules are only applied when they do not contradict the fingerprint’s statistical baselines (for example, avoiding “don’t use em‑dashes” when the author’s corpus uses them frequently).
- **Mandatory hygiene rules**: optional hard constraints (such as removing em‑dashes or replacing emojis) are enforced deterministically and recorded in the deviations log.
- **Structural preservation**: blockquotes, citations, footnotes, and code spans are shielded from stylistic edits to prevent false “humanization” changes in non‑authorial content.
- **Bounded stochastic variance**: when enabled, a small, seeded number of micro‑edits (e.g., swapping transition words or dropping filler terms) introduce controlled irregularity without semantic drift.

The practical benefit is a *measurable reduction in AI‑typical uniformity* while keeping the transformation aligned to the author’s measurable habits. Because the humanization layer is explicit, deterministic by default, and logged, it can be audited and tuned without introducing opaque behaviour. In short, humanization is treated as a constrained post‑processing step integrated into the same interpretability framework as stylometric profiling itself.

---

## 3. Problem Setup

Let an author corpus be a set of documents

$$\mathcal{D} = \{d_1,\dots,d_N\}, \quad d_i \in \Sigma^*$$

where $\Sigma$ is a character alphabet.

Define:

- An interpretable feature extractor $\phi: \Sigma^* \to \mathbb{R}^K$ that produces measurable statistics (rates, histograms, counts).
- A style fingerprint $\mathcal{F}$ that stores target statistics, distributions, and discrete constraints (lexicon rules, templates).
- A rewriter $\mathcal{R}_\theta$ (LLM with parameters $\theta$) mapping input text $x$ to output $y$:

$$y = \mathcal{R}_\theta(x \mid \mathcal{F}).$$

The primary constraint is meaning preservation: no new facts, claims, or examples; entities and numerals are preserved unless explicitly permitted.

In essence, the author’s writing habits are measured and compressed into a JSON fingerprint. The LLM is then asked to rewrite new text so that these habits are respected, while the underlying meaning remains unchanged.

---

## 4. Stylometric Measurements

This section describes the simple, interpretable statistics measured by the system. The aim is not linguistic perfection, but stability, explainability, and auditability. These measurements constitute the ground truth the LLM must follow.

### 4.1 Rate and Density Features

Let $W(d)$ be an approximate word-token count and $C_e(d)$ the count of an event $e$ (such as commas). Define per-1000-word rates:

$$r_e(d) = 1000 \cdot \frac{C_e(d)}{\max(1, W(d))}.$$

The fingerprint stores targets as tolerance intervals:

$$r_e \in [\underline{r}_e, \overline{r}_e],$$

reflecting intra-author variability across topics and subgenres.

### 4.2 Histogram Features

For sentence lengths $\ell_1,\dots,\ell_m$ (in words), define a binned histogram

$$\mathbf{h} \in \Delta^{B-1}, \quad h_b = \frac{1}{m}\sum_{i=1}^m \mathbf{1}[\ell_i \in \text{bin}(b)],$$

where $\Delta^{B-1}$ is the probability simplex and bins are ordinal intervals (for example, $<10$, 10–17, 18–25, ...).

Paragraph rhythm is also captured with a one-sentence paragraph rate:

$$\rho_{1}(d) = \frac{\#\{\text{paragraphs with exactly one sentence}\}}{\max(1,\ \#\{\text{paragraphs}\})}.$$

This rate serves as a stylistic baseline. Excessive one-sentence paragraphs are flagged as an AI artefact only if they exceed the author’s $\rho_1$ range.

### 4.3 Rare-Word Signals

Let $f(w)$ be the corpus frequency of a token $w$ after filtering stopwords, numerals, and short tokens. A rare-word list is recorded:

$$\mathcal{R} = \{w : f(w) \le c_{\max}\},$$

where $c_{\max}$ is a small threshold (for example, 2–5 occurrences). These terms can be surfaced as avoid-lexicon hints so the rewriter does not overuse words the author rarely employs.

### 4.4 Rhetorical and Epistemic Signals

Beyond surface statistics, the system tracks interpretable rhetorical moves and certainty bands. Let $\mathcal{S}$ denote sentences in the corpus. For a rhetorical marker set $\mathcal{M}_k$ (e.g., claim or concession indicators), define:

$$r_k = 1000 \cdot \frac{\#\{s \in \mathcal{S} : s \text{ contains any marker in } \mathcal{M}_k\}}{\max(1, W)}.$$

We compute rates for claim, evidence, counterpoint, concession, and synthesis markers. Epistemic stance bands (speculative, probabilistic, assertive, directive) are computed using simple token lists. These signals are intentionally approximate; they are used to set tolerances, not to classify sentences perfectly.

### 4.5 Paragraph Cadence and Discourse Marker Position

Let $s_1$ and $s_n$ be the opening and closing sentences of a paragraph. The system records distributions of opening/closing sentence lengths (means and standard deviations) to capture cadence. It also tracks the position of discourse markers (e.g., “however”, “therefore”) as start‑of‑sentence versus mid‑sentence rates:

$$r_{\text{start}} = 1000 \cdot \frac{\#\{\text{markers at sentence start}\}}{\max(1, W)}, \quad r_{\text{mid}} = 1000 \cdot \frac{\#\{\text{markers mid‑sentence}\}}{\max(1, W)}.$$

These features help preserve where transitions tend to appear in the author’s voice.

### 4.6 Repetition Signals (Self‑Echo)

AI‑generated text often repeats phrases locally. To detect this, we measure repetition rates for bigrams and trigrams:

$$\rho_n = \frac{\sum_{g \in \mathcal{G}_n} \mathbf{1}[c(g) \ge c_{\min}] \cdot c(g)}{\max(1, |\mathcal{G}_n|)},$$

where $\mathcal{G}_n$ is the multiset of n‑grams and $c_{\min}$ is a small repeat threshold (default 3). These rates define ceilings for acceptable self‑echo.

### 4.4 Delta-Style Diagnostics (Optional)

Although Stylometric-Transfer is not an authorship attribution system, Delta-style distances can serve as diagnostic measures of stylistic proximity. Following standardization and Manhattan-style aggregation:

$$\Delta(d,d') = \frac{1}{K}\sum_{k=1}^K \left|z_k(d) - z_k(d')\right|,$$

where $z_k$ is the z-transformed version of feature $k$. Detailed explanations of Delta variants motivate this approach ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com)).

---

## 5. The Style Fingerprint as a Constraint Model

The fingerprint is treated as a set of weighted constraints:

$$\mathcal{F} = \{(\psi_j, \mathcal{C}_j, w_j)\}_{j=1}^J,$$

where:
- $\psi_j(y)$ is a measurable statistic of output text (for example, comma rate or histogram vector).
- $\mathcal{C}_j$ is an admissible set (range, divergence tolerance, forbidden list).
- $w_j$ is a weight (priority).

Typical constraint types include:

1. Range constraints: $\psi_j(y) \in [a,b]$  
2. Histogram constraints: $D(\mathbf{h}^*, \mathbf{h}(y)) \le \tau$  
3. Lexicon constraints: forbidden phrases or words; preferred synonyms; avoid-rare words $\mathcal{R}$  
4. Template constraints: rhetorical move frequency bounds

The JSON format introduces practical control fields, such as `priority_order` and `strictness`, to specify constraint precedence.

**Plain-language explanation:** The fingerprint functions as a weighted checklist. Certain items are strict (for example, “never use em-dashes”); others are soft (such as “prefer shorter sentences”). The system tracks how closely the output adheres to each requirement.

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

A typical decomposition is:

$$\mathcal{L}_{style}(y;\mathcal{F}) = \sum_{j=1}^J w_j\, \ell_j(\psi_j(y), \mathcal{C}_j).$$

Examples of penalties include:

**Range penalty** for $\mathcal{C}_j=[a,b]$:

$$\ell_j(v,[a,b]) = \big(\max(0,a-v)\big)^2 + \big(\max(0,v-b)\big)^2.$$

**Histogram penalty** using KL divergence:

$$\ell_j(\mathbf{h},\mathbf{h}^{*}) = D_{KL}(\mathbf{h}^{*}\|\mathbf{h}) = \sum_{b=1}^B h_b^{*} \log \frac{h_b^{*}}{\max(\epsilon,h_b)}.$$

For ordinal bins, the Wasserstein distance $W_1$ may be preferable; the implementation can support either approach.

### 6.2 Hard Constraints (Feasibility)

Some constraints are best enforced as hard feasibility requirements:

- Entity and number preservation constraints must be satisfied unless explicitly overridden.
- Hard forbidden lexicon constraints (for instance, terms that must not appear).

The feasible set is defined as:

$$\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y\in \mathcal{Y} : \forall j\in \mathcal{H},\; \psi_j(y)\in \mathcal{C}_j\},$$

where $\mathcal{H}\subseteq\{1,\dots,J\}$ indexes the hard constraints.

Decoding then becomes:

$$\max_{y\in \mathcal{Y}_{hard}(x,\mathcal{F})} \log p_\theta(y\mid x) - \lambda\sum_{j\notin \mathcal{H}} w_j\,\ell_j(\psi_j(y),\mathcal{C}_j) - \mu\,\mathcal{L}_{sem}(y;x).$$

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

Examples include:

- **Range score**:
$$s_j(y) = 1 - \min\left(1, \frac{\ell_j(\psi_j(y),[a,b])}{\kappa_j}\right)$$
for a scaling constant $\kappa_j>0$.

- **Histogram score** (KL):
$$s_j(y) = \exp\big(-\alpha_j\, D_{KL}(\mathbf{h}^*\|\mathbf{h}(y))\big).$$

- **Lexicon hard constraint score**:
$$s_j(y)=\mathbf{1}[\text{no forbidden term appears}].$$

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

## 9. Conclusion

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
        t ← filter_non_voice(t)  # blockquotes, references, footnotes, inline citations
        if length(t) ≥ MIN_LEN:
            texts.append(t)

    M ← compute_measurements(texts)
        # includes: sentence/paragraph histograms, punctuation rates,
        # contractions/oxford comma, function words, stance signals,
        # sentence-openers/templates, n-grams

    if phrase_validation_enabled:
        V ← validate_common_phrases(M.common_phrases, llm=C)
        M.common_phrases_validation ← V

    E ← pick_representative_excerpts(files, char_budget=B, voice_scoring=on)
    L ← load_lexicon_hints(optional)

    prompt ← build_fingerprint_prompt(schema=S, measurements=M, excerpts=E,
                                      lexicon_hints=L, model=C.model)
    raw ← call_llm_chat_completions(prompt, C)

    if is_valid_json(raw):
        F ← parse_json(raw)
    else:
        repair_prompt ← build_json_repair_prompt(raw)
        raw2 ← call_llm_chat_completions(repair_prompt, C)
        F ← parse_json(raw2)

    # enforce invariants
    F.schema_version ← default_if_missing(F.schema_version, "1.0.0")
    F.measurements ← M  # embed verbatim

    write_json(out, F)
    return F
end procedure
```

### A.2 Rewrite (Fingerprint + Draft → Styled Draft)

**Inputs:** Fingerprint $\mathcal{F}$, input Markdown $x$, language model $\mathcal{R}_\theta$  
**Output:** Rewritten Markdown $y$ and deviations report

```text
procedure APPLY_FINGERPRINT(fingerprint F, markdown_path in, output_path out, llm_config C):
    x ← read_text(in)
    x ← strip_base64_images(x)
    x ← mask_non_voice_blocks(x)  # blockquotes, references, footnotes
    x ← mask_inline_citations(x)
    Mx ← compute_measurements(filter_non_voice(x))
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
        "emoji_policy": { "type": "string", "enum": ["remove", "replace", "none"] }
      },
      "additionalProperties": false
    },
    "humanizer_variance": {
      "type": "object",
      "properties": {
        "enabled": { "type": "boolean" },
        "seed": { "type": "integer" },
        "max_ops_per_1000w": { "type": "number", "minimum": 0 },
        "allowed_ops": { "type": "array", "items": { "type": "string" } }
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
- `section_restore.enabled`: enable/disable restoring missing sections after rewrite.
- `section_restore.max_restore_sections`: maximum number of missing sections to restore (0 disables restoration).
- `section_restore.heading_similarity_threshold`: fuzzy heading match threshold for considering a rewritten heading “present”.
- `section_restore.signature_similarity_threshold`: content‑signature similarity threshold for matching a section by its opening content.
- `section_restore.signature_min_overlap`: minimum number of overlapping signature tokens required for a content match.
- `line_count_warn_pct`: if the output line count changes by this percentage or more, a warning is issued for possible missing or expanded content.
- `word_count_warn_pct`: if the output word count changes by this percentage or more, a warning is issued for possible missing or expanded content.
- `paragraph_count_warn_pct`: if the output paragraph count changes by this percentage or more, a warning is issued for possible missing or expanded content.

---

## Licence Notice

This work is licensed under the PolyForm Noncommercial Licence 1.0.0.  
Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).  
See `LICENSE.md` for the full licence text and terms.
