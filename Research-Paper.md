# Stylometric-Transfer: Interpretable Stylometric Profiling and Constraint-Guided Author-Conditioned Style Transfer with Large Language Models

**Repository:** `stylometric-transfer`  
**Keywords:** stylometry, computational stylistics, authorship attribution, controllable text generation, text style transfer, interpretability

---

## Abstract

We present **Stylometric-Transfer**, a practical method for (i) **stylometric profiling** of an author's writing corpus into an explicit, interpretable JSON artifact (a *style fingerprint*) and (ii) **meaning-preserving style transfer** that rewrites new text to conform to the fingerprint using a large language model (LLM). The approach combines classic stylometric measurement--e.g., punctuation rates and sentence-length distributions--with LLM-mediated synthesis into human-editable constraints (ranges, histograms, lexicon rules, rhetorical templates). We formalize the fingerprint as a constraint set and provide a constraint-satisfaction decoding view for LLM rewriting, together with compliance scoring based on distributional divergences. This hybrid design offers an auditable alternative to purely latent "style embeddings" while remaining consistent with established stylometry and text style transfer literature.

---

## 1. Introduction

**Stylometry** studies quantitative signals of writing style for tasks including authorship attribution and author profiling. A canonical demonstration is the Federalist Papers authorship analysis, where frequent-word statistics support Bayesian inference over disputed authorship. ([press.uchicago.edu](https://press.uchicago.edu/ucp/books/book/distributed/I/bo5667096.html?utm_source=chatgpt.com))

Separately, **text style transfer (TST)** aims to transform text so stylistic properties match a target style while preserving style-independent content. A recurring challenge is separating "content" from "style" without parallel data, motivating methods such as cross-alignment approaches and ongoing evaluation/ethical discussions. ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com))

This paper motivates a hybrid approach: represent style explicitly as a **stylometric style fingerprint** (JSON) and use an LLM as a constrained rewriter guided by (a) the fingerprint and (b) locally measured statistics of both the author corpus and the candidate text.

---

## 2. Related Work

### 2.1 Stylometry and Distance-Based Measures

Stylometric authorship attribution typically uses robust, interpretable features (e.g., word frequency profiles) and distance measures. Burrows's Delta and its variants are widely used; more recent work provides detailed explanations that decompose feature selection, feature scaling (e.g., z-transformation), and distance metrics, clarifying why Delta-style measures can be effective. ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))

### 2.2 Text Style Transfer and Evaluation

Non-parallel TST methods such as **cross-alignment** demonstrate the feasibility of changing certain stylistic attributes without parallel sentence pairs. ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com))  
Recent surveys highlight broad application scenarios alongside open challenges in evaluation and ethical risk (e.g., misuse for impersonation), supporting explicit safeguards and transparency in TST pipelines. ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com))

---

## 3. Problem Setup

Let an author corpus be a set of documents

$$\mathcal{D} = \{d_1,\dots,d_N\}, \quad d_i \in \Sigma^*$$

where $\Sigma$ is a character alphabet.

We define:

- An **interpretable feature extractor** $\phi: \Sigma^* \to \mathbb{R}^K$ producing measurable statistics (rates, histograms, counts).
- A **style fingerprint** $\mathcal{F}$ storing target statistics, distributions, and discrete constraints (lexicon rules, templates).
- A **rewriter** $\mathcal{R}_\theta$ (LLM with parameters $\theta$) mapping input text $x$ to output $y$:

$$y = \mathcal{R}_\theta(x \mid \mathcal{F}).$$

**Primary constraint:** meaning preservation (no new facts, claims, or examples; preserve entities and numerals unless explicitly permitted).

---

## 4. Stylometric Measurements

### 4.1 Rate and Density Features

Let $W(d)$ be an approximate word-token count and $C_e(d)$ the count of an event $e$ (e.g., commas). Define per-1000-word rates:

$$r_e(d) = 1000 \cdot \frac{C_e(d)}{\max(1, W(d))}.$$

The fingerprint stores targets as tolerance intervals:

$$r_e \in [\underline{r}_e, \overline{r}_e],$$

reflecting intra-author variability across topics and subgenres.

### 4.2 Histogram Features

For sentence lengths $\ell_1,\dots,\ell_m$ (in words), define a binned histogram

$$\mathbf{h} \in \Delta^{B-1}, \quad h_b = \frac{1}{m}\sum_{i=1}^m \mathbf{1}[\ell_i \in \text{bin}(b)],$$

where $\Delta^{B-1}$ is the probability simplex and bins are ordinal intervals (e.g., $<10$, 10-17, 18-25, ...).

### 4.3 Delta-Style Diagnostics (Optional)

While Stylometric-Transfer is not an authorship attribution system, Delta-style distances can serve as *diagnostic* measures of stylistic proximity. Following standardization and Manhattan-style aggregation:

$$\Delta(d,d') = \frac{1}{K}\sum_{k=1}^K \left|z_k(d) - z_k(d')\right|,$$

where $z_k$ is the z-transformed version of feature $k$. Detailed decompositions and explanations of Delta variants motivate this lens. ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))

---

## 5. The Style Fingerprint as a Constraint Model

We treat the fingerprint as a set of weighted constraints:

$$\mathcal{F} = \{(\psi_j, \mathcal{C}_j, w_j)\}_{j=1}^J,$$

where:
- $\psi_j(y)$ is a measurable statistic of output text (e.g., comma rate, histogram vector).
- $\mathcal{C}_j$ is an admissible set (range, divergence tolerance, forbidden list).
- $w_j$ is a weight (priority).

Typical constraint types:

1. **Range constraints**: $\psi_j(y) \in [a,b]$  
2. **Histogram constraints**: $D(\mathbf{h}^*, \mathbf{h}(y)) \le \tau$  
3. **Lexicon constraints**: forbidden phrases/words; preferred synonyms  
4. **Template constraints**: rhetorical move frequency bounds  

The JSON representation adds practical control fields such as `priority_order` and `strictness` to determine constraint precedence.

---

## 6. Constraint Satisfaction Decoding and Compliance Scoring

This section expands the mathematical view of rewriting as a **constraint satisfaction** problem.

### 6.1 Soft-Constrained Objective

Let $p_\theta(y\mid x)$ be the LLM's conditional probability of an output $y$ given input $x$. We define a soft-constrained objective:

$$\max_{y \in \mathcal{Y}} \; \log p_\theta(y \mid x) - \lambda\, \mathcal{L}_{style}(y;\mathcal{F}) - \mu\,\mathcal{L}_{sem}(y;x),$$

where:
- $\mathcal{L}_{style}$ penalizes deviation from the fingerprint.
- $\mathcal{L}_{sem}$ penalizes semantic drift (approximated conservatively via invariants; optionally via semantic similarity models).

A standard decomposition is:

$$\mathcal{L}_{style}(y;\mathcal{F}) = \sum_{j=1}^J w_j\, \ell_j(\psi_j(y), \mathcal{C}_j).$$

Example penalties:

**Range penalty** for $\mathcal{C}_j=[a,b]$:

$$\ell_j(v,[a,b]) = \big(\max(0,a-v)\big)^2 + \big(\max(0,v-b)\big)^2.$$

**Histogram penalty** using KL divergence:

$$\ell_j(\mathbf{h},\mathbf{h}^*) = D_{KL}(\mathbf{h}^*\|\mathbf{h}) = \sum_{b=1}^B h^*_b \log \frac{h^*_b}{\max(\epsilon,h_b)}.$$

(For ordinal bins, Wasserstein distance $W_1$ is often preferable; the implementation may adopt either.)

### 6.2 Hard Constraints (Feasibility)

Some constraints are best treated as hard feasibility requirements:

- Entity/number preservation constraints $\Rightarrow$ must hold unless explicitly overridden.
- Hard forbidden lexicon constraints (e.g., "must not appear").

Define the feasible set:

$$\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y\in \mathcal{Y} : \forall j\in \mathcal{H},\; \psi_j(y)\in \mathcal{C}_j\},$$

where $\mathcal{H}\subseteq\{1,\dots,J\}$ indexes hard constraints.

Then decoding becomes:

$$\max_{y\in \mathcal{Y}_{hard}(x,\mathcal{F})} \log p_\theta(y\mid x) - \lambda\sum_{j\notin \mathcal{H}} w_j\,\ell_j(\psi_j(y),\mathcal{C}_j) - \mu\,\mathcal{L}_{sem}(y;x).$$

### 6.3 Practical Constraint-Satisfaction Decoding Procedure

In production LLM use, exact constrained decoding over $\mathcal{Y}_{hard}$ is rarely available. Stylometric-Transfer approximates constraint satisfaction using **(i) instruction prompting**, **(ii) self-audit**, and **(iii) repair**.

A practical decoding approximation:

1. Generate a candidate rewrite $y^{(0)}$ from the LLM under explicit instructions encoding $\mathcal{F}$.
2. Compute local measurements $\phi(y^{(t)})$ and audit constraint violations.
3. If violations exist, re-prompt the LLM with a structured report to obtain $y^{(t+1)}$.
4. Stop when compliance exceeds a threshold or iteration limit.

### 6.4 Compliance Scoring

Define a normalized compliance score $S(y;\mathcal{F})\in[0,1]$ aggregating constraint satisfaction:

$$S(y;\mathcal{F}) = \sigma\Big(\sum_{j=1}^J w_j\, s_j(y)\Big), \quad \sum_j w_j = 1,$$

where $\sigma$ is a squashing function (e.g., identity clipped to $[0,1]$, or logistic), and $s_j(y)\in[0,1]$ is a per-constraint score.

Examples:

- **Range score**:
$$s_j(y) = 1 - \min\left(1, \frac{\ell_j(\psi_j(y),[a,b])}{\kappa_j}\right)$$
for a scaling constant $\kappa_j>0$.

- **Histogram score** (KL):
$$s_j(y) = \exp\big(-\alpha_j\, D_{KL}(\mathbf{h}^*\|\mathbf{h}(y))\big).$$

- **Lexicon hard constraint score**:
$$s_j(y)=\mathbf{1}[\text{no forbidden term appears}].$$

This compliance score supports:
- reporting (`validators.weights` and `checks` in JSON)
- iterative repair thresholds
- regression tests for stability

---

## 7. Implementation Notes (Stylometric-Transfer)

The repository implements:

1. **Local measurement stage**
   - sentence-length histogram
   - paragraph-length histogram
   - punctuation rates per 1000 words
   - contraction/dash/ellipsis signals
   - frequent n-grams (diagnostic lexicon hints)

2. **LLM synthesis stage**
   - schema-guided JSON-only prompting
   - embed measurements verbatim
   - automated JSON repair pass if parsing fails

3. **Rewrite stage**
   - fingerprint + input measurements + markdown text
   - JSON output: rewritten markdown + deviations + self-check

These design choices align with stylometric traditions emphasizing interpretable features and with TST concerns about evaluation and ethical risk. ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))

---

## 8. Ethical Considerations

TST can be misused for impersonation-like behaviors; recent surveys explicitly highlight ethical considerations and the need for safeguards. ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com))

Stylometric-Transfer is intended for:
- self-authored corpora
- licensed/public-domain corpora
- editing assistance and personal voice consistency

Recommended safeguards:
- provenance tracking in `metadata`
- default controls that discourage third-party imitation
- deviation reporting when constraints conflict with meaning preservation

---

## 9. Conclusion

Stylometric-Transfer bridges **classic stylometry** and **LLM-based rewriting** by pairing interpretable, versionable style models with constraint-guided generation. The explicit JSON fingerprint improves auditability and editorial control while drawing on well-established stylometric measurement and style transfer insights. ([press.uchicago.edu](https://press.uchicago.edu/ucp/books/book/distributed/I/bo5667096.html?utm_source=chatgpt.com))

---

## References

- Mosteller, F., & Wallace, D. L. *Inference and Disputed Authorship: The Federalist.* Addison-Wesley (1964). ([archive.org](https://archive.org/details/inferencedispute00most?utm_source=chatgpt.com))
- Evert, S., et al. "Understanding and explaining Delta measures for authorship attribution." *Digital Scholarship in the Humanities* (2017). ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))
- Shen, T., Lei, T., Barzilay, R., & Jaakkola, T. "Style Transfer from Non-Parallel Text by Cross-Alignment." (2017). ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com))
- Mukherjee, S., et al. "A Survey of Text Style Transfer: Applications and Ethical Implications." (2024). ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com))
- Hu, Z., et al. "Text Style Transfer: A Review and Experimental Evaluation." *KDD Explorations* (PDF). ([kdd.org](https://www.kdd.org/exploration_files/vol24issue1_2._Text_Style_Transfer__A_Review_and_Experimental_Evaluation.pdf?utm_source=chatgpt.com))

---

## Appendix A. Methods (Pseudocode)

This appendix provides pseudocode for the **fingerprinter** (extractor) and **rewriter** stages.

### A.1 Fingerprint Extraction (Corpus → Style Fingerprint JSON)

**Inputs:** corpus archive $A$, LLM $\mathcal{R}_\theta$, schema template $S$  
**Output:** style fingerprint $\mathcal{F}$ (JSON)

```text
procedure FINGERPRINT_STYLE(archive A, output_path out, llm_config C):
    tmp_dir ← extract_archive(A)
    files ← list_textlike_files(tmp_dir, extensions={.txt,.md,.rst,.html,.docx})
    texts ← []
    for f in files:
        t ← read_and_normalize(f)
        if length(t) ≥ MIN_LEN:
            texts.append(t)

    M ← compute_measurements(texts)
        # includes: sentence histogram, paragraph histogram,
        # punctuation rates, contractions, dashes, n-grams

    E ← pick_representative_excerpts(files, char_budget=B)

    prompt ← build_fingerprint_prompt(schema=S, measurements=M, excerpts=E, model=C.model)
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

**Inputs:** fingerprint $\mathcal{F}$, input Markdown $x$, LLM $\mathcal{R}_\theta$  
**Output:** rewritten Markdown $y$ and deviations report

```text
procedure APPLY_FINGERPRINT(fingerprint F, markdown_path in, output_path out, llm_config C):
    x ← read_text(in)
    Mx ← compute_measurements(x)

    prompt ← build_rewrite_prompt(fingerprint=F, input_measurements=Mx, input_text=x)
    raw ← call_llm_chat_completions(prompt, C)

    if is_valid_json(raw):
        obj ← parse_json(raw)
    else:
        obj ← parse_json(call_llm_chat_completions(build_json_repair_prompt(raw), C))

    y ← obj.final_markdown
    deviations ← obj.deviations

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
        audit ← evaluate_constraints(F, y)
        if audit.score ≥ F.validators.overall_threshold.pass AND audit.hard_violations == 0:
            break
        y ← call_llm_repair(F, x, y, audit, C)

    return y
end procedure
```


---

## Appendix B. Formal Constrained Decoding Framing

This appendix tightens the decoding formulation into a standard constrained optimization / constrained MDP view.

### B.1 Constrained Maximum A Posteriori Decoding

Let $p_\theta(y\mid x)$ denote the base LLM distribution. Let constraints be indexed by $j=1,\dots,J$ with statistics $\psi_j(y)$ and admissible sets $\mathcal{C}_j$.

We define the feasible set of hard constraints:

$$\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y : \forall j \in \mathcal{H},\; \psi_j(y) \in \mathcal{C}_j\}$$

The constrained MAP problem is:

$$\hat y = \arg\max_{y \in \mathcal{Y}_{hard}(x,\mathcal{F})} \; \log p_\theta(y\mid x)$$

In practice, $\mathcal{Y}_{hard}$ is not explicitly enumerable. We therefore relax the problem using a **Lagrangian penalty formulation**:

$$\hat y = \arg\max_{y \in \mathcal{Y}} \; \log p_\theta(y\mid x)
- \sum_{j=1}^J \lambda_j \cdot g_j(\psi_j(y))
- \mu \cdot \mathcal{L}_{sem}(y;x),$$

where:

- $g_j(\cdot)$ is a non-negative violation function such that $g_j(v)=0$ iff $v \in \mathcal{C}_j$
- $\lambda_j \ge 0$ are Lagrange multipliers derived from `validators.weights`
- $\mathcal{L}_{sem}$ enforces meaning preservation

This matches the standard **soft-constrained decoding** paradigm used in controllable generation and lexically constrained decoding.

---

### B.2 Projection View

Equivalently, rewriting can be interpreted as projection of an unconstrained sample $y^{(0)} \sim p_\theta(\cdot \mid x)$ onto the admissible region:

$$\hat y = \Pi_{\mathcal{C}}(y^{(0)}) = \arg\min_{y} \; d(y, y^{(0)}) + \sum_j \lambda_j g_j(\psi_j(y)),$$

where $d(\cdot,\cdot)$ is an edit or semantic divergence.  
In practice, $\Pi_{\mathcal{C}}$ is approximated by **LLM self-repair passes** guided by explicit audit reports.

---

### B.3 Constrained Markov Decision Process (CMDP) Interpretation

Token generation may be framed as a CMDP:

- States: $s_t = y_{1:t}$  
- Actions: $a_t = y_{t+1}$  
- Reward: $r_t = \log p_\theta(a_t\mid s_t,x)$  
- Costs: $c_{j,t}$ accumulating toward $\psi_j(y)$

with terminal constraints:

$$\mathbb{E}\Big[ \sum_t c_{j,t} \Big] \le \tau_j$$

This clarifies that the system approximates **policy optimization under global style budgets**, implemented via instruction-guided generation and post-hoc repair.

---

## Appendix C. Evaluation and Acceptance Criteria

This appendix defines concrete divergence metrics and acceptance thresholds mapped directly to the fingerprint JSON fields.

### C.1 Metric Families

#### (1) Rate Constraints (scalar)

For a target interval $[a,b]$ and observed value $v$:

$$\text{viol}_r(v) = \max(0,a-v) + \max(0,v-b)$$

Score:

$$s_r(v) = \exp(-\alpha_r \cdot \text{viol}_r(v))$$

Mapped JSON paths:
- `/targets/punctuation/comma_density_per_100w`
- `/targets/orthography/contractions_rate`

---

#### (2) Histogram Constraints (sentence / paragraph)

Primary metric: **Wasserstein-1 distance**

$$W_1(\mathbf{h}^*, \mathbf{h}) =
\sum_{b=1}^{B-1} \left| \sum_{k=1}^b (h_k - h_k^*) \right|$$

Secondary diagnostic: KL divergence

$$D_{KL}(\mathbf{h}^* \| \mathbf{h}) =
\sum_b h_b^* \log \frac{h_b^*}{\max(\epsilon,h_b)}$$

Score:

$$s_h = \exp(-\alpha_h W_1)$$

Mapped JSON:
- `/targets/sentence/length_words/distribution`
- `/targets/paragraph/length_sentences`

---

#### (3) Lexicon Constraints

Hard:

$$s_{lex}^{hard} = \mathbf{1}[\text{no forbidden term appears}]$$

Soft:

$$s_{lex}^{soft} = \exp(-\alpha_{lex} \cdot |f_y - f^*|)$$

Mapped JSON:
- `/lexicon/avoid_words`
- `/lexicon/avoid_phrases`
- `/lexicon/preferred_phrases`

---

### C.2 Aggregated Compliance Score

Let weights $w_j$ come from `validators.weights` with $\sum_j w_j = 1$.

$$S(y;\mathcal{F}) = \sum_{j=1}^J w_j s_j(y)$$

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

Let $S_t$ be the score at iteration $t$. Stop when:

$$S_t \ge S_{pass} \quad \text{and} \quad H_t = 0$$

Else continue up to $T_{max}$ repair passes.

---

## Appendix D. Mechanism of Fingerprint-Conditioned Rewriting

This appendix provides a detailed account of **how an explicit stylometric fingerprint guides an LLM to rewrite text in the target author style**, despite the LLM's internal representations being latent and opaque. We formalize the process as *externalized style conditioning* through instruction embedding, constraint activation, and iterative projection.

---

## D.1 From Stylometric Profile to Control Signals

The style fingerprint $\mathcal{F}$ is not consumed by the LLM as raw statistics, but rather as a **compiled control representation** consisting of:

1. **Numeric constraints**  
   (ranges, histograms, tolerances)

2. **Discrete symbolic constraints**  
   (lexicon rules, rhetorical templates, structural policies)

3. **Priority and strictness controls**  
   (ordering, hard vs soft constraints)

4. **Derived natural-language instructions**  
   (compiled in `derived_instructions.*`)

We denote the compiled instruction set as:

$$\mathcal{I}(\mathcal{F}) = \text{Compile}(\mathcal{F})$$

where $\mathcal{I}(\mathcal{F})$ is a structured textual representation injected into the LLM prompt.

This compilation step performs three key transformations:

### (i) Constraint verbalization

Numeric constraints are converted into qualitative instructions:

- "Use short-to-medium sentences (10-18 words typical)"  
- "Favor one-sentence paragraphs occasionally (~15%)"  
- "Avoid heavy semicolon usage; commas preferred"  

This converts $\psi_j(y)\in\mathcal{C}_j$ into *behavioral descriptors*.

### (ii) Salience weighting

Constraint weights $w_j$ are mapped to:

- ordering in the prompt  
- emphasis (phrasing, repetition)  
- explicit "must" vs "prefer" language  

### (iii) Conflict resolution policy

The `controls.priority_order` field induces a partial order:

$$\text{meaning preservation} \succ \text{lexicon} \succ \text{sentence rhythm} \succ \text{punctuation} \succ \text{templates}$$

This ordering is verbalized explicitly to ensure that stylistic fidelity never overrides semantic fidelity.

---

## D.2 Conditioning as External Latent Space Steering

Let $h(x)$ denote the latent representation of the input text under the LLM, and let $c(\mathcal{I})$ denote the latent encoding of the instruction set.

The model samples from:

$$p_\theta(y \mid x, \mathcal{I}) = p_\theta(y \mid h(x), c(\mathcal{I}))$$

We interpret $c(\mathcal{I})$ as inducing a **soft bias over stylistic manifolds** in latent space.

Rather than learning a new style embedding, the fingerprint:

- activates regions of latent space corresponding to sentence rhythm  
- biases token transitions associated with punctuation patterns  
- suppresses lexical clusters disfavored by the lexicon rules  

This is analogous to **feature activation steering** in controllable generation, except the features are externalized and interpretable.

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
   
   - Let $\mathcal{P}(x)$ be the set of meaning-preserving paraphrases of $x$ (non-empty for any non-degenerate $x$).  
   - Feature maps $\psi_j$ are continuous (or piecewise continuous) under paraphrase operations.  
   - By assumption, tolerance intervals include a neighborhood around $\psi_j(x)$.  
   - Therefore, there exists $y \in \mathcal{P}(x)$ such that $\psi_j(y)\in\mathcal{C}_j$ for all $j$. ∎
   
   ---
   
   ## E.4 Constraint Compatibility and Conflict Graphs
   
   Define a constraint compatibility graph:
   
   - Nodes: constraints $j$  
   - Edge between $j,k$ if $\mathcal{C}_j \cap \mathcal{C}_k = \varnothing$ under semantic invariants  
   
   A necessary condition for feasibility is:
   
   $$\text{Graph}(\mathcal{F}) \text{ is bipartite with respect to hard constraints}$$
   
   In practice:
   
   - Sentence rhythm vs paragraph rhythm are compatible  
   - Lexicon vs semantic invariants may conflict  
   - Template vs rhythm may conflict on short texts  
   
   The system enforces a partial order:
   
   $$\text{meaning} \succ \text{lexicon} \succ \text{structure} \succ \text{punctuation} \succ \text{templates}$$
   
   ensuring that conflicts collapse in favor of feasibility.
   
   ---
   
   ## E.5 Minimal Tolerance Bounds
   
   Let $\sigma_j$ be the empirical standard deviation of feature $\psi_j$ over the author corpus.
   
   Recommended sufficient tolerances:
   
   - Range constraints:
   $$[a_j, b_j] = [\mu_j - 2\sigma_j,\; \mu_j + 2\sigma_j]$$
   
   - Histogram constraints:
   $$\tau_j \ge 2 \cdot \mathbb{E}[W_1(\mathbf{h},\mathbf{h}')]$$
   
   where $\mathbf{h},\mathbf{h}'$ are histograms from independent samples of the corpus.
   
   These ensure that:
   
   - intra-author variation is admissible  
   - projection steps remain contractive  
   
   ---
   
   ## E.6 Convergence of Iterative Repair
   
   Let $S(y;\mathcal{F})$ be the compliance score.
   
   Assume:
   
   1. Each repair step reduces total violation:
   $$\mathbb{E}[S(y^{(t+1)})] \ge S(y^{(t)}) + \eta$$
   for some $\eta > 0$
   
   2. $S$ is bounded above by 1
   
   Then:
   
   $$\exists T < \infty : S(y^{(T)}) \ge S_{pass}$$
   
   i.e., **finite-step convergence** in expectation.
   
   This explains empirically observed rapid convergence (1-3 iterations) in most rewrites.
   
   ---
   
   ## E.7 Degenerate and Infeasible Cases
   
   Feasibility may fail when:
   
   1. **Extremely short texts**  
      Insufficient degrees of freedom for histogram control.
   
   2. **Highly constrained technical content**  
      Semantic invariants dominate stylistic degrees of freedom.
   
   3. **Over-tight tolerances**  
      $\tau_j < \epsilon_j$
   
   In such cases, Stylometric-Transfer:
   
   - reports deviation  
   - relaxes lowest-priority constraints  
   - guarantees semantic correctness  
   
   ---
   
   ## E.8 Interpretation
   
   The fingerprint does not specify a point in style space, but a **convex (or approximately convex) admissible region**.
   
   Rewriting succeeds when:
   
   $$\mathcal{E}(x) \cap \mathcal{M}_{style} \neq \varnothing$$
   
   This framing clarifies why:
   
   - tolerances are essential  
   - strict imitation is ill-posed  
   - deviation reporting is principled  
   
   ---
   
## Appendix F. Comparison with Fine-Tuning, LoRA, and Latent Style Embedding Approaches

This appendix situates Stylometric-Transfer among existing approaches to author-style modeling and controlled generation.

---

## F.1 Taxonomy of Style Modeling Approaches

We distinguish four dominant paradigms:

| Paradigm | Representation | Training | Interpretability | Editability |
|----------|----------------|----------|------------------|-------------|
| Fine-tuning | Model weights | Required | None | None |
| LoRA / adapters | Low-rank deltas | Required | None | None |
| Latent embeddings | Vectors $z_{style}$ | Required | Low | None |
| **Stylometric-Transfer** | Explicit constraints | None | **High** | **Full** |

---

## F.2 Fine-Tuning Approaches

### Mechanism

Fine-tuning learns:

$$p_{\theta'}(y\mid x) \approx p(y\mid x,\text{author})$$

by modifying base parameters $\theta \to \theta'$.

### Limitations

- Style representation is **entirely implicit**  
- No inspection of learned stylistic features  
- No partial control (cannot weight sentence rhythm vs lexicon)  
- Catastrophic forgetting risk  
- Expensive retraining for each author  

### Contrast

Stylometric-Transfer instead solves:

$$\max_{y \in \mathcal{M}_{style}} p_\theta(y\mid x)$$

with:

- no parameter updates  
- explicit admissible region  
- post-hoc auditing  

---

## F.3 LoRA / Adapter-Based Style Conditioning

### Mechanism

Learn low-rank matrices $\Delta W$ such that:

$$h' = h + \Delta W h$$

encoding author-specific modulation.

### Advantages

- Efficient  
- Modular  

### Limitations

- Style encoded in **latent linear subspace**  
- Not interpretable  
- No direct mapping to stylometric features  
- Difficult to combine multiple styles  

### Contrast

Stylometric-Transfer:

- exposes every control dimension  
- allows continuous interpolation via tolerances  
- supports manual editing and versioning  

---

## F.4 Latent Style Embedding Methods

### Mechanism

Learn a vector:

$$z_{style} \in \mathbb{R}^d$$

and condition generation:

$$p(y\mid x,z_{style})$$

via cross-alignment, VAEs, or conditional decoders.

### Advantages

- Compact  
- Differentiable  

### Limitations

- Entangled dimensions  
- No semantic interpretation of coordinates  
- No guarantee that $z_{style}$ corresponds to human-meaningful features  
- No auditability  

### Contrast

Stylometric-Transfer replaces:

$$z_{style}
\quad \longrightarrow \quad
\mathcal{F} = \{(\psi_j,\mathcal{C}_j,w_j)\}$$

yielding:

- explicit axes of variation  
- measurable compliance  
- verifiable reproduction  

---

## F.5 Control Granularity and Editorial Authority

A key distinction is **who controls style**.

| Property | Fine-tune | LoRA | Embedding | Stylometric-Transfer |
|----------|-----------|------|-----------|----------------------|
| Human-readable model | ✗ | ✗ | ✗ | **✓** |
| Partial constraint weighting | ✗ | ✗ | ✗ | **✓** |
| Manual editing | ✗ | ✗ | ✗ | **✓** |
| Version control | ✗ | ✗ | ✗ | **✓** |
| Deviation reporting | ✗ | ✗ | ✗ | **✓** |

Stylometric-Transfer treats style as an **editorial object**, not merely a training artifact.

---

## F.6 Data Efficiency

Fine-tuning and embedding methods require:

$$N \gg 10^4 \text{ tokens}$$

to stabilize latent style representations.

Stylometric-Transfer requires:

- only sufficient data to estimate low-variance statistics  
- often $N \approx 10^3-10^4$ tokens  
- robust even on heterogeneous corpora  

---

## F.7 Transferability and Compositionality

Latent methods struggle with:

- combining multiple authors  
- interpolating interpretable features  
- transferring style across domains  

Stylometric-Transfer supports:

- convex combinations of fingerprints  
- selective inheritance of features  
- domain-specific constraint relaxation  

Formally, fingerprints compose as:

$$\mathcal{F}_\lambda = \lambda \mathcal{F}_1 + (1-\lambda)\mathcal{F}_2$$

at the level of:

- histogram mixtures  
- range interpolation  
- lexicon unions  

---

## F.8 Interpretability and Scientific Value

From a scientific standpoint:

- fine-tuning learns *unknown* features  
- embeddings encode *unlabeled* dimensions  
- Stylometric-Transfer recovers **measurable linguistic variables**  

This enables:

- hypothesis testing  
- ablation studies  
- stylistic causality analysis  
- reproducible experiments  

---

## F.9 Summary

Stylometric-Transfer differs fundamentally from existing approaches by:

1. Externalizing style into explicit constraints  
2. Avoiding training and latent embeddings  
3. Enabling auditability and editorial control  
4. Supporting theoretical analysis of feasibility and convergence  

Rather than learning *what* style is, it defines *where* style is allowed to live in feature space.

---

## Appendix G. JSON schema
   
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
            "limitations": { "type": "array", "items": { "type": "string" } }
          }
        }
      }
    },
    "measurements": {
      "type": "object",
      "description": "Raw stylometric measurements extracted from corpus",
      "required": [
        "sentence_length_histogram",
        "paragraph_length_histogram",
        "punctuation_rates",
        "orthography",
        "structure"
      ],
      "properties": {
        "sentence_length_histogram": { "$ref": "#/definitions/histogram" },
        "paragraph_length_histogram": { "$ref": "#/definitions/histogram" },
        "punctuation_rates": {
          "type": "object",
          "description": "Per-1000-word punctuation densities",
          "properties": {
            "comma": { "type": "number" },
            "semicolon": { "type": "number" },
            "colon": { "type": "number" },
            "dash": { "type": "number" },
            "ellipsis": { "type": "number" },
            "exclamation": { "type": "number" },
            "question": { "type": "number" }
          }
        },
        "orthography": {
          "type": "object",
          "properties": {
            "contractions_rate": { "type": "number" },
            "uppercase_sentence_rate": { "type": "number" },
            "quotes_double_ratio": { "type": "number" }
          }
        },
        "structure": {
          "type": "object",
          "properties": {
            "one_sentence_paragraph_rate": { "type": "number" },
            "avg_sentences_per_paragraph": { "type": "number" }
          }
        }
      }
    },
    "targets": {
      "type": "object",
      "description": "Target ranges and distributions for rewriting",
      "required": ["sentence", "paragraph", "punctuation", "orthography"],
      "properties": {
        "sentence": {
          "type": "object",
          "properties": {
            "length_words": {
              "type": "object",
              "required": ["distribution", "tolerance"],
              "properties": {
                "distribution": { "$ref": "#/definitions/histogram" },
                "tolerance": { "type": "number", "default": 0.08 }
              }
            }
          }
        },
        "paragraph": {
          "type": "object",
          "properties": {
            "length_sentences": {
              "type": "object",
              "required": ["distribution", "tolerance"],
              "properties": {
                "distribution": { "$ref": "#/definitions/histogram" },
                "tolerance": { "type": "number", "default": 0.10 }
              }
            }
          }
        },
        "punctuation": {
          "type": "object",
          "properties": {
            "comma_density_per_100w": { "$ref": "#/definitions/range" },
            "semicolon_density_per_100w": { "$ref": "#/definitions/range" },
            "dash_density_per_100w": { "$ref": "#/definitions/range" },
            "ellipsis_density_per_100w": { "$ref": "#/definitions/range" }
          }
        },
        "orthography": {
          "type": "object",
          "properties": {
            "contractions_rate": { "$ref": "#/definitions/range" },
            "uppercase_sentence_rate": { "$ref": "#/definitions/range" }
          }
        }
      }
    },
    "lexicon": {
      "type": "object",
      "required": ["avoid_words", "avoid_phrases", "preferred_phrases"],
      "properties": {
        "avoid_words": {
          "type": "array",
          "items": { "type": "string" }
        },
        "avoid_phrases": {
          "type": "array",
          "items": { "type": "string" }
        },
        "preferred_phrases": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "templates": {
      "type": "object",
      "description": "Rhetorical and syntactic templates",
      "properties": {
        "sentence_openers": {
          "type": "array",
          "items": { "type": "string" }
        },
        "paragraph_openers": {
          "type": "array",
          "items": { "type": "string" }
        },
        "preferred_transitions": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "controls": {
      "type": "object",
      "required": ["priority_order", "strictness", "rewrite_policy"],
      "properties": {
        "priority_order": {
          "type": "array",
          "description": "Constraint families in decreasing priority",
          "items": {
            "enum": [
              "meaning_preservation",
              "lexicon",
              "sentence_rhythm",
              "paragraph_structure",
              "punctuation",
              "orthography",
              "templates"
            ]
          }
        },
        "strictness": {
          "type": "object",
          "properties": {
            "hard_constraints": {
              "type": "array",
              "items": { "type": "string" }
            },
            "soft_constraints": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        },
        "rewrite_policy": {
          "type": "object",
          "properties": {
            "preserve_entities": { "type": "boolean", "default": true },
            "preserve_numbers": { "type": "boolean", "default": true },
            "allow_sentence_split": { "type": "boolean", "default": true },
            "allow_sentence_merge": { "type": "boolean", "default": true }
          }
        }
      }
    },
    "validators": {
      "type": "object",
      "required": ["weights", "scoring", "checks"],
      "properties": {
        "weights": {
          "type": "object",
          "description": "Per-constraint weights (normalized externally)",
          "additionalProperties": { "type": "number", "minimum": 0 }
        },
        "scoring": {
          "type": "object",
          "properties": {
            "overall_threshold": {
              "type": "object",
              "properties": {
                "pass": { "type": "number", "default": 0.75 },
                "warn": { "type": "number", "default": 0.60 }
              }
            }
          }
        },
        "checks": {
          "type": "object",
          "properties": {
            "max_iterations": { "type": "integer", "default": 3 },
            "require_zero_hard_violations": { "type": "boolean", "default": true }
          }
        }
      }
    },
    "derived_instructions": {
      "type": "object",
      "description": "Compiled prompts and guidance for LLM",
      "properties": {
        "fingerprint_prompt": { "type": "string" },
        "rewrite_prompt": { "type": "string" },
        "repair_prompt": { "type": "string" }
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
    }
  }
}
```
