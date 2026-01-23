# Stylometric-Transfer: Interpretable Stylometric Profiling and Constraint-Guided Author-Conditioned Style Transfer with Large Language Models

**Repository:** `stylometric-transfer`  
**Keywords:** stylometry, computational stylistics, authorship attribution, controllable text generation, text style transfer, interpretability

---

## Abstract

We present **Stylometric-Transfer**, a practical method for (i) **stylometric profiling** of an author’s writing corpus into an explicit, interpretable JSON artifact (a *style fingerprint*) and (ii) **meaning-preserving style transfer** that rewrites new text to conform to the fingerprint using a large language model (LLM). The approach combines classic stylometric measurement—e.g., punctuation rates and sentence-length distributions—with LLM-mediated synthesis into human-editable constraints (ranges, histograms, lexicon rules, rhetorical templates). We formalize the fingerprint as a constraint set and provide a constraint-satisfaction decoding view for LLM rewriting, together with compliance scoring based on distributional divergences. This hybrid design offers an auditable alternative to purely latent “style embeddings” while remaining consistent with established stylometry and text style transfer literature.

---

## 1. Introduction

**Stylometry** studies quantitative signals of writing style for tasks including authorship attribution and author profiling. A canonical demonstration is the Federalist Papers authorship analysis, where frequent-word statistics support Bayesian inference over disputed authorship. ([press.uchicago.edu](https://press.uchicago.edu/ucp/books/book/distributed/I/bo5667096.html?utm_source=chatgpt.com))

Separately, **text style transfer (TST)** aims to transform text so stylistic properties match a target style while preserving style-independent content. A recurring challenge is separating “content” from “style” without parallel data, motivating methods such as cross-alignment approaches and ongoing evaluation/ethical discussions. ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com))

This paper motivates a hybrid approach: represent style explicitly as a **stylometric style fingerprint** (JSON) and use an LLM as a constrained rewriter guided by (a) the fingerprint and (b) locally measured statistics of both the author corpus and the candidate text.

---

## 2. Related Work

### 2.1 Stylometry and Distance-Based Measures

Stylometric authorship attribution typically uses robust, interpretable features (e.g., word frequency profiles) and distance measures. Burrows’s Delta and its variants are widely used; more recent work provides detailed explanations that decompose feature selection, feature scaling (e.g., z-transformation), and distance metrics, clarifying why Delta-style measures can be effective. ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))

### 2.2 Text Style Transfer and Evaluation

Non-parallel TST methods such as **cross-alignment** demonstrate the feasibility of changing certain stylistic attributes without parallel sentence pairs. ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com))  
Recent surveys highlight broad application scenarios alongside open challenges in evaluation and ethical risk (e.g., misuse for impersonation), supporting explicit safeguards and transparency in TST pipelines. ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com))

---

## 3. Problem Setup

Let an author corpus be a set of documents

\[
\mathcal{D} = \{d_1,\dots,d_N\}, \quad d_i \in \Sigma^*
\]

where \(\Sigma\) is a character alphabet.

We define:

- An **interpretable feature extractor** \(\phi: \Sigma^* \to \mathbb{R}^K\) producing measurable statistics (rates, histograms, counts).
- A **style fingerprint** \(\mathcal{F}\) storing target statistics, distributions, and discrete constraints (lexicon rules, templates).
- A **rewriter** \(\mathcal{R}_\theta\) (LLM with parameters \(\theta\)) mapping input text \(x\) to output \(y\):

\[
 y = \mathcal{R}_\theta(x \mid \mathcal{F}).
\]

**Primary constraint:** meaning preservation (no new facts, claims, or examples; preserve entities and numerals unless explicitly permitted).

---

## 4. Stylometric Measurements

### 4.1 Rate and Density Features

Let \(W(d)\) be an approximate word-token count and \(C_e(d)\) the count of an event \(e\) (e.g., commas). Define per-1000-word rates:

\[
 r_e(d) = 1000 \cdot \frac{C_e(d)}{\max(1, W(d))}.
\]

The fingerprint stores targets as tolerance intervals:

\[
 r_e \in [\underline{r}_e, \overline{r}_e],
\]

reflecting intra-author variability across topics and subgenres.

### 4.2 Histogram Features

For sentence lengths \(\ell_1,\dots,\ell_m\) (in words), define a binned histogram

\[
\mathbf{h} \in \Delta^{B-1}, \quad h_b = \frac{1}{m}\sum_{i=1}^m \mathbf{1}[\ell_i \in \text{bin}(b)],
\]

where \(\Delta^{B-1}\) is the probability simplex and bins are ordinal intervals (e.g., \(<10\), 10–17, 18–25, …).

### 4.3 Delta-Style Diagnostics (Optional)

While Stylometric-Transfer is not an authorship attribution system, Delta-style distances can serve as *diagnostic* measures of stylistic proximity. Following standardization and Manhattan-style aggregation:

\[
\Delta(d,d') = \frac{1}{K}\sum_{k=1}^K \left|z_k(d) - z_k(d')\right|,
\]

where \(z_k\) is the z-transformed version of feature \(k\). Detailed decompositions and explanations of Delta variants motivate this lens. ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))

---

## 5. The Style Fingerprint as a Constraint Model

We treat the fingerprint as a set of weighted constraints:

\[
\mathcal{F} = \{(\psi_j, \mathcal{C}_j, w_j)\}_{j=1}^J,
\]

where:
- \(\psi_j(y)\) is a measurable statistic of output text (e.g., comma rate, histogram vector).
- \(\mathcal{C}_j\) is an admissible set (range, divergence tolerance, forbidden list).
- \(w_j\) is a weight (priority).

Typical constraint types:

1. **Range constraints**: \(\psi_j(y) \in [a,b]\)  
2. **Histogram constraints**: \(D(\mathbf{h}^*, \mathbf{h}(y)) \le \tau\)  
3. **Lexicon constraints**: forbidden phrases/words; preferred synonyms  
4. **Template constraints**: rhetorical move frequency bounds  

The JSON representation adds practical control fields such as `priority_order` and `strictness` to determine constraint precedence.

---

## 6. Constraint Satisfaction Decoding and Compliance Scoring

This section expands the mathematical view of rewriting as a **constraint satisfaction** problem.

### 6.1 Soft-Constrained Objective

Let \(p_\theta(y\mid x)\) be the LLM’s conditional probability of an output \(y\) given input \(x\). We define a soft-constrained objective:

\[
\max_{y \in \mathcal{Y}} \; \log p_\theta(y \mid x) - \lambda\, \mathcal{L}_{style}(y;\mathcal{F}) - \mu\,\mathcal{L}_{sem}(y;x),
\]

where:
- \(\mathcal{L}_{style}\) penalizes deviation from the fingerprint.
- \(\mathcal{L}_{sem}\) penalizes semantic drift (approximated conservatively via invariants; optionally via semantic similarity models).

A standard decomposition is:

\[
\mathcal{L}_{style}(y;\mathcal{F}) = \sum_{j=1}^J w_j\, \ell_j(\psi_j(y), \mathcal{C}_j).
\]

Example penalties:

**Range penalty** for \(\mathcal{C}_j=[a,b]\):

\[
\ell_j(v,[a,b]) = \big(\max(0,a-v)\big)^2 + \big(\max(0,v-b)\big)^2.
\]

**Histogram penalty** using KL divergence:

\[
\ell_j(\mathbf{h},\mathbf{h}^*) = D_{KL}(\mathbf{h}^*\|\mathbf{h}) = \sum_{b=1}^B h^*_b \log \frac{h^*_b}{\max(\epsilon,h_b)}.
\]

(For ordinal bins, Wasserstein distance \(W_1\) is often preferable; the implementation may adopt either.)

### 6.2 Hard Constraints (Feasibility)

Some constraints are best treated as hard feasibility requirements:

- Entity/number preservation constraints \(\Rightarrow\) must hold unless explicitly overridden.
- Hard forbidden lexicon constraints (e.g., “must not appear”).

Define the feasible set:

\[
\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y\in \mathcal{Y} : \forall j\in \mathcal{H},\; \psi_j(y)\in \mathcal{C}_j\},
\]

where \(\mathcal{H}\subseteq\{1,\dots,J\}\) indexes hard constraints.

Then decoding becomes:

\[
\max_{y\in \mathcal{Y}_{hard}(x,\mathcal{F})} \log p_\theta(y\mid x) - \lambda\sum_{j\notin \mathcal{H}} w_j\,\ell_j(\psi_j(y),\mathcal{C}_j) - \mu\,\mathcal{L}_{sem}(y;x).
\]

### 6.3 Practical Constraint-Satisfaction Decoding Procedure

In production LLM use, exact constrained decoding over \(\mathcal{Y}_{hard}\) is rarely available. Stylometric-Transfer approximates constraint satisfaction using **(i) instruction prompting**, **(ii) self-audit**, and **(iii) repair**.

A practical decoding approximation:

1. Generate a candidate rewrite \(y^{(0)}\) from the LLM under explicit instructions encoding \(\mathcal{F}\).
2. Compute local measurements \(\phi(y^{(t)})\) and audit constraint violations.
3. If violations exist, re-prompt the LLM with a structured report to obtain \(y^{(t+1)}\).
4. Stop when compliance exceeds a threshold or iteration limit.

### 6.4 Compliance Scoring

Define a normalized compliance score \(S(y;\mathcal{F})\in[0,1]\) aggregating constraint satisfaction:

\[
S(y;\mathcal{F}) = \sigma\Big(\sum_{j=1}^J w_j\, s_j(y)\Big), \quad \sum_j w_j = 1,
\]

where \(\sigma\) is a squashing function (e.g., identity clipped to \([0,1]\), or logistic), and \(s_j(y)\in[0,1]\) is a per-constraint score.

Examples:

- **Range score**:
\[
 s_j(y) = 1 - \min\left(1, \frac{\ell_j(\psi_j(y),[a,b])}{\kappa_j}\right)
\]
for a scaling constant \(\kappa_j>0\).

- **Histogram score** (KL):
\[
 s_j(y) = \exp\big(-\alpha_j\, D_{KL}(\mathbf{h}^*\|\mathbf{h}(y))\big).
\]

- **Lexicon hard constraint score**:
\[
 s_j(y)=\mathbf{1}[\text{no forbidden term appears}].
\]

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
- Evert, S., et al. “Understanding and explaining Delta measures for authorship attribution.” *Digital Scholarship in the Humanities* (2017). ([academic.oup.com](https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676?utm_source=chatgpt.com))
- Shen, T., Lei, T., Barzilay, R., & Jaakkola, T. “Style Transfer from Non-Parallel Text by Cross-Alignment.” (2017). ([arxiv.org](https://arxiv.org/abs/1705.09655?utm_source=chatgpt.com))
- Mukherjee, S., et al. “A Survey of Text Style Transfer: Applications and Ethical Implications.” (2024). ([arxiv.org](https://arxiv.org/abs/2407.16737?utm_source=chatgpt.com))
- Hu, Z., et al. “Text Style Transfer: A Review and Experimental Evaluation.” *KDD Explorations* (PDF). ([kdd.org](https://www.kdd.org/exploration_files/vol24issue1_2._Text_Style_Transfer__A_Review_and_Experimental_Evaluation.pdf?utm_source=chatgpt.com))

---

## Appendix A. Methods (Pseudocode)

This appendix provides pseudocode for the **fingerprinter** (extractor) and **rewriter** stages.

### A.1 Fingerprint Extraction (Corpus → Style Fingerprint JSON)

**Inputs:** corpus archive \(A\), LLM \(\mathcal{R}_\theta\), schema template \(S\)  
**Output:** style fingerprint \(\mathcal{F}\) (JSON)

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

**Inputs:** fingerprint \(\mathcal{F}\), input Markdown \(x\), LLM \(\mathcal{R}_\theta\)  
**Output:** rewritten Markdown \(y\) and deviations report

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

---

## Appendix B. Formal Constrained Decoding Framing

This appendix tightens the decoding formulation into a standard constrained optimization / constrained MDP view.

### B.1 Constrained Maximum A Posteriori Decoding

Let \(p_\theta(y\mid x)\) denote the base LLM distribution. Let constraints be indexed by \(j=1,\dots,J\) with statistics \(\psi_j(y)\) and admissible sets \(\mathcal{C}_j\).

We define the feasible set of hard constraints:

\[
\mathcal{Y}_{hard}(x,\mathcal{F}) = \{y : \forall j \in \mathcal{H},\; \psi_j(y) \in \mathcal{C}_j\}
\]

The constrained MAP problem is:

\[
\hat y = \arg\max_{y \in \mathcal{Y}_{hard}(x,\mathcal{F})} \; \log p_\theta(y\mid x)
\]

In practice, \(\mathcal{Y}_{hard}\) is not explicitly enumerable. We therefore relax the problem using a **Lagrangian penalty formulation**:

\[
\hat y = \arg\max_{y \in \mathcal{Y}} \; \log p_\theta(y\mid x)
- \sum_{j=1}^J \lambda_j \cdot g_j(\psi_j(y))
- \mu \cdot \mathcal{L}_{sem}(y;x),
\]

where:

- \(g_j(\cdot)\) is a non-negative violation function such that \(g_j(v)=0\) iff \(v \in \mathcal{C}_j\)
- \(\lambda_j \ge 0\) are Lagrange multipliers derived from `validators.weights`
- \(\mathcal{L}_{sem}\) enforces meaning preservation

This matches the standard **soft-constrained decoding** paradigm used in controllable generation and lexically constrained decoding.

---

### B.2 Projection View

Equivalently, rewriting can be interpreted as projection of an unconstrained sample \(y^{(0)} \sim p_\theta(\cdot \mid x)\) onto the admissible region:

\[
\hat y = \Pi_{\mathcal{C}}(y^{(0)}) = \arg\min_{y} \; d(y, y^{(0)}) + \sum_j \lambda_j g_j(\psi_j(y)),
\]

where \(d(\cdot,\cdot)\) is an edit or semantic divergence.  
In practice, \(\Pi_{\mathcal{C}}\) is approximated by **LLM self-repair passes** guided by explicit audit reports.

---

### B.3 Constrained Markov Decision Process (CMDP) Interpretation

Token generation may be framed as a CMDP:

- States: \(s_t = y_{1:t}\)  
- Actions: \(a_t = y_{t+1}\)  
- Reward: \(r_t = \log p_\theta(a_t\mid s_t,x)\)  
- Costs: \(c_{j,t}\) accumulating toward \(\psi_j(y)\)

with terminal constraints:

\[
\mathbb{E}\Big[ \sum_t c_{j,t} \Big] \le \tau_j
\]

This clarifies that the system approximates **policy optimization under global style budgets**, implemented via instruction-guided generation and post-hoc repair.

---

## Appendix C. Evaluation and Acceptance Criteria

This appendix defines concrete divergence metrics and acceptance thresholds mapped directly to the fingerprint JSON fields.

### C.1 Metric Families

#### (1) Rate Constraints (scalar)

For a target interval \([a,b]\) and observed value \(v\):

\[
\text{viol}_r(v) = \max(0,a-v) + \max(0,v-b)
\]

Score:

\[
s_r(v) = \exp(-\alpha_r \cdot \text{viol}_r(v))
\]

Mapped JSON paths:
- `/targets/punctuation/comma_density_per_100w`
- `/targets/orthography/contractions_rate`

---

#### (2) Histogram Constraints (sentence / paragraph)

Primary metric: **Wasserstein-1 distance**

\[
W_1(\mathbf{h}^*, \mathbf{h}) =
\sum_{b=1}^{B-1} \left| \sum_{k=1}^b (h_k - h_k^*) \right|
\]

Secondary diagnostic: KL divergence

\[
D_{KL}(\mathbf{h}^* \| \mathbf{h}) =
\sum_b h_b^* \log \frac{h_b^*}{\max(\epsilon,h_b)}
\]

Score:

\[
s_h = \exp(-\alpha_h W_1)
\]

Mapped JSON:
- `/targets/sentence/length_words/distribution`
- `/targets/paragraph/length_sentences`

---

#### (3) Lexicon Constraints

Hard:

\[
s_{lex}^{hard} = \mathbf{1}[\text{no forbidden term appears}]
\]

Soft:

\[
s_{lex}^{soft} = \exp(-\alpha_{lex} \cdot |f_y - f^*|)
\]

Mapped JSON:
- `/lexicon/avoid_words`
- `/lexicon/avoid_phrases`
- `/lexicon/preferred_phrases`

---

### C.2 Aggregated Compliance Score

Let weights \(w_j\) come from `validators.weights` with \(\sum_j w_j = 1\).

\[
S(y;\mathcal{F}) = \sum_{j=1}^J w_j s_j(y)
\]

Acceptance levels:

| Level | Condition |
|------|-----------|
| **Pass** | \(S \ge 0.75\) and no hard violations |
| **Warn** | \(0.60 \le S < 0.75\) |
| **Fail** | \(S < 0.60\) or any hard violation |

Mapped JSON:
- `/validators/scoring/overall_threshold/pass`
- `/validators/scoring/overall_threshold/warn`

---

### C.3 Field-Level Thresholds (Defaults)

| Field family | Metric | Threshold |
|-------------|--------|-----------|
| Sentence histogram | \(W_1\) | \(\le 0.08\) |
| Paragraph histogram | \(W_1\) | \(\le 0.10\) |
| Punctuation rates | relative error | \(\le 20\%\) |
| One-sentence paras | abs diff | \(\le 0.05\) |
| Exclamations | hard max | must satisfy |
| Forbidden lexicon | indicator | must satisfy |

---

### C.4 Iterative Repair Stopping Rule

Let \(S_t\) be the score at iteration \(t\). Stop when:

\[
S_t \ge S_{pass} \quad \text{and} \quad H_t = 0
\]

Else continue up to \(T_{max}\) repair passes.
