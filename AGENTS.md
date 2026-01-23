# AGENTS.md

> Guidance for automated coding agents working on `stylometric-transfer`

This document describes the **purpose, architecture, conventions, constraints, and development roadmap** for the `stylometric-transfer` project so that an automated agent can reliably continue development without prior conversation context.

---

## 1. Project Mission

`stylometric-transfer` implements:

- **Stylometric profiling** — extracting an explicit, interpretable style model from an author’s writing corpus
- **Author-conditioned style transfer** — rewriting text to match that style while preserving meaning

The defining design principle is:

> **Explicit, interpretable, versionable style models** rather than fine‑tuning or opaque embeddings.

The system must:
- Remain inspectable by humans
- Keep the style model as JSON
- Avoid fine‑tuning or hidden representations
- Preserve meaning strictly in rewriting

---

## 2. Conceptual Terminology (Canonical)

Use the following terms consistently in code, docs, and comments:

- **Stylometry** — quantitative analysis of writing style
- **Stylometric profile / style fingerprint** — the JSON artifact
- **Style transfer** — rewriting while preserving semantics
- **Author‑conditioned generation** — generation guided by fingerprint
- **Explicit style model** — rule‑ and feature‑based JSON

Avoid ambiguous terms like “clone” in public documentation.

---

## 3. Current Architecture

### Files

- `fingerprint_style.py`
  - Input: compressed corpus archive (`.zip`, `.tar*`)
  - Output: `style_fingerprint.json`
  - Responsibilities:
    - Extract archive
    - Read text files
    - Compute stylometric measurements locally
    - Select representative excerpts
    - Call LLM to synthesise fingerprint JSON
    - Repair invalid JSON if necessary
  - CLI short flags: `-c` (config, optional; defaults to `./config.llm.json` if present, else next to script), `-a` (archive), `-o` (out), `-v` (verbose)
  - Defaults: if `--profile-id` or `--author-name` are omitted, both default to the output filename without the `.json` extension

- `apply_fingerprint.py`
  - Input: fingerprint JSON + Markdown file
  - Output: rewritten Markdown + deviations report
  - Responsibilities:
    - Measure input text
    - Call LLM with fingerprint + measurements
    - Enforce preservation of meaning
    - Return rewritten text and deviations
  - CLI short flags: `-c` (config, optional; defaults to `./config.llm.json` if present, else next to script), `-f` (fingerprint; adds `.json` if missing), `-i` (input), `-o` (out), `-v` (verbose)

- `scripts/fingerprint_style.sh` and `scripts/apply_fingerprint.sh`
  - Bash wrappers around the Python entry points
  - Pass all CLI args through unchanged

- `config.llm.json`
  - Stores API configuration
  - OpenAI‑compatible

### Data Flow

```
Corpus → Local stats → LLM synthesis → Fingerprint JSON
Fingerprint + Draft → Local stats → LLM rewrite → Styled Markdown
```

---

## 4. Design Constraints (Critical)

These must always be preserved.

### 4.1 Explicit Models Only

- The style model MUST remain:
  - JSON
  - Human readable
  - Editable
  - Versionable

Do NOT introduce:
- Fine‑tuning
- Embedding‑only representations
- Hidden latent style vectors without interpretation

### 4.2 Meaning Preservation

In rewriting:
- No new facts
- No new claims
- No new examples
- No semantic drift

Deviation reporting is mandatory when conflicts occur.

### 4.3 Interpretable Metrics

Local measurements must:
- Remain simple and explainable
- Use distributions over single averages where possible
- Prefer sentence/paragraph/lexical statistics

Avoid black‑box scoring models unless clearly labeled optional.

---

## 5. Style Fingerprint Schema (High‑Level)

The fingerprint JSON contains these top‑level keys:

- `schema_version`
- `profile_id`
- `metadata`
- `measurements` (verbatim local statistics)
- `targets` (style constraints)
- `lexicon` (preferred / avoided words & phrases)
- `templates` (syntactic + rhetorical moves)
- `controls` (priority + strictness + rewrite policy)
- `validators` (scoring weights + checks)
- `derived_instructions` (compiled prompts)

The schema is **extensible**, but backward compatibility should be preserved when possible.

Note: `metadata.corpus` includes `document_count` and `documents` (per-document metadata list, including title when available).

---

## 6. Prompting Strategy (Canonical)

### 6.1 Fingerprint Construction

Process:
1. Compute local measurements
2. Select representative excerpts
3. Send both to LLM with:
   - Schema hint
   - JSON‑only requirement
   - Controlled vocabulary instructions

LLM must:
- Embed measurements verbatim
- Infer stylistic targets from measurements + excerpts
- Produce derived instructions for reuse

### 6.2 Application / Rewriting

Prompt must:
- Include full fingerprint JSON
- Include local measurements of input
- Enforce:
  - Preservation of meaning
  - Markdown validity
  - Priority order in constraints

LLM output format is JSON:

```
{
  "final_markdown": "...",
  "deviations": [...],
  "self_check": {...}
}
```

---

## 7. Coding Conventions

### 7.1 API Client

- Use OpenAI‑compatible REST calls
- Prefer `/v1/chat/completions` unless migrated
- Centralize retry/backoff logic

Recommended improvements (future work):
- Exponential backoff for 429 / 5xx
- Streaming optional flag

### 7.2 Error Handling

Always handle:
- Invalid JSON output
- Partial JSON
- Code‑fenced JSON

Current strategy:
- Attempt strict parse
- If failure, call LLM repair mode

Preserve this pattern.

---

## 8. Measurement Layer (Do Not Over‑Engineer)

Current measurements include:

- Sentence length histogram
- Paragraph length histogram
- One‑sentence paragraph rate
- Punctuation rates per 1000 words
- Contraction rate
- Dash / ellipsis counts
- Frequent bigrams / trigrams

Guidelines:

- Prefer approximate signals over fragile exact metrics
- Avoid heavy NLP pipelines unless optional
- Do NOT require spaCy / transformers by default

Future additions allowed:
- POS tag distributions (optional)
- Readability indices
- Function word profiles

---

## 9. Ethics & Safeguards

Always preserve:

- Intended use: personal writing, self‑modeling, editing assistance
- Avoid impersonation of living authors

Controls in fingerprint:

- `do_not_imitate_living_author = true`
- `copyright_sensitive = true`

Agents must NOT add features that:
- Automate impersonation
- Mask authorship
- Remove deviation reporting

---

## 10. Roadmap (Agent Guidance)

High‑value next steps, in priority order:

### Phase 1 — Reliability

- [ ] Add retry + exponential backoff wrapper for API calls  
- [ ] Add request timeout + cancellation handling  
- [ ] Add logging verbosity flag  

### Phase 2 — Validation

- [ ] Add JSON Schema file (`schema/style_fingerprint.schema.json`)  
- [ ] Validate fingerprint outputs with `jsonschema`  
- [ ] Emit warnings on missing or null critical fields  

### Phase 3 — Scoring & Feedback

- [ ] Add post‑rewrite scoring against fingerprint  
- [ ] Compute divergence metrics (sentence length, punctuation)  
- [ ] Emit compliance score (0–1)  

### Phase 4 — Tooling

- [ ] Batch rewrite mode  
- [ ] Batch fingerprint mode  
- [ ] Diff visualizer (original vs styled)  
- [ ] CLI subcommands (`stylometric fingerprint`, `stylometric apply`)  

### Phase 5 — Generation

- [ ] Add `generate_from_fingerprint.py`  
- [ ] Support outlines / prompts conditioned on fingerprint  

---

## 11. Testing Strategy

Current state: no formal tests

Recommended additions:

- Golden corpus test:
  - Run fingerprint on fixed small corpus
  - Verify stable JSON fields

- Rewrite invariants:
  - Word count within tolerance
  - Named entities preserved
  - No added sentences unless allowed

- JSON validity tests

---

## 12. Naming & Public Interface

Preferred public names:

- Project: `stylometric-transfer`
- Artifact: **Style Fingerprint** or **Stylometric Profile**
- Process: **Stylometric profiling** + **Author style transfer**

Avoid marketing terms like:
- “Magic”
- “Clone”
- “Deep copy”

---

## 13. Summary for Agents

If you are an automated agent continuing this project:

You are building an **interpretable stylometric profiling and style‑transfer system** with these invariants:

- Explicit JSON style models
- Local statistical grounding
- LLM used only for synthesis and rewriting
- Meaning preservation is sacred
- Deviations must be reported
- Human inspectability is a core feature

Any new feature should improve:

- Reliability
- Interpretability
- Reproducibility
- Control

—not raw imitation fidelity alone.

---

*AGENTS.md — operational context for stylometric-transfer*
