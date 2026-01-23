# stylometric-transfer

> **Stylometric profiling + controllable author-style transfer for personal writing**

`stylometric-transfer` builds an explicit, interpretable **stylometric style profile** from an author’s corpus and then applies that profile to rewrite or generate new text in the same voice.

This repository provides:
- `fingerprint_style.py` — extract a **style fingerprint (stylometric profile)** from an archive of writing
- `apply_fingerprint.py` — rewrite Markdown to match that fingerprint
- `scripts/` — bash wrappers for invoking the Python entry points

Unlike fine‑tuning or opaque embeddings, this system uses an **explicit, versionable JSON style model** that you can inspect, edit, audit, and reuse.

---

## Table of Contents

- [Overview](#overview)
- [Concepts & Terminology](#concepts--terminology)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Build a Style Fingerprint](#1-build-a-style-fingerprint)
  - [2. Apply a Fingerprint](#2-apply-a-fingerprint)
- [Output Files](#output-files)
- [Style Model Schema](#style-model-schema)
- [Ethics & Intended Use](#ethics--intended-use)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

This project implements a two‑stage pipeline:

1. **Stylometric profiling**  
   Quantitatively analyze an author’s corpus and construct a structured **style fingerprint** (JSON)

2. **Author‑conditioned style transfer**  
   Rewrite new text so it conforms to that fingerprint while preserving meaning

The system combines:
- Local statistical measurement (sentence length, punctuation rates, paragraph structure, etc.)
- LLM‑based synthesis into an explicit style model
- Constraint‑driven rewriting using that model

This is a practical implementation of:

> **Stylometric profiling + controlled author-style transfer**

---

## Concepts & Terminology

| Term | Meaning |
|------|---------|
| **Stylometry** | Quantitative analysis of writing style |
| **Stylometric profile** | Feature‑based representation of an author’s style |
| **Style fingerprint** | Explicit JSON encoding of stylistic constraints |
| **Style transfer** | Rewriting text while preserving meaning but altering style |
| **Author‑conditioned generation** | Text generation guided by an author profile |

In research terms, this system implements:
- Feature‑based stylometric profiling
- Interpretable controllable text generation
- Constraint‑augmented author style transfer

---

## Features

- Accepts `.zip` / `.tar*` archives of writing corpora
- Reads `.txt`, `.md`, `.rst`, `.html`, `.docx` (via `python-docx`)
- Computes statistical measurements locally:
  - Sentence length distributions
  - Paragraph structure
  - Punctuation rates
  - Contraction and dash usage
  - Common n‑grams
- Produces a **comprehensive JSON style profile**
- Rewrites Markdown with:
  - Meaning preservation
  - Structural fidelity
  - Deviation reporting
- OpenAI‑compatible (works with OpenAI, Azure OpenAI, vLLM, etc.)
- Interpretable, editable, versionable style models

---

## Architecture

```
corpus.tar.gz
      │
      ▼
[fingerprint_style.py]
      │
      ├─ local statistical analysis
      ├─ representative excerpts
      └─ LLM synthesis
      │
      ▼
style_fingerprint.json
      │
      ▼
[apply_fingerprint.py]
      │
      ├─ local measurement of input
      ├─ constraint‑driven rewriting
      └─ deviation audit
      │
      ▼
rewritten_text.styled.md
```

---

## Installation

### Requirements

- Python 3.9+
- `requests`
- `python-docx` (optional, for `.docx` corpora)

```bash
pip install requests python-docx
```

---

## Configuration

Create `config.llm.json` in the project root (used by default):

```json
{
  "api_key": "YOUR_OPENAI_KEY",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4.1-mini",
  "max_tokens": 6000,
  "temperature": 0.2,
  "timeout_seconds": 120
}
```

Notes:
- `base_url` should be the API root (no `/chat/completions`)
- Any OpenAI‑compatible endpoint can be used
- Lower temperature is recommended for consistency

---

## Usage

### 1. Build a Style Fingerprint

Create a compressed archive of your writing corpus:

```bash
tar -czf my_corpus.tar.gz essays/ notes/ drafts/
```

Run fingerprinting:

```bash
python fingerprint_style.py \
  -a my_corpus.tar.gz \
  -o my_fingerprint.json \
  --profile-id "me_style_v1" \
  --author-name "Me"
```

Or use the wrapper script:

```bash
./scripts/fingerprint_style.sh \
  -a my_corpus.tar.gz \
  -o my_fingerprint.json \
  --profile-id "me_style_v1" \
  --author-name "Me"
```

Pass `-c/--config` to use a non-default config path. If `--profile-id` or `--author-name` are omitted, they default to the output filename without the `.json` extension (e.g., `my_fingerprint`).

This will:
- Extract the archive
- Measure stylistic statistics
- Send measurements + excerpts to the LLM
- Produce `my_fingerprint.json`

---

### 2. Apply a Fingerprint

Rewrite a Markdown file in your style:

```bash
python apply_fingerprint.py \
  -f my_fingerprint.json \
  -i draft.md
```

Or use the wrapper script:

```bash
./scripts/apply_fingerprint.sh \
  -f my_fingerprint.json \
  -i draft.md
```

Pass `-c/--config` to use a non-default config path.

Outputs:

- `draft.md.styled.md` — rewritten text
- `draft.md.styled.md.deviations.json` — any rule conflicts or deviations

---

## Output Files

### Style Fingerprint (`*.json`)

Contains:

- `metadata` — corpus and extraction info  
- `measurements` — raw statistical signals  
- `targets` — stylistic constraints and distributions  
- `lexicon` — preferred / avoided words and phrases  
- `templates` — syntactic and rhetorical patterns  
- `controls` — strictness and priority ordering  
- `validators` — scoring weights and checks  
- `derived_instructions` — compiled prompts for generation and rewriting

This file is:
- Human readable
- Editable
- Version controllable
- Reusable across projects

---

## Style Model Schema

The schema models multiple layers:

- Orthography & formatting  
- Punctuation signature  
- Sentence rhythm & clause structure  
- Paragraph architecture  
- Lexical preferences  
- Semantic tendencies  
- Rhetorical moves  
- Persona & stance  

It supports:
- Target values with tolerance ranges  
- Histograms instead of single averages  
- Hard vs soft constraints  
- Priority‑ordered enforcement  

This enables **interpretable controllable generation** rather than opaque imitation.

---

## Ethics & Intended Use

This project is designed for:

- Personal writing consistency  
- Author self‑modeling  
- Editing assistance  
- Long‑term voice preservation  

It is **not intended** for:

- Impersonating living authors without consent  
- Passing generated text off as another person  
- Circumventing authorship or attribution

Recommended practice:

- Use only your own writing or licensed/public‑domain corpora
- Set `do_not_imitate_living_author = true` in controls
- Clearly label AI‑assisted outputs when appropriate

---

## Roadmap

Planned extensions:

- [ ] JSON Schema validation (`jsonschema`)  
- [ ] HTML / PDF corpus ingestion  
- [ ] Streaming + retry/backoff support  
- [ ] Style similarity scoring CLI  
- [ ] Batch rewriting  
- [ ] Fine‑grained constraint toggles  
- [ ] Visualization of stylistic distributions  

---

## References

Core research areas:

- Stylometry / computational stylistics  
- Authorship attribution  
- Text style transfer  
- Interpretable controllable generation  

Representative terms:

- *Stylometric profiling*  
- *Author‑conditioned text generation*  
- *Feature‑based style modeling*  
- *Constraint‑augmented rewriting*

---

## License

MIT (recommended) or your preferred license.

---

## Acknowledgments

Inspired by:
- Stylometric authorship research
- Controllable text generation literature
- Practical needs for long‑term personal voice consistency

---

*stylometric-transfer — explicit style models for interpretable author voice transfer*
