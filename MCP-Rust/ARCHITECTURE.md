# Architecture - MCP-Rust

How `stylometric-mcp` is structured, and why.

---

## 1. Goals

| Goal | Consequence |
| --- | --- |
| Zero runtime dependency on Python | all config and prompt data embedded via `include_str!` |
| Relocatable binary | no compile-time paths; runtime resolution only |
| Interpretable output | explicit JSON fingerprints, named measurement keys |
| Agent-usable | MCP over stdio, plus a minimal HTTP transport |
| Deterministic where possible | statistics, spelling, post-processing in Rust; the LLM only synthesizes |

---

## 2. Layers

```
Transport        mcp.rs          stdio (JSON-RPC lines) | HTTP/1.1
Entry points     main.rs         CLI subcommands
Orchestration    fingerprint.rs (profiling)   apply.rs (rewriting)
Domain           measurements  scoring  postprocess  spelling
                 filters  headings  text
Infrastructure   llm.rs (HTTP client)   store.rs   resources.rs
```

The domain layer is pure: it takes `&str` / `&Value` and returns `Value` or
`String`. Only `llm.rs`, `store.rs`, and `main.rs` touch the filesystem or network.

---

## 3. Fingerprint flow

```
documents -> join -> fiction detection -> filter_author_voice
                                              |
                                              v
                                     compute_measurements
                                              |
                     representative_excerpt <-+-> per-doc strided sampling
                                              |
                                              v
                        prompts.json[fingerprint] + schema_hint
                                              |
                                              v
                     chat_completions -> parse_json_lenient
                                              |
                                              v
                   measurements force-overwritten; schema_version,
                   profile_id, metadata.author_name defaulted;
                   tunables_snapshot embedded
                                              |
                                              v
                                       fingerprint JSON
```

Key decisions:

- **Measurements are authoritative.** The model's `measurements` field is
  overwritten with the locally computed bundle, so a hallucinating model cannot
  fabricate statistics.
- **Excerpts are strided, not head-sampled**, so the sample spans the document
  instead of over-weighting the opening (`stride = sentence_count / 24`).
- **Scaffold is opt-in.** On LLM failure the default path returns `Err`; with
  `allow_scaffold` it emits a deterministic fingerprint containing real
  measurements, empty `targets`/`validators`, and the global avoid list.

---

## 4. Apply flow

```
markdown -> strip base64 images -> mask non-voice blocks -> mask citations
         -> mask quoted passages -> BTreeMap<placeholder, original>
                                   |
                                   v
                     chunk_text(masked, max_input_tokens)
                                   |
                     per chunk:    |
              compute_measurements(chunk)
                                   v
              prompts.json[apply] + fingerprint
                                   v
              chat_completions -> parse_json_lenient -> extract_markdown
                                   v
              compute_style_compliance -> score
                                   v
         score >= threshold or retries exhausted
                                   v
              keep best-scoring attempt per chunk
                                   v
         restore_placeholders -> apply_mandatory_humanizer
                                   v
         final compliance + calibrated probability
```

Notes:

- The **best-scoring attempt per chunk** is kept, not the last one.
- Masks merge into one `BTreeMap`, so restore order is deterministic and
  duplicate content is deduplicated naturally.
- `apply` returns `final_markdown`, `deviations`, `postprocess_changes`,
  `compliance`, `style_match_probability`, and `chunks`.
- Post-processing runs on restored text, and final compliance is recomputed
  afterwards so the reported score reflects what was actually written.

---

## 5. Measurement bundle

`compute_measurements` returns 15 top-level sections, including:

- `sentence` and `paragraph` length histograms (5 bins each)
- `punctuation` rates per 1000 words
- `orthography_signals` (US vs Canadian/British heuristic)
- function-word profile
- stance signals (hedging, boosting, pronouns)
- sentence-opener and transition templates
- rhetorical move signals
- paragraph cadence (opening and closing sentence length)
- `epistemic_profile`, `syntax_texture`
- `lexical_avoidance` category rates
- `repetition` (bigram and trigram repeat rates)
- `comma_density_per_100w`

All values are rates, means, standard deviations, or histograms. There are no
black-box models: every number can be explained to a user and reproduced.

---

## 6. Scoring

`scoring.rs` provides:

- `compute_style_compliance` - per-metric agreement, per-section average, then a
  weighted aggregate using `validators.weights` (unweighted when absent).
- `jensen_shannon_divergence`, `cosine_similarity`, `jaccard_similarity`,
  `l1_distance`, `relative_diff` - interpretable comparison primitives.
- `calibrated_style_match_probability` - maps score plus text length to a 0-1
  probability.
- `compute_fingerprint_similarity` - weighted component comparison of two
  fingerprints with per-component diagnostics.

Metric paths in `scoring.rs` are string arrays that mirror the measurement key
tree. That coupling is the main compatibility hazard when editing either file.

---

## 7. Resources and prompts

`resources.rs` embeds seven files and exposes:

- `prompt_group(name)` and `prompt(group, field)` - prompt templates
- `tunables()`, `tunable(path)`, `tunable_f64`, `tunable_u64` - runtime policy
- `avoid_entries()` and `avoid_json()` - global avoid list
- `common_words()` and `entity_blacklist()` - embedded, currently unused
- `LLM_CONFIG_JSON` - default connection config

Prompts live in externalized JSON rather than Rust string literals so they can be
diffed against the Python originals and updated without touching code.

---

## 8. Transports

**stdio** reads lines from stdin; each line is one JSON-RPC message. Requests
without an `id`, and any method starting with `notifications/`, produce no
response.

**HTTP** is a hand-rolled HTTP/1.1 listener on a `TcpListener` with a
single-threaded accept loop. It is a local convenience surface, not a production
API: no TLS, no auth, no concurrency pool.

Both call the same `handle_request()`, so protocol behavior cannot drift between
transports. Unknown methods return `-32601`.

---

## 9. State

The server is stateless apart from the fingerprint store:

- `store::save` writes `<guid>.fingerprint.json` and `<guid>.meta.json`
- `store::load` validates the GUID before touching the filesystem
- Directory resolution: explicit base, then `./fingerprint_store`, then
  `<exe-dir>/fingerprint_store`, then the system temp directory

No global mutable state, no per-request caches, no background threads.

---

## 10. Error handling

- `parse_json_lenient` tries a strict parse, then a fenced code block, then the
  outermost `{...}` span.
- Retry and backoff apply to 429, 5xx, and transport errors only.
- Other 4xx return immediately with a truncated body, so auth failures are
  obvious rather than silently retried.
- Tool failures become `isError: true` with the message; protocol errors use
  standard JSON-RPC codes.

---

## 11. Divergence from the Python implementation

| Area | Python | MCP-Rust |
| --- | --- | --- |
| Prompt chunking + merge | yes | not yet |
| Rolling chunk summaries | yes | not yet |
| Perplexity presets / roster rotation | yes | not yet |
| `general-guidelines.md` LLM parsing | yes | not yet |
| Rare-word / entity-blacklist filtering | yes | embedded, not consumed |
| Prompt storage | external `prompts.json` | embedded copy |
| Transports | HTTP API + CLI | MCP stdio + minimal HTTP |
| Measurement / scoring | reference implementation | ported, approximately equivalent |

Scores are therefore comparable but not bit-identical to the Python
implementation.


