# AGENTS.md - MCP-Rust

Operational context for automated agents working on the Rust MCP server in
`MCP-Rust/`. Read this before changing code.

---

## 1. Purpose

`stylometric-mcp` is a self-contained Rust re-implementation of the
`stylometric-transfer` pipeline, served over MCP (stdio and minimal HTTP) so agents
can call it as a tool.

Defining constraint: **no runtime dependency on the Python tree.** Everything the
server needs is either compiled in via `include_str!` or supplied by the caller.

---

## 2. Hard rules

1. **Never read from the parent Python tree at runtime.** No `../prompts.json`, no
   `../config.tunables.json`, no `../general-guidelines.md`. If you need new data,
   copy it into `MCP-Rust/resources/` and embed it in `src/resources.rs`.
2. **Never bake build-machine paths into the binary.** No `env!("CARGO_MANIFEST_DIR")`
   in runtime path resolution. Resolve from `current_dir`, `current_exe`, env vars,
   or an explicit argument.
3. **No silent fallbacks.** If the LLM fails or returns unparseable output, return
   an error. The deterministic scaffold is opt-in via `allow_scaffold` /
   `--scaffold` and must log a warning when used.
4. **The style model stays explicit JSON.** No embeddings, no fine-tuning, no opaque
   style vectors.
5. **Meaning preservation is sacred.** Rewrites may not add facts, claims, or
   examples. Report `deviations` instead of guessing.
6. **Report only what you verified.** `cargo check` passing is not functional
   correctness.

---

## 3. Module map

| File | Lines | Responsibility |
| --- | --- | --- |
| `src/main.rs` | 263 | CLI parsing, config resolution, `fingerprint` / `apply` subcommands |
| `src/mcp.rs` | 412 | JSON-RPC dispatch, tool catalogue, stdio and HTTP transports |
| `src/fingerprint.rs` | 262 | corpus -> voice filter -> measurements -> prompt -> fingerprint |
| `src/apply.rs` | 256 | mask -> chunk -> rewrite loop -> post-process -> score |
| `src/measurements.rs` | 624 | deterministic measurement bundle |
| `src/scoring.rs` | 781 | compliance, JSD/cosine/Jaccard, calibrated probability, similarity |
| `src/filters.rs` | 564 | non-voice masking, base64 images, citations, fiction detection |
| `src/postprocess.rs` | 449 | humanizer rules, guarded spelling, emoji/quotes/dashes |
| `src/spelling.rs` | 629 | locale spelling maps, noun/verb disambiguation |
| `src/headings.rs` | 530 | heading detection, case normalization, qualifier cleanup |
| `src/text.rs` | 196 | tokenization, sentence/paragraph splitting, stats helpers |
| `src/llm.rs` | 252 | OpenAI-compatible client, retry/backoff, lenient JSON parse |
| `src/resources.rs` | 118 | embedded data plus accessors |
| `src/store.rs` | 89 | GUID-addressed fingerprint store |

`resources/` holds embedded data. `fingerprint_store/` is runtime state, not source.

---

## 4. Build and verify

```bash
cargo check --offline
cargo build --release --offline

./target/release/stylometric-mcp --help
./target/release/stylometric-mcp tools

printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | ./target/release/stylometric-mcp serve
```

Self-containment checks (run before claiming a change is safe):

```bash
grep -rn 'CARGO_MANIFEST_DIR' src/ || echo OK
grep -rn 'include_str!' src/resources.rs
strings target/release/stylometric-mcp | grep -c '/workspace/Projects/stylometric-transfer'   # expect 0
```

Isolation test: copy only the binary into an empty directory, run `initialize`,
`tools/list`, and a `rate` call, then confirm no files were created and stderr is
empty.

Functional test without an API key: `fingerprint ... --scaffold` must exit 0 with a
warning; the same command without `--scaffold` must exit 1 with the LLM error.

---

## 5. Common change recipes

**Add or change a tunable.** Edit `resources/config.tunables.json` and read it via
`resources::tunable_f64(&["section", "key"], default)` or `tunable_u64`. Do not
hardcode a new default in Rust without also adding it to the JSON.

**Add a tool.** Add an entry in `tool_definitions()` and a match arm in
`call_tool()` in `src/mcp.rs`. Keep the handler thin: parse arguments, call a
library function, return JSON. `Err(String)` becomes `isError: true`.

**Add a CLI flag.** Parse it in the relevant `run_*` function in `src/main.rs`,
then update `USAGE`. Flags are scanned by hand; there is no clap.

**Add embedded data.** Drop the file in `resources/`, add an `include_str!` const
and (if JSON) a `Lazy<Value>` in `src/resources.rs`, and reference it only through
that module.

**Change scoring.** `compute_style_compliance` averages per-section scores and
weights them by `validators.weights` when present, otherwise unweighted. Section
names in `scoring.rs` must match measurement keys in `measurements.rs`.

---

## 6. Pitfalls

- Retry counts are **additional attempts** after the first (`max_retries = 2`
  means up to 3 attempts).
- `apply` masks content (base64 images, non-voice blocks, citations, quotes) into
  a `BTreeMap` of placeholders. Any new masking pass must add to that same map and
  be restored after rewriting.
- Chunking is paragraph-based and token-budgeted (`chunking.max_input_tokens`,
  floor 512). Do not assume sentence-level splitting.
- Measurement keys are a compatibility surface: prompts and the Python scorer
  expect the same shapes. Renaming a key silently degrades scoring.
- `validators.weights` is often absent from LLM output; unweighted averaging is
  the intended graceful path, not a bug.
- Two embedded lists (`config.common_words.txt`, `config.entity_blacklist.txt`)
  are compiled in but not yet consumed; they are annotated `#[allow(dead_code)]`.

---

## 7. Known gaps / next steps

1. Port prompt chunking and fingerprint merge for large corpora.
2. Port rare-word and entity-blacklist filtering into `measurements.rs`.
3. Port rolling chunk summaries, perplexity presets, and roster rotation.
4. Add unit tests for `measurements`, `scoring`, `postprocess`, `spelling`, and
   `filters`. Currently there are none; validation is CLI and JSON-RPC smoke
   testing only.
5. Consider a `validate` subcommand that checks a fingerprint against the
   semantics of `style_fingerprint_schema.json` (not embedded yet).
6. Clear the remaining dead-code warnings by wiring helpers up or deleting them.

---

## 8. Non-goals

- Impersonating living authors. `controls.do_not_imitate_living_author` and
  `controls.copyright_sensitive` must stay `true`.
- Production HTTP hardening (auth, TLS, rate limiting).
- Batch or bulk modes; they are not designed yet.


