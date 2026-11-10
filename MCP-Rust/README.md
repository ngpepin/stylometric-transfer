# stylometric-mcp (MCP-Rust)

A self-contained, high-performance Rust implementation of the `stylometric-transfer`
pipeline, exposed as a **Model Context Protocol (MCP)** server for agents such as
OpenClaw and Hermes.

It builds **explicit, interpretable, versionable JSON style fingerprints** from an
author corpus and rewrites Markdown to match that fingerprint while preserving
meaning.

> **No runtime dependency on the Python implementation.** Prompts, tunables,
> spelling rules, word lists, and the default LLM config are compiled into the
> binary with `include_str!`. The binary can be copied anywhere and run alone.

---

## Design invariants

1. **Explicit models only** — the style model is JSON: human-readable, editable,
   diffable. No fine-tuning, no embedding-only style vectors.
2. **Local statistical grounding** — measurements are computed deterministically in
   Rust; the LLM only synthesizes and rewrites.
3. **Meaning preservation** — rewriting may not add facts, claims, or examples;
   conflicts are reported as `deviations`.
4. **Loud failures** — LLM or parse failures return errors. The deterministic
   scaffold is opt-in (`--scaffold` / `"allow_scaffold": true`), never silent.
5. **Inspectability** — every score is traceable to named measurements and deltas.

---

## Build

```bash
cd MCP-Rust
cargo build --release --offline      # target/release/stylometric-mcp
```

Requires a Rust toolchain (edition 2021). Release profile: `opt-level = 3`, thin
LTO, `codegen-units = 1`, `strip = true`.

Artifact: ~4.6 MB stripped binary, 5,425 lines across 14 modules, 12 direct
dependencies (`serde`, `serde_json`, `regex`, `ureq`, `zip`, `flate2`, `tar`,
`uuid`, `chrono`, `rand`, `sha2`, `once_cell`).

---

## CLI

```
stylometric-mcp serve [--config PATH]           MCP over stdio (default)
stylometric-mcp http [--bind ADDR] [--config PATH]
stylometric-mcp tools                           Print the MCP tool catalogue
stylometric-mcp fingerprint --input FILE [--title T] [--profile-id ID]
                           [--author-name N] [--fiction|--non-fiction]
                           [--out FILE] [--store] [--scaffold]
stylometric-mcp apply --fingerprint FILE --input FILE [--out FILE]
                      [--1st-person|--2nd-person|--3rd-person]
                      [--author-name N]
stylometric-mcp --help | --version
```

Short flags: `-c` (config), `-i` (input), `-o` (out), `-f` (fingerprint; `.json`
appended when missing).

```bash
./target/release/stylometric-mcp fingerprint \
  --input corpus.md --title "On Small Tools" --profile-id me --author-name me \
  --non-fiction --out me.fingerprint.json --store

./target/release/stylometric-mcp apply \
  --fingerprint me.fingerprint.json --input draft.md --out styled.md
```

`apply` writes only the rewritten Markdown to `--out` (or stdout) and prints the
compliance score to stderr.

---

## MCP server

JSON-RPC 2.0, protocol revision `2024-11-05`, server name `stylometric-mcp`.

- **stdio** - newline-delimited JSON-RPC on stdin/stdout (`serve` / `stdio`).
  Notifications (`notifications/*`, or any request without `id`) get no response.
- **HTTP** - minimal HTTP/1.1 listener (`http --bind`, default `127.0.0.1:8765`).
  Local convenience only: no auth, no TLS.

Methods: `initialize`, `ping`, `tools/list`, `tools/call`. Unknown methods return
`-32601`.

### Tools

| Tool | Purpose | Key arguments |
| --- | --- | --- |
| `fingerprint` | Build a fingerprint from a corpus | `documents[]` or `text`, `title`, `profile_id`, `author_name`, `fiction`, `excerpt_words` (>=200, default 1200), `store`, `allow_scaffold` |
| `apply` | Rewrite Markdown to match a fingerprint | `markdown` (required), `fingerprint` or `guid`, `fiction`, `person`, `author_name`, `max_retries` |
| `rate` | Score text against a fingerprint | `text` (required), `fingerprint` or `guid` |
| `similarity` | Compare two fingerprints | `fingerprint_a`/`guid_a`, `fingerprint_b`/`guid_b` |
| `get_fingerprint` | Fetch a stored fingerprint | `guid` (required) |

Results carry a JSON `content` text block plus `structuredContent`; failures set
`isError: true` with the message.

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | ./target/release/stylometric-mcp serve
```


---

## Configuration

LLM connection config resolution order:

1. `--config PATH` / `-c PATH`
2. `./config.llm.json` in the working directory (if present)
3. **embedded** `resources/config.llm.json` - a copy of the Python config with
   `api_key` blank

Recognized keys: `api_key`, `base_url`, `model`, `max_tokens`,
`max_prompt_tokens`, `temperature`, `timeout_seconds`, `max_retries`,
`backoff_base_seconds`, `backoff_max_seconds`, `extra_headers`.

Transient failures (429 / 5xx / transport) retry with exponential backoff and
jitter up to `max_retries`; other 4xx fail immediately so auth errors are visible.

### Embedded resources

| File | Bytes | Used for |
| --- | --- | --- |
| `prompts.json` | 14,020 | fingerprint and apply prompts |
| `config.tunables.json` | 6,925 | thresholds, chunking, retry budgets, humanizer policy |
| `config.local_spelling_rules.json` | 34,269 | deterministic locale spelling |
| `config.avoid.txt` | 140 | global hard-avoid list |
| `config.common_words.txt` | 63,025 | embedded for parity; not yet consumed |
| `config.entity_blacklist.txt` | 192,738 | embedded for parity; not yet consumed |
| `config.llm.json` | 238 | default LLM connection config |

All six data files are byte-identical to their Python originals (md5-verified).
Only LLM connection settings can come from outside the binary.

---

## Fingerprint store

GUID-addressed artifacts `<guid>.fingerprint.json` and `<guid>.meta.json`.
Directory resolution is runtime-only, so no build paths are baked in:

1. `STYLOMETRIC_MCP_STORE_DIR` environment variable, if set and non-empty
2. `./fingerprint_store`
3. `<exe-dir>/fingerprint_store`
4. system temp directory

GUIDs are validated (alphanumeric, `-`, `_`, max 64 chars) to prevent path escape.

---

## Status and limitations

- Deterministic measurement, compliance scoring, similarity, spelling, and
  humanizer post-processing are ported and exercised.
- Chunked rewrite with a per-chunk compliance retry loop is implemented
  (paragraph-boundary chunking).
- **Not yet ported:** prompt chunking and merge for large fingerprints, rolling
  chunk summaries, perplexity presets, roster/model rotation,
  `general-guidelines.md` LLM parsing, rare-word and entity-blacklist filtering.
- No unit tests yet; validation is CLI and JSON-RPC smoke testing.
- End-to-end LLM runs need a valid `api_key`.

---

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) - layers, data flow, module map
- [`AGENTS.md`](AGENTS.md) - guidance for automated agents working on this crate
- [`BENCHMARKS.md`](BENCHMARKS.md) - measured Rust vs Python performance

## License

PolyForm Noncommercial 1.0.0 (matches the parent project).

