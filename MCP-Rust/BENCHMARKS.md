# Benchmarks - Rust vs Python

Measured comparison of the `stylometric-transfer` Python implementation and the
`MCP-Rust` server, restricted to work both can do without an LLM call: measurement
and compliance scoring.

---

## 1. Method

**Host:** single Linux workstation, shared multi-tenant machine, no CPU pinning, no
cgroup isolation. Numbers are indicative, not a controlled microbenchmark.

**Corpus:** synthetic Markdown - 16,923 words, 240 paragraphs, 1,200 sentences,
drawn from a 20-word vocabulary with `random.seed(7)`.

**Fingerprint:** one real LLM-produced fingerprint (`measurements` only; `targets`
and `validators` empty), reused for both implementations so the work is identical.

**Rust:** a full `tools/call` for `rate` over stdio - process spawn, stdin write,
JSON parse, measurement of 16,923 words, and compliance scoring. 7 runs, median
reported.

**Python:** in-process `apply_fingerprint.compute_measurements` and
`apply_fingerprint.compute_style_compliance` after import. 5 runs each, median
reported. Interpreter startup measured separately.

---

## 2. Results

### Startup

| Operation | Median | Min |
| --- | --- | --- |
| Rust `stylometric-mcp --version` | 1.3 ms | 0.9 ms |
| Python `python3 -c "import apply_fingerprint"` | 120.6 ms | 88.5 ms |
| Python `apply_fingerprint.py --help` | 112.0 ms | 110.0 ms |

### Measurement + scoring (16,923-word corpus)

| Operation | Median | Min |
| --- | --- | --- |
| Python `compute_measurements` | 118.5 ms | 94.3 ms |
| Python `compute_style_compliance` | 93.8 ms | 93.1 ms |
| **Python combined (in-process)** | **~212 ms** | - |
| **Rust `rate` end-to-end (incl. spawn)** | **26.2 ms** | **23.7 ms** |

**Approximate speedup: ~8x** on the combined measure and score workload.

Because the Rust figure includes ~1.3 ms of process startup, the pipeline-only cost
is roughly 25 ms (about 8.5x). That subtraction is an estimate, not a separate
measurement.

### Footprint

| Metric | Python | Rust |
| --- | --- | --- |
| Runtime artifact | 1.3 GB working tree + interpreter + deps | 4.6 MB static binary |
| Source lines (pipeline) | 8,542 + 3,781 + 579 | 5,425 across 14 modules |
| Direct dependencies | stdlib + project modules | 12 crates |
| Startup to usable | ~110-120 ms | ~1.3 ms |

---

## 3. Interpretation

- The Rust advantage is dominated by **removing interpreter startup** and by
  **compiled regex and tokenization**. Both implementations are single-threaded;
  neither uses parallelism.
- The gap matters more for agents than the raw ratio suggests: an MCP server pays
  startup once and then serves many calls, so per-call cost (~25 ms) dominates.
- The Python figures are in-process, which **favours Python** in this comparison;
  a real Python CLI run adds another ~110 ms.
- **Scores are not compared for parity.** The Rust scorer is a port, not a
  bit-exact reimplementation; on this corpus Python reported compliance 0.457.
  Cross-implementation score deltas are expected, not regressions.

---

## 4. Caveats

1. Single shared host, no isolation; run-to-run variance was roughly 10-20%.
2. Synthetic corpus; real prose differs in tokenization and quote density.
3. The fingerprint has empty `targets` and `validators`, so both implementations
   fall back to unweighted section averaging.
4. LLM latency is excluded entirely. In real use it dominates (seconds per call),
   so end-to-end speedup is bounded by the model.
5. No memory measurement was taken. The Rust resident set is small but
   unquantified here.

---

## 5. Reproducing

Generate the synthetic corpus:

```bash
python3 - <<'PY'
import random
random.seed(7)
words = ['system','review','code','quality','the','a','of','and','function',
         'clear','reader','document','test','reasoning','because','however',
         'therefore','simple','small','tool']
sents = []
for _ in range(1200):
    n = random.randint(6, 22)
    sents.append(' '.join(random.choice(words) for _ in range(n)).capitalize() + '.')
paras = [' '.join(sents[i:i+5]) for i in range(0, len(sents), 5)]
open('/tmp/bench_corpus.md', 'w').write('\n\n'.join(paras))
PY
```

Rust timing (end-to-end `rate` tool call over stdio, includes process spawn):

```bash
cd MCP-Rust && cargo build --release --offline
python3 - <<'PY'
import json, statistics, subprocess, time
text = open('/tmp/bench_corpus.md').read()
fp = {'measurements': json.load(open('/tmp/fp.json'))['measurements'],
      'targets': {}, 'validators': {}}
req = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
       'params': {'name': 'rate', 'arguments': {'text': text, 'fingerprint': fp}}}
payload = json.dumps(req) + '\n'
times = []
for _ in range(7):
    t = time.perf_counter()
    p = subprocess.run(['./target/release/stylometric-mcp', 'serve'],
                       input=payload, capture_output=True, text=True)
    times.append((time.perf_counter() - t) * 1000)
    assert '"error"' not in p.stdout
print('rust rate ms: median %.1f min %.1f' % (statistics.median(times), min(times)))
PY
```

Python timing (in-process, interpreter startup excluded):

```bash
cd .. && python3 - <<'PY'
import json, statistics, time
import apply_fingerprint as A
text = open('/tmp/bench_corpus.md').read()
fp = {'measurements': json.load(open('/tmp/fp.json'))['measurements'],
      'targets': {}, 'validators': {}}
m, c = [], []
for _ in range(5):
    t = time.perf_counter(); A.compute_measurements(text)
    m.append((time.perf_counter() - t) * 1000)
for _ in range(5):
    t = time.perf_counter(); A.compute_style_compliance(fp, text)
    c.append((time.perf_counter() - t) * 1000)
print('python measurements ms: median %.1f' % statistics.median(m))
print('python compliance   ms: median %.1f' % statistics.median(c))
PY
```

When publishing new numbers, record the host, corpus word count, run count,
median and min, and whether process startup is included.


