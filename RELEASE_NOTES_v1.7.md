### Release Notes (v1.7 vs v1.5)

1. v1.7 moves beyond core stylometric transfer into a more controllable and auditable pipeline.

2. Retry logic is now split by concern: style retries and forced-person voice retries use separate budgets.

3. When retries are exhausted, the best-scoring attempt is kept instead of always taking the last attempt.

4. Verbose logs now show per-chunk attempt scores and when best-attempt replacement is applied.

5. Chunking is more robust: configurable split mode (paragraph/sentence/word) with automatic fallback when a unit is too large.

6. Added perturbation-aware minimum chunk count and variance-aware chunk sizing controls.

7. Added rolling chunk-summary continuity chaining with safeguards for empty/meta summaries.

8. Added corpus-derived humanization baselines and per-chunk controller overlays.

9. Added bounded controller-feedback retries so overlay feedback does not create unlimited retry growth.

10. Added deterministic heading controls: qualifier sanitization, allowlist support, and heading-case normalization modes.

11. Added per-level heading-case policies across H1-H8.

12. Added preserve_proper_name_case for deterministic heading case transforms.

13. Added deterministic normalization of curly double quotes to straight double quotes.

14. Local spelling support is stronger via force_local_spelling and an external rules file.

15. Lexical avoidance matching now uses a US-normalized comparison pass before final locale spelling output.

16. Fingerprinting now better normalizes and de-duplicates rewrite policy and priority order fields.

17. Rare-word hygiene improved (de-duplication and filtering of obvious malformed tokens).

18. Non-author voice preservation remains strict through rewrite and deterministic post-processing.

19. Smoke testing now defaults to regression-only; LLM tests are opt-in via --llm-tests.

20. v1.7 regression coverage expanded significantly for chunking, retries, spelling, and deterministic normalization.

21. Documentation and AGENTS guidance were updated to align with runtime behavior and operator workflows.
