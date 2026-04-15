# User Testing

## Validation Surface
- CLI: pytest unit tests + python -c integration scripts
- No web UI, no browser testing needed

## Validation Concurrency
- Max 1 concurrent validator (Groq rate limits, API costs)
- Serial execution only

## Benchmark Access
- BrowseComp corpus: /Users/isaac/projects/langprobe_recurring/data/cache/browsecomp/
- Tau-banking: /Users/isaac/projects/langprobe_recurring/benchmarks/tau_banking/
- Run via PYTHONPATH override: PYTHONPATH=/Users/isaac/projects/dspy-worktrees/isaac/react-v2
