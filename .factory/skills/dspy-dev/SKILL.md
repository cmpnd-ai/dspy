---
name: dspy-dev
description: DSPy module development with strict LOC budget
---

# DSPy Dev Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that modify DSPy library code (dspy/predict/, dspy/adapters/, dspy/clients/) and their tests.

## Required Skills

None

## Work Procedure

1. **Read the feature description carefully.** Note the LOC budget — total diff must be < +1000 lines.

2. **Check current LOC usage:** `git diff 09eba10a --stat -- '*.py' | tail -1`. If approaching 900+, be extremely conservative.

3. **Write failing tests first** in `tests/predict/test_reactv2.py`. Use DummyLM and mock patterns — no real API calls in tests. Tests should be minimal (5-10 lines each, no verbose setup).

4. **Implement the minimum code** to make tests pass. No verbose docstrings, no redundant comments, no defensive coding that isn't tested. Every line must serve a purpose.

5. **Run tests:** `uv run pytest tests/predict/test_reactv2.py -x -v`

6. **Run regression tests:** `uv run pytest tests/adapters/test_chat_adapter.py tests/adapters/test_json_adapter.py -x -v`

7. **Check LOC:** `git diff 09eba10a --stat -- '*.py' | tail -1` — report the number.

8. **For integration verification** (real API calls), run as verification commands (not tests):
   ```
   uv run python -c "import dspy; from dspy.predict.reactv2 import ReActV2; ..."
   ```

## Example Handoff

```json
{
  "salientSummary": "Completed forward loop + submit tool. 8 tests passing, LOC at +320. Submit returns dict of output fields, forced submit on max_iters, error handling for parse errors and None tool_calls.",
  "whatWasImplemented": "Fixed submit tool to return kwargs dict, completed forward() with error handling, forced submit fallback, per-call max_iters override. Removed debug print. Fixed History frozen=True. 8 unit tests.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv run pytest tests/predict/test_reactv2.py -x -v", "exitCode": 0, "observation": "8 passed"},
      {"command": "uv run pytest tests/adapters/ -x -v", "exitCode": 0, "observation": "63 passed, no regressions"},
      {"command": "git diff 09eba10a --stat -- '*.py' | tail -1", "exitCode": 0, "observation": "5 files changed, 320 insertions(+), 15 deletions(-)"}
    ]
  },
  "tests": {
    "added": [{"file": "tests/predict/test_reactv2.py", "cases": [
      {"name": "test_basic_forward_with_submit", "verifies": "VAL-CORE-003"},
      {"name": "test_submit_returns_dict", "verifies": "VAL-CORE-002"}
    ]}]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- LOC budget is about to be exceeded (>900 lines and feature needs more)
- A pre-existing bug in adapter/LM code blocks the feature
- Requirements are ambiguous about native vs non-native behavior
