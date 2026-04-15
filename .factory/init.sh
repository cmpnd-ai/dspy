#!/bin/bash
set -e
cd /Users/isaac/projects/dspy-worktrees/isaac/react-v2
uv sync --all-extras 2>/dev/null || uv pip install -e ".[all]" 2>/dev/null || true
uv run python -c "import dspy; assert 'react-v2' in dspy.__file__; print('dspy OK:', dspy.__file__)"
