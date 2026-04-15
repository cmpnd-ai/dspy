# Environment

- Python 3.14 via uv venv at .venv/
- dspy installed as editable from this worktree
- OPENAI_API_KEY and GROQ_API_KEY in environment
- DSPy sets LITELLM_LOCAL_MODEL_COST_MAP=True — newer models (gpt-5-nano) may not be in litellm's bundled DB
- History model_config has frozen=True in the pre-mission state — must change to allow message mutation
