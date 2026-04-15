# Architecture

## ReActV2 vs ReActV1

| Aspect | v1 (react.py) | v2 (reactv2.py) |
|--------|--------------|-----------------|
| Trajectory | flat dict: thought_N, tool_name_N, tool_args_N, observation_N | History with semantic events: REQUEST, ACTION, FINAL |
| Tool selection | Literal[tool_names] enum + dict args (2 output fields) | dspy.ToolCalls (single structured output field) |
| Termination | finish tool -> separate extract LM call | submit tool returns output fields directly (no extract) |
| Compaction | 3 retries on ContextWindowExceededError | Pluggable compact_if_needed() each iteration |
| Native FC | Not supported | Supported when adapter + LM both support it |

## Data Flow

1. User calls `agent(question="...", history=None)`
2. Forward creates/reuses History, enters iteration loop
3. Each iteration: compact_if_needed() -> predict(history, tools, inputs) -> execute tool calls -> add_message to history
4. On submit: return Prediction(answer=..., history=history)
5. On max_iters: attempt forced submit, then None

## Key Invariants

- History is stateless on the module — passed in and returned out
- submit tool's args match signature output fields exactly
- Both native and non-native paths produce clear output format guidance for the model
- Total diff from 09eba10a must be < +1000 LOC
