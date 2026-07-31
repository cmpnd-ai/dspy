"""The deliberately small DSPy facade installed in Monty guests."""

SHIM = r"""
class _Module: pass
class _PredictorResult:
    def __init__(self, fields):
        self._fields = fields
        for name, value in fields.items(): setattr(self, name, value)
    def get(self, name, default=None): return self._fields.get(name, default)
def _dspy_fields(value):
    return value._fields if isinstance(value, _PredictorResult) else value
def _predictor(kind, signature, config):
    handle = __dspy_construct__(kind, signature, config)
    def call(**inputs): return _PredictorResult(__dspy_call__(handle, inputs))
    return call
class _DSPy:
    Module = _Module
    def Signature(self, signature, instructions=None):
        return {"__dspy_signature__": True, "signature": signature, "instructions": instructions}
    def Prediction(self, **fields): return fields
    def Tool(self, tool): return tool
dspy = _DSPy()
def _constructor(kind):
    def construct(signature, **config): return _predictor(kind, signature, config)
    return construct
dspy.Predict = _constructor("Predict")
dspy.ChainOfThought = _constructor("ChainOfThought")
dspy.RLM = _constructor("RLM")
dspy.CodeAct = _constructor("CodeAct")
dspy.ProgramOfThought = _constructor("ProgramOfThought")
dspy.ReAct = _constructor("ReAct")
dspy.ReActV2 = _constructor("ReActV2")
"""
