import dspy

from dspy.predict.reactv2 import ReActV2

dspy.configure(lm=dspy.LM("openai/gpt-5-nano"))

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

react = ReActV2("question->answer", tools=[get_weather])

result = react(question="What is the weather in Tokyo and in New York? Answer using parallel tool calls.")

print(result)