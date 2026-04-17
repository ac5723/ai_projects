from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
import operator


# ─────────────────────────────────────────
# State with message history
# ─────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, operator.add]  # 👈 list that accumulates
    topic: str
    research: str
    summary: str


# ─────────────────────────────────────────
# Setup Ollama
# ─────────────────────────────────────────
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0
)


# ─────────────────────────────────────────
# Nodes with LLM
# ─────────────────────────────────────────
def research_node(state: State) -> State:
    """Research the topic using LLM"""
    print(f"\n🔍 Researching: {state['topic']}")

    prompt = f"""Research this topic and provide 3 key facts:
    Topic: {state['topic']}

    Provide exactly 3 numbered facts. Keep each fact to 1-2 sentences."""

    research = llm.invoke(prompt)
    print(f"✅ Research complete")

    return {
        "messages": [AIMessage(content=f"Research: {research}")],
        "topic": state["topic"],
        "research": research,
        "summary": ""
    }


def summary_node(state: State) -> State:
    """Summarize the research"""
    print(f"\n📝 Summarizing research...")

    prompt = f"""Summarize these research findings in one short paragraph:

    {state['research']}

    Keep summary to 3-4 sentences."""

    summary = llm.invoke(prompt)
    print(f"✅ Summary complete")

    return {
        "messages": [AIMessage(content=f"Summary: {summary}")],
        "topic": state["topic"],
        "research": state["research"],
        "summary": summary
    }


def quality_check(state: State) -> str:
    """Check if summary is good enough"""
    # simple check - if summary is long enough
    if len(state["summary"]) > 100:
        print("\n✅ Quality check passed!")
        return "good"
    else:
        print("\n⚠️  Summary too short, researching again...")
        return "retry"


# ─────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────
graph = StateGraph(State)

graph.add_node("research", research_node)
graph.add_node("summarize", summary_node)

graph.set_entry_point("research")
graph.add_edge("research", "summarize")

# conditional edge - retry if quality is bad
graph.add_conditional_edges(
    "summarize",
    quality_check,
    {
        "good": END,  # good quality → end
        "retry": "research"  # bad quality → research again!
    }
)

app = graph.compile()

# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
print("=" * 50)
print("🚀 LangGraph + Ollama Research Agent")
print("=" * 50)

result = app.invoke({
    "messages": [HumanMessage(content="Research Python programming")],
    "topic": "Python programming language",
    "research": "",
    "summary": ""
})

print(f"\n{'=' * 50}")
print("📊 FINAL RESULT:")
print(f"{'=' * 50}")
print(f"\n🔍 Research:\n{result['research']}")
print(f"\n📝 Summary:\n{result['summary']}")