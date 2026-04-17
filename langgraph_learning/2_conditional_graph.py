from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────
# State
# ─────────────────────────────────────────
class State(TypedDict):
    question: str
    category: str    # math, general, unknown
    answer: str

# ─────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────
def classify_question(state: State) -> State:
    """Classify the question into a category"""
    question = state["question"].lower()

    if any(word in question for word in
           ["calculate", "sum", "multiply", "divide", "plus", "minus"]):
        category = "math"
    elif any(word in question for word in
             ["what", "who", "when", "where", "why", "how"]):
        category = "general"
    else:
        category = "unknown"

    print(f"🔍 Classified as: {category}")
    return {**state, "category": category}

def handle_math(state: State) -> State:
    """Handle math questions"""
    print("🔢 Handling math question...")
    # simple math handling
    question = state["question"]
    try:
        # extract numbers and operation
        if "plus" in question or "+" in question:
            nums = [int(s) for s in question.split() if s.isdigit()]
            answer = f"The answer is {sum(nums)}"
        else:
            answer = "I can handle basic math. Try asking '5 plus 3'"
    except:
        answer = "Could not calculate. Please rephrase."

    return {**state, "answer": answer}

def handle_general(state: State) -> State:
    """Handle general questions"""
    print("💬 Handling general question...")
    return {
        **state,
        "answer": f"You asked: '{state['question']}'. "
                  f"This is a general knowledge question!"
    }

def handle_unknown(state: State) -> State:
    """Handle unknown questions"""
    print("❓ Handling unknown question...")
    return {
        **state,
        "answer": "I'm not sure how to categorize this question."
    }

# ─────────────────────────────────────────
# Conditional Router
# ─────────────────────────────────────────
def route_question(state: State) -> Literal["math", "general", "unknown"]:
    """Decide which node to go to based on category"""
    return state["category"]   # returns "math", "general", or "unknown"

# ─────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────
graph = StateGraph(State)

# add nodes
graph.add_node("classify", classify_question)
graph.add_node("math", handle_math)
graph.add_node("general", handle_general)
graph.add_node("unknown", handle_unknown)

# set entry point
graph.set_entry_point("classify")

# add CONDITIONAL edge! 👈
graph.add_conditional_edges(
    "classify",           # from this node
    route_question,       # call this function to decide
    {
        "math": "math",        # if returns "math" → go to math node
        "general": "general",  # if returns "general" → go to general node
        "unknown": "unknown"   # if returns "unknown" → go to unknown node
    }
)

# all paths lead to END
graph.add_edge("math", END)
graph.add_edge("general", END)
graph.add_edge("unknown", END)

app = graph.compile()

# ─────────────────────────────────────────
# Test with different questions
# ─────────────────────────────────────────
questions = [
    "Calculate 5 plus 3",
    "What is the capital of India?",
    "xyz abc 123"
]

for question in questions:
    print(f"\n{'='*50}")
    print(f"❓ Question: {question}")
    result = app.invoke({
        "question": question,
        "category": "",
        "answer": ""
    })
    print(f"💬 Answer: {result['answer']}")