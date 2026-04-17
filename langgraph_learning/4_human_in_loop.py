from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import OllamaLLM

# ─────────────────────────────────────────
# State
# ─────────────────────────────────────────
class State(TypedDict):
    task: str
    plan: str
    approved: bool
    result: str

# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────
llm = OllamaLLM(model="llama3.2:3b", temperature=0)

# ─────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────
def create_plan(state: State) -> State:
    """AI creates a plan"""
    print("\n🤖 AI is creating a plan...")
    plan = llm.invoke(
        f"Create a brief 3-step plan for: {state['task']}"
    )
    print(f"\n📋 AI Plan:\n{plan}")
    return {**state, "plan": plan, "approved": False}

def get_human_approval(state: State) -> State:
    """Ask human to approve the plan"""
    print("\n" + "="*50)
    print("👤 HUMAN APPROVAL REQUIRED")
    print("="*50)
    print(f"Task: {state['task']}")
    print(f"\nProposed Plan:\n{state['plan']}")
    print("\nDo you approve this plan?")

    approval = input("Enter 'yes' to approve or 'no' to reject: ")
    approved = approval.lower() == "yes"

    return {**state, "approved": approved}

def execute_plan(state: State) -> State:
    """Execute the approved plan"""
    print("\n🚀 Executing approved plan...")
    result = llm.invoke(
        f"Execute this plan and show results:\n{state['plan']}"
    )
    return {**state, "result": result}

def reject_plan(state: State) -> State:
    """Handle rejected plan"""
    print("\n❌ Plan rejected by human.")
    return {**state, "result": "Plan was rejected. Please try again."}

def route_approval(state: State) -> str:
    """Route based on human approval"""
    if state["approved"]:
        return "approved"
    else:
        return "rejected"

# ─────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────
graph = StateGraph(State)

graph.add_node("create_plan", create_plan)
graph.add_node("human_approval", get_human_approval)
graph.add_node("execute", execute_plan)
graph.add_node("reject", reject_plan)

graph.set_entry_point("create_plan")
graph.add_edge("create_plan", "human_approval")

graph.add_conditional_edges(
    "human_approval",
    route_approval,
    {
        "approved": "execute",
        "rejected": "reject"
    }
)

graph.add_edge("execute", END)
graph.add_edge("reject", END)

app = graph.compile()

# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
print("="*50)
print("🚀 LangGraph Human-in-the-Loop Demo")
print("="*50)

result = app.invoke({
    "task": "Learn Python in 30 days",
    "plan": "",
    "approved": False,
    "result": ""
})

print(f"\n{'='*50}")
print("📊 FINAL RESULT:")
print(result["result"])