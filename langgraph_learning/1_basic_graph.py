from typing import TypedDict
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────
# STEP 1 — Define State
# ─────────────────────────────────────────
# State is shared data passed between all nodes
class State(TypedDict):
    message: str      # input message
    result: str       # output result
    step: int         # which step we're on

# ─────────────────────────────────────────
# STEP 2 — Define Nodes
# ─────────────────────────────────────────
# Each node is a function that takes state and returns updated state

def node_greet(state: State) -> State:
    print(f"👋 Node 1: Greeting")
    return {
        "message": state["message"],
        "result": f"Hello! You said: {state['message']}",
        "step": 1
    }

def node_process(state: State) -> State:
    print(f"⚙️  Node 2: Processing")
    return {
        "message": state["message"],
        "result": state["result"] + " | Processed!",
        "step": 2
    }

def node_finish(state: State) -> State:
    print(f"✅ Node 3: Finishing")
    return {
        "message": state["message"],
        "result": state["result"] + " | Done!",
        "step": 3
    }

# ─────────────────────────────────────────
# STEP 3 — Build Graph
# ─────────────────────────────────────────
graph = StateGraph(State)

# add nodes
graph.add_node("greet", node_greet)
graph.add_node("process", node_process)
graph.add_node("finish", node_finish)

# add edges (connections between nodes)
graph.set_entry_point("greet")      # start here
graph.add_edge("greet", "process")  # greet → process
graph.add_edge("process", "finish") # process → finish
graph.add_edge("finish", END)       # finish → end

# compile the graph
app = graph.compile()

# ─────────────────────────────────────────
# STEP 4 — Run Graph
# ─────────────────────────────────────────
print("="*50)
print("🚀 Running Basic Graph")
print("="*50)

result = app.invoke({
    "message": "I am learning LangGraph!",
    "result": "",
    "step": 0
})

print(f"\n📊 Final State:")
print(f"Message: {result['message']}")
print(f"Result: {result['result']}")
print(f"Steps completed: {result['step']}")