import os
from crewai import Agent, Task, Crew, Process, LLM

# ─────────────────────────────────────────
# Setup LOCAL LLM with Ollama
# ─────────────────────────────────────────
llm = LLM(
    model="ollama/llama3.2:3b",  # 👈 ollama/ prefix tells CrewAI to use Ollama
    base_url="http://localhost:11434"  # 👈 Ollama runs on this port locally
)

# ─────────────────────────────────────────
# Create Agents
# ─────────────────────────────────────────
researcher = Agent(
    role="Python Research Analyst",
    goal="Research and explain Python concepts clearly",
    backstory="""You are an experienced Python developer with 
    10 years of experience. You explain concepts simply.""",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear Python tutorials",
    backstory="""You are a technical writer who creates 
    beginner-friendly Python tutorials.""",
    llm=llm,
    verbose=True
)

# ─────────────────────────────────────────
# Create Tasks
# ─────────────────────────────────────────
research_task = Task(
    description="""Research and explain these Python concepts:
    1. What are lists?
    2. What are dictionaries?
    3. What are functions?
    Keep each explanation to 2-3 sentences.""",
    expected_output="""Clear explanation of lists, 
    dictionaries and functions in Python.""",
    agent=researcher
)

write_task = Task(
    description="""Take the research about Python concepts and 
    write a beginner-friendly tutorial. Add a simple code 
    example for each concept.""",
    expected_output="""A beginner Python tutorial with 
    explanations and code examples for lists, 
    dictionaries and functions.""",
    agent=writer,
    context=[research_task]
)

# ─────────────────────────────────────────
# Create and Run Crew
# ─────────────────────────────────────────
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

print("\n🚀 Starting LOCAL CrewAI with Ollama...")
print("No internet needed! Everything runs on your machine!")
print("="*50)

result = crew.kickoff()
print("\n" + "="*50)
print("✅ CREW COMPLETE!")
print("="*50)
print(result)