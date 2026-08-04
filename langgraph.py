import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict
from tools import search_web, summarize

# Load env 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 
llm = ChatOpenAI(
    model="gpt-4.1-mini-2025-04-14",
    api_key=OPENAI_API_KEY,
    temperature=0
)

search_web_tool = Tool(
    name="search_web",
    func=search_web,
    description="Search the web and return text results."
)

summarize_tool = Tool(
    name="summarize",
    func=summarize,
    description="Summarize given text."
)

# State 
class AgentState(TypedDict):
    query: str
    search_results: str
    summary: str

# Node: Researcher 
def researcher_node(state: AgentState):
    results = search_web.run(state["query"])
    return {"search_results": results}

# Node: Summarizer 
def summarizer_node(state: AgentState):
    summary = summarize.run(state["search_results"])
    return {"summary": summary}

# Graph 
graph = StateGraph(AgentState)

graph.add_node("researcher", researcher_node)
graph.add_node("summarizer", summarizer_node)

graph.add_edge("researcher", "summarizer")
graph.add_edge("summarizer", END)

graph.set_entry_point("researcher")

# Compile 
app = graph.compile()

# store all history
chat_history = []

# run loop
while True:
    query = input("Enter your research query or 'exit' to quit: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    result_state = app.invoke({"query": query})
    summary = result_state["summary"]

    chat_history.append((query, summary))

    print("\n" + "=" * 60)
    print("Chat History")
    for i, (q, a) in enumerate(chat_history, 1):
        print(f"\n{i}. You: {q}\n")
        print(f"   Assistant: {a}")
        print("-" * 60)
    print("=" * 60 + "\n")
