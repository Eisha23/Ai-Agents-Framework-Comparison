from crewai import Task
from agents import researcher, summarizer

def define_tasks(query):
    research_task = Task(
        description=f"Search the web for the most relevant and recent articles about: '{query}'",
        agent=researcher,
        expected_output="A list of links and article texts related to the query.",
    )

    summarize_task = Task(
        description=(
            "Summarize the research findings, creating a 1-2 paragraph research summary "
            f"that answers the query: '{query}'"
        ),
        agent=summarizer,
        expected_output="A concise research summary.",
        context=[research_task]
    )

    return [research_task, summarize_task]
