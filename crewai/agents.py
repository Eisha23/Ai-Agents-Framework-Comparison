from crewai import Agent
from tools import search_web, summarize

# Researcher Agent
researcher = Agent(
    role="Researcher",
    goal="Find relevant and reliable web articles for the given topic",
    backstory="An internet-savvy researcher skilled at finding the most relevant " \
    "content from search engines.",
    tools=[search_web],
    verbose=True
)

# Summarizing Agent
summarizer = Agent(
    role="Summarizer",
    goal="Summarize collected articles",
    backstory="An expert at summarizing long articles using clear, easy-to-read language, " \
    "creating a 1-2 paragraph research summary.",
    tools=[summarize],
    verbose=True
)
