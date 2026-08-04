import os
import requests
from newspaper import Article
from crewai.tools import tool
from openai import OpenAI
from dotenv import load_dotenv

# Load API keys
load_dotenv()

serper_key = os.getenv("SERPER_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)
model="gpt-4.1-mini-2025-04-14"


@tool("Web Search Tool")
def search_web(query: str) -> str:
    """Use Serper to search the web and return article texts"""
    print("Using: Web Search Tool")

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json"
    }
    payload = {"q": query}

    response = requests.post(url, headers=headers, json=payload)
    results = response.json()

    text = []

    for r in results.get("organic", [])[:3]:
        article_url = r.get("link")
        print(article_url)
        try:
            article = Article(article_url)
            article.download()
            article.parse()
            text.append(article.text)
        except:
            continue

    return "\n\n".join(text)

@tool("Summarizing Tool")
def summarize(text: str) -> str:
    """Summarizes long text"""
    print("Using: Summarizing Tool")

    if not text.strip():
        return "No text to summarize."
    
    prompt = "Summarize the following text and create a 1-2 paragraph research summary:\n\n" + text
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
