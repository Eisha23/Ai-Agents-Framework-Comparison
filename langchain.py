import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools import search_web, summarize
from langchain.agents import ZeroShotAgent, AgentExecutor  
from langchain.chains import LLMChain
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", openai_api_key=OPENAI_API_KEY)

tools = [search_web, summarize]
prefix = """You are an AI assistant that MUST use tools when applicable. 
If the user enters a query, you MUST search the web for relevant articles.
Then you MUST use the summarizing tool to create a 1-2 paragraph research summary.
You have access to the following tools:"""
suffix = """ 'Begin!'

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True
)

chat_history = [] 

while True:
    query = input("Enter your research query or 'exit' to quit: ").strip()
    if query.lower() in ['exit', 'quit']:
            print("Exiting.")
            break

    result = agent_executor.run(query)  
    chat_history.append((query, result))

    # Display chat history 
    print("\n" + "=" * 60)
    print("Chat History")
    for i, (q, a) in enumerate(chat_history, 1):
        print(f"\n{i}. You: {q}\n")
        print(f"   Assistant: {a}")
        print("-" * 60)
    print("=" * 60 + "\n")

 
