import os
from dotenv import load_dotenv
from autogen.oai import OpenAIWrapper
from autogen.agentchat import GroupChat, AssistantAgent, UserProxyAgent, GroupChatManager
from tools import search_web, summarize

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
#model = "gpt-4.1-mini-2025-04-14"
model = "gpt-4.1-2025-04-14"

model_client = OpenAIWrapper(
    model=model,
    api_key=openai_key
)

# Define tool schemas (JSON format)
# needed because AutoGen follows OpenAI's strict function-calling format.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarizes long text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    }
]

# Initialize agents 
researcher = AssistantAgent(
    name="Researcher",
    system_message=(
        "You ONLY perform web searches using the search_web tool. "
        "After calling the search_web tool and receiving results, "
        "you MUST respond with a short message confirming the search is complete. "
        "For example, you can say: 'The search is done. Results will be passed to the Summarizer agent.' "
        "Do NOT provide any additional information or summaries."
    ),
    llm_config={
        "config_list": [{"model": model, "api_key": openai_key}],"temperature": 0}
)

summarizer = AssistantAgent(
    name="Summarizer",
    system_message=(
        "You ONLY summarize the text provided using the summarize tool. "
        "After the summary is returned, it is MUST to respond with EXACTLY 'TERMINATE' on a new message. "
        "Do NOT combine the summary and 'TERMINATE' in the same message. "
        "Do NOT add any commentary."

    ),
    llm_config={
        "config_list": [{"model": model, "api_key": openai_key}],"temperature": 0}
)

#links tool names in JSON to the functions
def create_function_map():
    return {
        "search_web": search_web,
        "summarize": summarize
    }

# Register the functions with both agents
function_map = create_function_map()

researcher.register_for_llm(name="search_web", description="Search the web for information")(search_web) # tell agent about this tool
researcher.register_for_execution(name="search_web")(search_web) # connects tool name to function

summarizer.register_for_llm(name="summarize", description="Summarize text content")(summarize)  
summarizer.register_for_execution(name="summarize")(summarize)


user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",  # "ALWAYS" for manual approval
    code_execution_config=False
)

# Group chat setup
groupchat = GroupChat(
    agents=[user_proxy, researcher, summarizer],
    messages=[],
    max_round=8,    
)
manager = GroupChatManager(
    groupchat=groupchat, 
    llm_config=researcher.llm_config,
    system_message="""Manage conversation flow:
    1. If user asks a question → Researcher
    2. If Researcher returns results → Summarizer
    3. After summary, end the chat. """,
    )

chat_history = [] 
while True:
    query = input("Enter your research query or 'exit' to quit: ").strip()
    if query.lower() in ['exit', 'quit']:
            print("Exiting.")
            break
    
    # start the agents
    result = user_proxy.initiate_chat(
        manager,
        message=query
        )
    
    # Get the last message from the chat history
    final_message = result.chat_history[-2]['content']
    chat_history.append((query, final_message))

    # Display chat history 
    print("\n" + "=" * 60)
    print("Chat History")
    for i, (q, a) in enumerate(chat_history, 1):
        print(f"\n{i}. You: {q}\n")
        print(f"   Assistant: {a}")
        print("-" * 60)
    print("=" * 60 + "\n")
