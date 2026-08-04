from crewai import Crew
from tasks import define_tasks
import os
from dotenv import load_dotenv
load_dotenv()

chat_history = [] 

if __name__ == "__main__":
    while True:
        query = input("Enter your research query or type 'exit' to quit: ").strip()

        if query.lower() in ['exit', 'quit']:
            print("Exiting.")
            break

        # Load tasks
        tasks = define_tasks(query)

        # Set up the crew with those tasks
        crew = Crew(
            agents=[task.agent for task in tasks],
            tasks=tasks,
            verbose=True
        )
        # Run the crew 
        result = crew.kickoff()
        chat_history.append((query, result))

        # Display chat history 
        print("\n" + "=" * 60)
        print("Chat History")
        for i, (q, a) in enumerate(chat_history, 1):
            print(f"\n{i}. You: {q}\n")
            print(f"   Assistant: {a}")
            print("-" * 60)
        print("=" * 60 + "\n")
