import os
from dotenv import load_dotenv
import google.generativeai as genai
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process, LLM
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory

load_dotenv()

os.environ['GEMINI_API_KEY'] =  os.getenv('GEMINI_API_KEY')
os.environ['SERPER_API_KEY'] =  os.getenv('SERPER_API_KEY')

llm = LLM(
    model="gemini/gemini-2.0-flash",
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

google_embedder = {
    "provider": "google",
    "config": {
         "model": "models/text-embedding-004",
         "api_key": os.getenv('GEMINI_API_KEY'),
         }
}

search_tool = SerperDevTool()

# Create an agent with the knowledge store
agent = Agent(
    role='Student Helper',
    goal='Answer students’ questions or search the web/YouTube if unsure.',
    backstory='A knowledgeable AI assistant who helps students with their queries and provides relevant resources.',
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

def student_query_task(question):
    return Task(
        description=f"Answer the student query or search online if needed: {question}",
        expected_output='A helpful answer to the student’s question, with links to additional resources if available.',
        agent=agent,
    )


# Assemble the crew
def run_chatbot(question):
    task = student_query_task(question)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        memory=True,
        long_term_memory=LongTermMemory(
            storage=LTMSQLiteStorage(
                db_path="./memory/long_term_memory.db"
            )
        ),
        short_term_memory=ShortTermMemory(
            storage=RAGStorage(
                embedder_config=google_embedder,
                type="short_term",
                path="./memory/short_term/"
            )
        ),
        entity_memory=EntityMemory(
            storage=RAGStorage(
                embedder_config=google_embedder,
                type="short_term",
                path="./memory/entity/"
            )
        ),
        # Long-term memory for persistent storage across sessions
        long_term_memory = LongTermMemory(
            storage=LTMSQLiteStorage(
                db_path="./long_term_memory.db"
            )
        ),
        # Short-term memory for current context using RAG
        short_term_memory = ShortTermMemory(
            storage = RAGStorage(
                    embedder_config=google_embedder,
                    type="short_term",
                    path="./short_term/"
                )
            ),
    
        # Entity memory for tracking key information about entities
        entity_memory = EntityMemory(
            storage=RAGStorage(
                embedder_config=google_embedder,
                type="short_term",
                path="./entity/"
            )
        ),
        verbose=True,
        process=Process.sequential,
        embedder=google_embedder
    )
    result = crew.kickoff(inputs={"question": question})
    return result

# Example query
question = "Where the holding period exceeds 4 Years but does not exceed 5 Years"
response = run_chatbot(question)
print("Bot's Response:", response)