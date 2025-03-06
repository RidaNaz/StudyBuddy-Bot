import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process, LLM
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory

load_dotenv()

os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

llm = LLM(model="gemini/gemini-2.0-flash")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

google_embedder = {
    "provider": "google",
    "config": {
         "model": "models/text-embedding-004",
         "api_key": os.getenv('GEMINI_API_KEY'),
    }
}

search_tool = SerperDevTool()

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

def run_chatbot(question):
    task = student_query_task(question)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        memory=True,
        long_term_memory=LongTermMemory(
            storage=LTMSQLiteStorage(db_path="./memory/long_term_memory.db")
        ),
        short_term_memory=ShortTermMemory(
            storage=RAGStorage(embedder_config=google_embedder, type="short_term", path="./memory/short_term/")
        ),
        entity_memory=EntityMemory(
            storage=RAGStorage(embedder_config=google_embedder, type="short_term", path="./memory/entity/")
        ),
        verbose=True,
        process=Process.sequential,
        embedder=google_embedder
    )
    result = crew.kickoff(inputs={"question": question})
    return result

st.set_page_config(page_title="Student Helper", layout="wide")
st.markdown("# 🎓 Student Helper")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown({message['content']})
    elif message["role"] == "assistant" and message['content'].strip():
        st.chat_message("assistant").markdown(f"💼 {message['content']}")

user_input = st.chat_input("Ask me anything about your studies...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.spinner("Thinking..."):
        response = run_chatbot(user_input)
        if response:
            # Extract text from CrewOutput safely
            response_text = str(response) if hasattr(response, 'strip') else str(response)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.chat_message("assistant").markdown(response_text)