import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process, LLM
from crewai.memory.storage.rag_storage import RAGStorage
# from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
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
    role='Study Buddy',
    goal='Answer students’ questions or search the web/YouTube if unsure, but strictly limit responses to study-related topics.',
    backstory='A knowledgeable AI assistant who helps students with their queries and provides relevant resources. It is programmed to strictly focus on study-related topics and refuses to answer anything outside this scope.',
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
        # long_term_memory=LongTermMemory(
        #     storage=LTMSQLiteStorage(db_path="./memory/long_term.db")
        # ),
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

st.set_page_config(page_title="StudyBuddy AI", page_icon="favicon.ico", layout="wide")

# Custom CSS for buttons
st.markdown("""
    <style>
    /* General button styles */
    .stButton>button {
        color: white !important;
        font-size: 16px !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        border: none !important;
        transition: background-color 0.3s ease, transform 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #005F5F !important;
        transform: scale(1.05) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for chats and current chat
if 'chats' not in st.session_state:
    st.session_state.chats = {}

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None

if 'next_chat_id' not in st.session_state:
    st.session_state.next_chat_id = 0

# Function to create a new chat
def create_new_chat():
    new_chat_id = st.session_state.next_chat_id
    st.session_state.chats[new_chat_id] = {"messages": [], "title": "Buddy, Your chat appears here!"}
    st.session_state.current_chat_id = new_chat_id
    st.session_state.next_chat_id += 1

# Function to delete all chat history
def delete_all_chats():
    st.session_state.chats = {}
    st.session_state.current_chat_id = None
    st.session_state.next_chat_id = 0
    create_new_chat()  # Create a new default chat after deletion

# Create a default chat if no chats exist
if not st.session_state.chats:
    create_new_chat()

# Sidebar for chat history and new chat button
st.sidebar.title("💬 Chat History")
if st.sidebar.button("➕  New Chat"):
    create_new_chat()

# Display chat history buttons
for chat_id, chat_data in st.session_state.chats.items():
    chat_title = chat_data["title"]
    if st.sidebar.button(chat_title, key=f"chat_{chat_id}"):
        st.session_state.current_chat_id = chat_id

# Add a delete button at the bottom of the sidebar
st.sidebar.markdown("---")  # Add a separator

# Use a session state variable to track the delete confirmation state
if 'show_delete_confirmation' not in st.session_state:
    st.session_state.show_delete_confirmation = False

# Delete button with custom key for CSS targeting
if st.sidebar.button("🗑️ Delete All History"):
    st.session_state.show_delete_confirmation = True

if st.session_state.show_delete_confirmation:
    st.sidebar.warning("Are you sure you want to delete all chat history?")
    if st.sidebar.button("Yes, delete all history"):
        delete_all_chats()
        st.session_state.show_delete_confirmation = False
        st.rerun()  # Refresh the app to reflect changes
    if st.sidebar.button("Cancel"):
        st.session_state.show_delete_confirmation = False

# Main chat interface
st.markdown("# 🎓 StudyBuddy Bot 😎")

# Ensure current_chat_id is always set
if st.session_state.current_chat_id is not None:
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]

    # Display chat messages
    for message in messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message['content'])
        elif message["role"] == "assistant" and message['content'].strip():
            st.chat_message("assistant").markdown(f"{message['content']}")

    # Chat input for user to ask questions
    user_input = st.chat_input("What can I help you learn today?")

    if user_input:
        # Add user message to current chat
        current_chat["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        with st.spinner("Thinking..."):
            response = run_chatbot(user_input)
            if response:
                response_text = str(response) if hasattr(response, 'strip') else str(response)
                current_chat["messages"].append({"role": "assistant", "content": response_text})
                st.chat_message("assistant").markdown(response_text)

        # Update chat title if it's still "Buddy, Your chat history here!"
        if current_chat["title"] == "Buddy, Your chat appears here!":
            current_chat["title"] = user_input[:30] + ("..." if len(user_input) > 30 else "")