import streamlit as st # importing streamlit to make web app
import google.generativeai as genai #Importing Google Generative AI SDK to use Gemini AI model
import os
from dotenv import load_dotenv
from embeddings.pinecone_index import initialize_pinecone, store_query, retrieve_queries
from prompts.prompts import generate_conversation_prompt #added import

# loading .env file to access env variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # fetching GEMINI key
genai.configure(api_key=GOOGLE_API_KEY) # Configuring genai model with api key
model = genai.GenerativeModel('gemini-2.0-flash') # creating instance of gemini model

st.title("Berlin") # Setting title of streamlit model

@st.cache_resource # Ye decorator function ko cache kar dega taake baar-baar re-run na ho.
def setup_resources():
    """Sets up Pinecone."""
    index = initialize_pinecone() # Initializing pinecone index
    return index # returning initilized index

# Setting up pinecone index
index = setup_resources()

# Iitializing messages aur conversation history in streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [] # initializing message list
    st.session_state.conversation_history = [] # iitilaizing conversatio history

for message in st.session_state.messages: # Pehle se stored messages ko UI me show karne ke liye loop chala rahe hain.
    with st.chat_message(message["role"]): # Chat message ka role set kar rahe hain (user/assistant).
        st.markdown(message["content"]) # Message content ko markdown format me show kar rahe hain.

if prompt := st.chat_input("Ask me anything"): # User input ko chat box se fetch kar rahe hain.
    st.session_state.messages.append({"role": "user", "content": prompt}) # User ke message ko session state me store kar rahe hain.
    with st.chat_message("user"): # Chat interface me user ka message dikhane ke liye
        st.markdown(prompt)

    # Conversation history me bhi user ke message ko add kar rahe hain.
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    # Assistant ka response generate karne ke liye chat message block create kar rahe hain.
    with st.chat_message("assistant"): 
        with st.spinner("Generating response..."): # Spinner show kar rahe hain jab tak response generate ho raha hai.
            try:
                # Conversation history aur prompt ka use karke final prompt generate kar rahe hain.
                final_prompt = generate_conversation_prompt(st.session_state.conversation_history, prompt) #using the prompt function
                response = model.generate_content(final_prompt) # Gemini AI model ka use karke response generate kar rahe hain.
                st.markdown(response.text) # Assistant ka response chat me show kar rahe hain.
                # Assistant ke response ko session state me store kar rahe hain.
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.session_state.conversation_history.append({"role": "assistant", "content": response.text})
                store_query(prompt, index) # User ka prompt Pinecone database me store kar rahe hain.
            except Exception as e:
                st.error(f"An error occurred: {e}") # Agar koi error aaye toh error message show kar rahe hain.
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, an error occurred."})
                st.session_state.conversation_history.append({"role": "assistant", "content": "Sorry, an error occurred."})