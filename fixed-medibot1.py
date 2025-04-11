import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
import requests
import json

# Load environment variables
load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"

# Initialize APIs
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Custom CSS for dark theme and fixed input
st.markdown("""
<style>
    :root {
        --primary: #2A5C82;
        --secondary: #5DA9E9;
        --background: #0F172A;
        --surface: #1E293B;
        --text: #F1F5F9;
        --accent: #FF6B6B;
        --input-bg: #334155;
    }

    html, body, #root, .stApp {
        height: 100vh;
        overflow: auto;
    }

     .stApp {
        padding-bottom: 100px !important;
    }
    
    .fixed-input {
        position: fixed !important;
        bottom: 0;
        left: 0;
        right: 0;
        background: #1E293B;
        padding: 1rem;
        z-index: 9999;
        border-top: 1px solid #334155;
        box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-messages {
        padding-bottom: 150px !important;
    }
            
    .input-container {
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        gap: 8px;
        align-items: center;
    }

    .chat-input {
        flex-grow: 1;
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        border: 1px solid #334155 !important;
    }

    .stChatFloatingInputContainer {
        z-index: 10000 !important;
    }

    footer {
        display: none !important;
    }

    .stStatusWidget>div {
        background: var(--surface) !important;
        border: 1px solid #334155 !important;
            
    }
    /* Model indicator tags */
    .model-indicator {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 8px;
        vertical-align: middle;
    }
    .mistral-indicator {
        background-color: #7b2ff7;
        color: white;
    }
    .gemini-indicator {
        background-color: #047857;
        color: white;
    }
    .deepseek-indicator {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    condense_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    Chat History: {chat_history}
    Follow-Up Question: {question}
    Standalone question:"""
    condense_prompt = PromptTemplate.from_template(condense_template)
   
    # Improved prompt template for medical responses
    qa_template = """As a medical professional, provide accurate and concise information to the patient's question. 
    Structure your response with clear bullet points (*) and use appropriate medical terminology.

    Chat History: {chat_history}
    Context: {context}
    Question: {question}

    Answer guidelines:
    1. Be precise and professional
    2. Use bullet points for clarity
    3. Reference sources when available
    4. If unsure, state limitations
    5. Recommend professional care when appropriate

    Answer:"""
    qa_prompt = PromptTemplate.from_template(qa_template)
   
    return condense_prompt, qa_prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        client=InferenceClient(model=huggingface_repo_id, token=HF_TOKEN),
        model_kwargs={"max_length":"512"}
    )

def query_deepseek(prompt, chat_history, context=None, sources=None):
    """Query DeepSeek via OpenRouter API"""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",  # OpenRouter API key
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-name.com",  # Required by OpenRouter
            "X-Title": "MedAssistant"  # Your app name
        }

        # Format messages with chat history
        messages = [{"role": "system", "content": "You are a medical assistant. Provide accurate, concise answers."}]
        
        # Add previous chat history
        for i in range(0, len(chat_history), 2):
            if i < len(chat_history):
                messages.append({"role": "user", "content": chat_history[i]})
            if i + 1 < len(chat_history):
                messages.append({"role": "assistant", "content": chat_history[i + 1]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "deepseek/deepseek-v3-base:free",  # Must match OpenRouter's model name
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",  # OpenRouter endpoint
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            return {
                "answer": f"API Error (HTTP {response.status_code}): {response.text}",
                "sources": sources or []
            }

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return {
            "answer": answer.replace("•", "*"),  # Standardize bullet points
            "sources": sources or []
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": sources or []
        }

def query_gemini(prompt, chat_history, context=None, sources=None):
    """Gemini query with proper source handling"""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
        
        # Format source references
        source_refs = ""
        if sources:
            source_refs = "\n\nReference these documents:\n" + "\n".join(
                f"[Doc {idx+1}] {doc['title']} (Page {doc['page']})"
                for idx, doc in enumerate(sources)
            )
        
        full_prompt = f"""**Medical Response Task**
        
        **Question:** {prompt}
        
        **Relevant Context:**
        {context if context else "No specific context provided"}
        
        {source_refs}
        
        **Chat History:**
        {"\n".join(chat_history)}
        
        **Response Requirements:**
        - Use medical terminology
        - Structure with bullet points (*)
        - Reference documents like [Doc 1] when applicable
        - If context is insufficient, state "Based on general medical knowledge"
        - Include all relevant details from context
        
        **Answer:**"""
        
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.9,
                "max_output_tokens": 1000
            }
        )
        
        # Process response safely
        if not response.candidates:
            return {
                "answer": "Error: No response generated",
                "sources": sources if sources else []
            }
            
        if response.parts:
            answer = response.text
        else:
            answer = "".join(part.text for part in response.candidates[0].content.parts)
        
        # Standardize formatting
        answer = answer.replace('•', '*').strip()
        
        return {
            "answer": answer,
            "sources": sources if sources else []
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": sources if sources else []
        }

def record_audio():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            with st.status("🎤 Listening...", expanded=True, state="running") as status:
                audio = recognizer.listen(source)
                status.update(label="✅ Listening complete", state="complete")
            return audio
    except Exception as e:
        st.error(f"Microphone error: {str(e)}")
        return None

def speech_to_text(audio):
    try:
        return sr.Recognizer().recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Audio service error"

def main():
    st.title("⚕️ MedAssistant")
    st.markdown("Your intelligent healthcare companion - *Evidence-based medical insights*")

    # Model selection
    model_choice = st.sidebar.radio(
        "Select AI Model",
        ["Mistral-7B (RAG)", "Gemini Pro", "DeepSeek"],
        index=0,
        help="Mistral uses your medical documents, Gemini/DeepSeek use general knowledge"
    )

    # Initialize session states
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'voice_prompt' not in st.session_state:
        st.session_state.voice_prompt = ""

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            # Add model indicator for assistant messages
            if message['role'] == 'assistant':
                model_class = ""
                if message.get('model') == "Mistral":
                    model_class = "mistral-indicator"
                elif message.get('model') == "Gemini":
                    model_class = "gemini-indicator"
                elif message.get('model') == "DeepSeek":
                    model_class = "deepseek-indicator"
                
                model_name = message.get('model', 'AI')
                st.markdown(
                    f'<span class="model-indicator {model_class}">{model_name}</span>',
                    unsafe_allow_html=True
                )
            
            # Display message content
            st.markdown(message['content'])
            
            # Show sources if available
            if message.get('sources'):
                with st.expander("📚 Source Documents", expanded=False):
                    for src in message['sources']:
                        st.markdown(f"**{src.get('title', 'Untitled Document')}**")
                        st.markdown(f"- File: `{src.get('file', 'Unknown')}`")
                        st.markdown(f"- Page: {src.get('page', 'N/A')}")
                        st.markdown("**Excerpt:**")
                        st.markdown(f'<div class="source-content">{src.get("content", "")}</div>',
                                  unsafe_allow_html=True)
                        st.markdown("---")
    
    # Fixed input controls at bottom
    st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
    input_container = st.container()
    
    # Initialize audio variable
    audio = None
    
    with input_container:
        cols = st.columns([0.82, 0.1, 0.08])
        with cols[0]:
            prompt = st.chat_input("Ask your medical question...", key="chat_input")
        with cols[1]:
            mic_clicked = st.button("🎤", key="mic_btn", help="Start voice input")
        with cols[2]:
            stop_clicked = st.button("⏹️", key="stop_btn", help="Stop current response")
    st.markdown('</div>', unsafe_allow_html=True)

    # Handle voice input
    if mic_clicked:
        audio = record_audio()
        if audio:
            st.session_state.voice_prompt = speech_to_text(audio)
            st.rerun()

    # Use voice prompt if available
    if st.session_state.voice_prompt:
        prompt = st.session_state.voice_prompt
        st.session_state.voice_prompt = ""

    if prompt:
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            relevant_docs = vectorstore.similarity_search(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            sources = [{
                'file': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'title': doc.metadata.get('title', 'Untitled Document'),
                'content': doc.page_content
            } for doc in relevant_docs]

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Analyzing..."):
                    if model_choice == "Mistral-7B (RAG)":
                        # Existing RAG implementation
                        memory = ConversationBufferMemory(
                            memory_key="chat_history",
                            return_messages=True,
                            output_key="answer"
                        )
                        
                        for i in range(0, len(st.session_state.chat_history), 2):
                            if i+1 < len(st.session_state.chat_history):
                                memory.chat_memory.add_user_message(st.session_state.chat_history[i])
                                memory.chat_memory.add_ai_message(st.session_state.chat_history[i+1])

                        condense_prompt, qa_prompt = set_custom_prompt()
                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=load_llm("mistralai/Mistral-7B-Instruct-v0.3", os.getenv("HF_TOKEN")),
                            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                            memory=memory,
                            condense_question_prompt=condense_prompt,
                            combine_docs_chain_kwargs={"prompt": qa_prompt},
                            return_source_documents=True
                        )
                        
                        response = qa_chain({"question": prompt})
                        result = {
                            "answer": response["answer"],
                            "sources": sources
                        }
                        
                    elif model_choice == "Gemini Pro":
                        result = query_gemini(prompt, st.session_state.chat_history, context, sources)
                    else:  # DeepSeek
                        result = query_deepseek(prompt, st.session_state.chat_history, context, sources)
                    
                    # Display response
                    message_placeholder.markdown(result["answer"])
                    
                    # Store response
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': result["answer"],
                        'sources': result["sources"],
                        'model': model_choice.split()[0]  # Gets "Mistral", "Gemini", or "DeepSeek"
                    })
                    
                    # Update chat history
                    st.session_state.chat_history.extend([prompt, result["answer"]])
                    
                    # Show sources if available
                    if result["sources"]:
                        with st.expander("📚 Source Documents", expanded=False):
                            for src in result["sources"]:
                                st.markdown(f"**{src['title']}**")
                                st.markdown(f"- File: `{src['file']}`")
                                st.markdown(f"- Page: {src['page']}")
                                st.markdown("**Excerpt:**")
                                st.markdown(f'<div class="source-content">{src["content"]}</div>',
                                          unsafe_allow_html=True)
                                st.markdown("---")

        except Exception as e:
            st.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()