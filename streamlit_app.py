import streamlit as st
import pandas as pd
from recommendation_engine import full_rag_recommendation

# Page setup
st.set_page_config(page_title="MovieBot AI", page_icon="🎬", layout="wide")
st.sidebar.title("🎬 MovieBot AI")
st.sidebar.button("➕ New Chat", on_click=lambda: st.session_state.clear(), use_container_width=True)

# Chat memory
def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm MovieBot AI. What's your mood today? (e.g., 'sad', 'excited', 'romantic')"}
    ]

if "messages" not in st.session_state:
    clear_chat()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list):
            st.markdown("### 🎥 Top 5 Recommendations:")
            for rec in msg["content"][:5]:
                st.markdown(f"**{rec['title']}**")
                st.markdown(f"*{rec.get('outline', 'Outline not available.')}*")
                st.markdown("---")
        else:
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your mood here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Fetching recommendations..."):
            try:
                response_text, raw_recs = full_rag_recommendation(prompt, top_k=5)
                st.markdown(response_text)
                
                if raw_recs:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.session_state.messages.append({"role": "assistant", "content": raw_recs})
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Error generating recommendations."})

st.markdown("---")

