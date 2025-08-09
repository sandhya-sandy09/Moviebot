import streamlit as st
from streamlit import rerun
from llm import get_movie_genres_from_llm
from tmdb import search_tmdb_movies
from llm import generate_intent_question

st.header("MovieBot")

if "stage" not in st.session_state:
    st.session_state.stage = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.stage == 0:
    user = st.chat_input("What do u feel like watching today")
    if user:
        st.session_state.mood = user
        question = generate_intent_question(user)
        
        st.session_state.messages.append({"role": "user", "content": user})
        st.session_state.messages.append({"role": "assistant", "content": question})
        st.session_state.stage = 1
        st.rerun()

elif st.session_state.stage == 1:
    # Ask what kind of movie they want now (chat_input)
    intent = st.chat_input("What kind of movie do you want right now?")
    if intent:
        st.session_state.intent = intent
        st.session_state.messages.append({"role": "user", "content": intent})
        st.session_state.messages.append({"role": "assistant", "content": "Great! Choose any filters below and click Generate 🎬"})
        st.session_state.stage = 2
        st.rerun()


elif st.session_state.stage == 2:
    ott = st.multiselect("Preferred OTT:",["Any","Disney+ Hotstar", "Amazon Prime Video", "Netflix", "ZEE5", "Sony LIV", "JioCinema", "Aha", "Sun NXT", "Voot", "MX Player"])
    st.write("You selected", len(ott), 'ott')
    year = st.selectbox("Year of release:",["Any","2025 - 2020", "2020 - 2015", "2015 - 2010", "2010-2000","2000-1990","Before 1990", ])
    ratings = st.selectbox("Ratings",["Any", "9+", "8+", "7+", "6+","5+"])
    language_options = {
    "Any": None,
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn"
}
    language_ui = st.selectbox("Preferred Language:", list(language_options.keys()))
    language_code = language_options[language_ui]

           
    if st.button("Generate Response"):
        st.session_state.messages.append({"role": "assistant", "content": "Awesome! Let me fetch some movies for you 🍿"})
        st.session_state.stage = 3
        st.session_state.year = year
        st.session_state.ratings = ratings
        st.session_state.language_code = language_code
        st.rerun()

elif  st.session_state.stage == 3:

    if "genres_fetched" not in st.session_state:
        st.session_state.genres_fetched = False 

    if not st.session_state.genres_fetched:
        genres = get_movie_genres_from_llm(st.session_state.mood,st.session_state.intent)
        st.session_state.messages.append({
        "role": "assistant",
        "content": f"Based on what you said, these genres might suit you best: {', '.join(genres)}"})
        movies = search_tmdb_movies(genres,  st.session_state.year, st.session_state.ratings ,st.session_state.language_code)
        st.session_state.movies = movies
        st.session_state.genres_fetched = True 
        st.rerun()
     
    for movie in st.session_state.movies:
        with st.chat_message("assistant"):
            st.image(movie["poster"], width=100)
            st.markdown(f"### {movie['title']}")
            st.markdown(f"⭐ {movie['rating']}")
            st.markdown(movie["overview"])

    follow_up = st.chat_input("Ask for more or type 'new' to restart")

    if follow_up:
        st.session_state.messages.append({"role": "user", "content": follow_up})

        if "new" in follow_up.lower():
            st.session_state.messages.append({"role": "assistant", "content": "Sure! Let’s start fresh. What's your mood now?"})
            st.session_state.stage = 0

        elif "more" in follow_up.lower():
            st.session_state.messages.append({"role": "assistant", "content": "Here's another great one for you! 🎬"})
            st.session_state.stage = 2

        else:
            st.session_state.messages.append({"role": "assistant", "content": "Got it! Let me know how I can help."})

        st.rerun()

    if st.button("🆕 New Chat"):
        st.session_state.stage = 0
        st.session_state.messages = []
        st.session_state.genres_fetched = False 
        st.rerun()


