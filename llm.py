import ast
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


def generate_intent_question(mood):
    prompt = f"""
    The user says: '{mood}'.
    Based on this, generate a short, conversational question that asks what type of movie experience they want, in a way that feels personalized.
    Don't say 'What kind of movie do you want?'. Be natural and creative.
    Examples:
    - If mood is horror: 'Got it! Do you want something scary, supernatural, or psychological?'
    - If mood is comedy: 'Nice! Want something silly, romantic, or witty?'
    - If mood is action: 'Awesome! Looking for explosions, martial arts, or spy thrillers?'
    - If mood is drama: 'I see! Want something emotional, inspiring, or thought-provoking?'
    - If mood is bored: 'Bored, huh? Want something thrilling, fun, or mind-bending?'
    Return just the question text.
    """

    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates smart, mood-based follow-up questions for a movie chatbot. Always provide options that match the user's stated mood or genre preference."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()


        
def get_movie_genres_from_llm(mood, intent):
    prompt = f"""
    The user says they are feeling '{mood}' and wants a movie that is '{intent}'.
    Based on this, return a plain Python list of 3-5 suitable TMDB movie genres.
    
    Available TMDB genres: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western
    
    Example format only: ["Comedy", "Drama", "Family"]
    Do not add any extra explanation or text. Just return the list.
    
    Focus on matching the intent closely - if they want "scary" or "supernatural", prioritize Horror. If they want "psychological", include Thriller and Mystery.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that maps emotions and user movie intent to TMDB genre categories. Always prioritize the user's specific intent over general mood."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        genre_list = ast.literal_eval(raw_output)
        if isinstance(genre_list, list):
            return genre_list
    except:
        pass

    return []
