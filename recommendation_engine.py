import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from langchain_ollama import OllamaLLM  # Ollama Mistral
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load assets
movie_embeddings = torch.tensor(np.load('assets/movie_embeddings.npy')).to(device)
df_meta = pd.read_csv('assets/movie_metadata.csv')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
emotion_embeddings = torch.load('assets/emotion_embeddings.pt', weights_only=False)

EMOTION_TAG_MAP = {
    'SAD': ['sad','tragedy','melancholy','depressing','tearjerker','heartbreak'],
    'HAPPY': ['happy','feel-good','uplifting','funny','comedy','heartwarming'],
    'MOTIVATION': ['inspiring','uplifting','empowering','motivational','success','achievement'],
    'ACTION': ['action','thriller','fast-paced','intense','explosions','combat'],
    'ROMANCE': ['romantic','love','relationship','passion','rom-com','affection'],
    'THRILL': ['suspense','thriller','mystery','twist','investigation'],
    'HORROR': ['horror','scary','terror','creepy','paranormal','slasher'],
    'ADVENTURE': ['adventure','journey','quest','fantasy','exploration','discovery'],
    'FAMILY': ['family','children','kids','heartwarming','fun','bonding'],
    'SCI-FI': ['science fiction','sci-fi','space','future','technology','aliens'],
    'FANTASY': ['fantasy','magic','mythical','legend','epic','heroic'],
    'DOCUMENTARY': ['documentary','history','real','true story','educational'],
    'THERAPY': ['therapy','healing','mental health','trauma','recovery']
}

# LLM Setup (Ollama Mistral for RAG)
llm = None
try:
    llm = OllamaLLM(model="mistral")  # Uses your Ollama Mistral
    print("✅ Ollama Mistral loaded for RAG.")
except Exception as e:
    print(f"WARNING: Ollama not available ({e}). Using template fallback.")
    llm = None

def generate_movie_outline(movie_title, plot_synopsis):
    """Generate a concise 1-2 sentence outline using LLM (spoiler-free teaser)."""
    if not llm:
        # Fallback: Shorten raw plot to 100 chars
        return plot_synopsis[:100] + "..." if len(plot_synopsis) > 100 else plot_synopsis
    
    prompt = PromptTemplate(
        input_variables=["title", "plot"],
        template="""Create a very brief 1-2 sentence outline of the movie '{title}' based on this plot summary. Keep it spoiler-free and engaging, like a teaser to know what the movie is about.

Plot: {plot}

Outline:"""
    )
    chain = prompt | llm
    try:
        outline = chain.invoke({"title": movie_title, "plot": plot_synopsis}).strip()
        return outline
    except Exception as e:
        print(f"Outline generation failed for {movie_title}: {e}")
        return plot_synopsis[:100] + "..."

def get_emotion_recommendations(user_input, top_k=5):
    """Hybrid Semantic + Emotion Matching with Fuzzy Tags"""
    user_embedding = model.encode(user_input, convert_to_tensor=True).to(device)

    # Detect top 1-2 emotions
    emotion_scores = {e: util.cos_sim(user_embedding, embeds).max().item()
                      for e, embeds in emotion_embeddings.items()}
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    selected_emotions = [e for e, s in sorted_emotions[:2]]

    required_tags = [tag for e in selected_emotions for tag in EMOTION_TAG_MAP[e]]

    # Semantic similarity
    cosine_scores = util.cos_sim(user_embedding, movie_embeddings)[0]
    results_df = df_meta.copy()
    results_df['similarity_score'] = cosine_scores.cpu().numpy()

    # Emotion match with fuzzy logic
    def emotion_tag_score(tags_str):
        tags_lower = str(tags_str).lower().split(', ')
        score = 0.0
        for tag in required_tags:
            for t in tags_lower:
                if SequenceMatcher(None, tag, t).ratio() >= 0.7:
                    score += 0.5
        return score / len(required_tags) if required_tags else 0

    results_df['emotion_match_score'] = results_df['tags'].apply(emotion_tag_score)

    # Combine scores
    alpha = 0.6
    beta = 0.4
    results_df['final_score'] = alpha*results_df['similarity_score'] + beta*results_df['emotion_match_score']

    top_results = results_df.sort_values(by='final_score', ascending=False).head(top_k)
    return top_results[['title','imdb_id','plot_synopsis','tags','final_score']].to_dict('records'), selected_emotions

def full_rag_recommendation(user_input, top_k=5):
    """RAG with Ollama Mistral for user interaction and outlines"""
    retrieved, emotions = get_emotion_recommendations(user_input, top_k=top_k)
    if not retrieved:
        return "Sorry, no matches found!", []

    # Generate outlines for top movies
    for rec in retrieved[:top_k]:
        rec['outline'] = generate_movie_outline(rec['title'], rec['plot_synopsis'])

    if llm:
        # Use Mistral for creative explanations
        top_movies = retrieved[:3]
        context = "\n".join([f"Title: {m['title']} | Outline: {m['outline']}" for m in top_movies])
        
        prompt_template = PromptTemplate(
            input_variables=["user_mood", "context", "emotions"],
            template="""You are MovieBot AI, a fun, empathetic movie recommender. User mood: {user_mood}. Emotions: {emotions}. 

Recommend 2-3 movies from the context. For each, explain in 1 sentence why it matches the mood (short, positive). End with 'What else can I help with?'.

Context: {context}

Recommendations:"""
        )
        chain = (
            {"user_mood": RunnablePassthrough(), "context": RunnablePassthrough(), "emotions": RunnablePassthrough()}
            | prompt_template
            | llm
        )
        response = chain.invoke({"user_mood": user_input, "context": context, "emotions": ', '.join(emotions)})
        return response.strip(), retrieved
    else:
        # Template fallback
        explanation_text = f"Here are top {len(retrieved)} recommendations for your mood: {', '.join(emotions)}\n"
        for m in retrieved:
            explanation_text += f"- {m['title']}: {m['outline']}\n"
        return explanation_text, retrieved


