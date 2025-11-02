import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
os.makedirs('assets', exist_ok=True)

# Step 1: Load dataset
df = pd.read_csv('data/mpst_full_data.csv')
df.dropna(subset=['title', 'plot_synopsis', 'tags'], inplace=True)
df['title'] = df['title'].astype(str)
df['tags'] = df['tags'].astype(str)

# Step 2: Feature engineering
def create_super_feature(row):
    title = str(row['title'])
    tags = str(row['tags']).replace(', ', ' ')
    plot_snip = str(row['plot_synopsis'])[:500]
    return f"TITLE: {title}. PLOT: {plot_snip}. KEYWORDS: {tags}"

df['super_text_feature'] = df.apply(create_super_feature, axis=1)

# Step 3: Emotion tag map (simplified for speed)
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

# Step 4: Load model & encode emotions
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
emotion_embeddings = {}
for emotion, keywords in EMOTION_TAG_MAP.items():
    emotion_embeddings[emotion] = model.encode(keywords, convert_to_tensor=True, device=device)

torch.save(emotion_embeddings, 'assets/emotion_embeddings.pt')

# Step 5: Encode movies (GPU-optimized)
movie_embeddings = model.encode(df['super_text_feature'].tolist(),
                                convert_to_tensor=True,
                                show_progress_bar=True,
                                batch_size=64 if device=='cuda' else 8)
np.save('assets/movie_embeddings.npy', movie_embeddings.cpu().numpy())

# Save metadata
df[['title','imdb_id','plot_synopsis','tags','super_text_feature']].to_csv('assets/movie_metadata.csv', index=False)
print("✅ Preprocessing complete!")