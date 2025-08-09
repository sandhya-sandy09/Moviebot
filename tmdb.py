import os
import requests
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

GENRE_NAME_TO_ID = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37
}

def search_tmdb_movies(genres,year,rating,language_code = None):
    genre_ids = [GENRE_NAME_TO_ID[g] for g in genres if g in GENRE_NAME_TO_ID]
    min_vote = int(rating.replace("+","")) if rating != "Any" else 0

    year_params = {}
    if year != "Any":
        parts = year.split("-")
        if len(parts) == 2:
            year_params["primary_release_date.gte"] = f"{parts[1]}-01-01"
            year_params["primary_release_date.lte"] = f"{parts[1]}-12-31"

    params = {
        "api_key": TMDB_API_KEY,
        "with_genres": ",".join(map(str, genre_ids)),
        "sort_by": "popularity.desc",
        "vote_average.gte": min_vote,
        **year_params
    }
    if language_code and language_code != "Any":
        params["with_original_language"] = language_code

    response = requests.get("https://api.themoviedb.org/3/discover/movie", params=params)
    data = response.json()

    results = []
    for movie in data.get("results", [])[:5]:
        results.append({
            "title": movie.get("title"),
            "poster": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get("poster_path") else "",
            "rating": movie.get("vote_average"),
            "overview": movie.get("overview")
        })

    return results

