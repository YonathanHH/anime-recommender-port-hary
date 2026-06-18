import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
from PIL import Image

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DARK ANIME THEME CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Nunito:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: #0d0d1a;
    color: #e8e8f0;
  }
  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0d0d1a 100%);
  }
  h1, h2, h3 { font-family: 'Rajdhani', sans-serif; color: #c77dff; }
  .stButton > button {
    background: linear-gradient(90deg, #7b2ff7, #f107a3);
    color: white; border: none; border-radius: 10px;
    padding: 0.5rem 2rem; font-weight: 700; font-size: 1rem;
    transition: transform 0.2s;
  }
  .stButton > button:hover { transform: scale(1.05); }
  .stSelectbox > div > div, .stMultiSelect > div > div {
    background-color: #1a1a2e !important;
    border: 1px solid #7b2ff7 !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
  }
  .stSlider > div { color: #c77dff; }
  .sidebar .sidebar-content { background-color: #10102b; }
  div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10102b, #1a0a2e);
    border-right: 1px solid #7b2ff7;
  }
  .anime-card {
    background: rgba(123,47,247,0.08);
    border: 1px solid rgba(123,47,247,0.3);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
    transition: transform 0.2s, border-color 0.2s;
  }
  .anime-card:hover {
    transform: translateY(-4px);
    border-color: #f107a3;
  }
  .score-badge {
    display: inline-block;
    background: linear-gradient(90deg, #7b2ff7, #f107a3);
    color: white; padding: 2px 10px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700;
  }
  .genre-pill {
    display: inline-block;
    background: rgba(199,125,255,0.15);
    border: 1px solid rgba(199,125,255,0.4);
    color: #c77dff; padding: 2px 8px;
    border-radius: 12px; font-size: 0.72rem; margin: 2px;
  }
  .stMetric { background: rgba(123,47,247,0.1); border-radius: 10px; padding: 8px; }
  hr { border-color: rgba(123,47,247,0.3); }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("animes.csv")
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    df[genre_cols] = df[genre_cols].fillna(0.0)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0.0)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    return df, genre_cols

df, GENRE_COLS = load_data()

# Filtered pool (≥50 votes)
df_filtered = df[df["votes"] >= 50].reset_index(drop=True)

GENRE_LABELS = [c.replace("genre_", "").replace("-", " ").title() for c in GENRE_COLS]

# ─── SIMILARITY ENGINE ────────────────────────────────────────────────────────
@st.cache_data
def build_feature_matrix(genre_weight: float = 0.7):
    """Build weighted feature matrix: genre_weight * genre + (1-genre_weight) * norm_rate."""
    genre_matrix = df_filtered[GENRE_COLS].values.astype(float)
    norm_rate = (df_filtered["rate"].values / 5.0).reshape(-1, 1)
    weighted = np.hstack([
        genre_matrix * genre_weight,
        norm_rate * (1 - genre_weight)
    ])
    return weighted

def recommend_by_title(anime_name: str, top_n: int = 10, genre_weight: float = 0.7):
    """Recommend based on a selected anime title."""
    matrix = build_feature_matrix(genre_weight)
    idx = df_filtered[df_filtered["anime"] == anime_name].index
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[0]
    sims = cosine_similarity([matrix[idx]], matrix)[0]
    sim_scores = list(enumerate(sims))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]
    recs = df_filtered.iloc[[i for i, _ in sim_scores]].copy()
    recs["similarity"] = [round(s * 100, 1) for _, s in sim_scores]
    return recs

def recommend_by_genres(selected_genres: list, top_n: int = 10, genre_weight: float = 0.7):
    """Recommend based on selected genre preferences."""
    if not selected_genres:
        return pd.DataFrame()
    query_vec = np.zeros(len(GENRE_COLS) + 1)
    for i, col in enumerate(GENRE_COLS):
        label = col.replace("genre_", "").replace("-", " ").title()
        if label in selected_genres:
            query_vec[i] = genre_weight
    query_vec[-1] = (1 - genre_weight)  # max rating signal
    matrix = build_feature_matrix(genre_weight)
    sims = cosine_similarity([query_vec], matrix)[0]
    sim_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]
    recs = df_filtered.iloc[[i for i, _ in sim_scores]].copy()
    recs["similarity"] = [round(s * 100, 1) for _, s in sim_scores]
    return recs

# ─── IMAGE LOADING ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_img(url: str):
    try:
        r = requests.get(url, timeout=4)
        return Image.open(BytesIO(r.content))
    except:
        return None

# ─── CARD RENDERER ────────────────────────────────────────────────────────────
def render_card(row, rank: int):
    img = load_img(row["anime_img"])
    genres_active = [GENRE_LABELS[i] for i, c in enumerate(GENRE_COLS) if row[c] == 1.0]

    with st.container():
        st.markdown('<div class="anime-card">', unsafe_allow_html=True)
        col_img, col_info = st.columns([1, 3])
        with col_img:
            if img:
                st.image(img, use_container_width=True)
            else:
                st.markdown("🎌")
        with col_info:
            st.markdown(f"### #{rank} [{row['anime']}]({row['anime_url']})")
            pills = " ".join([f'<span class="genre-pill">{g}</span>' for g in genres_active])
            st.markdown(pills, unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⭐ Rate", f"{row['rate']:.2f} / 5")
            c2.metric("🗳️ Votes", f"{row['votes']:,}")
            c3.metric("📺 Episodes", row['episodes'] if row['episodes'] > 0 else "N/A")
            c4.metric("🎯 Match", f"{row['similarity']}%")
        st.markdown('</div>', unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    mode = st.radio(
        "🔍 Recommendation Mode",
        ["By Anime Title", "By Genre Preferences"],
        help="Choose how you want to discover anime"
    )

    st.markdown("---")
    genre_weight = st.slider(
        "🎚️ Genre vs Rating Weight",
        min_value=0.5, max_value=1.0, value=0.7, step=0.05,
        help="Higher = more genre-driven. Lower = more rating-driven."
    )
    st.caption(f"Genre: {genre_weight:.0%} | Rating: {(1-genre_weight):.0%}")

    top_n = st.slider("📋 Number of Recommendations", 5, 20, 10)

    st.markdown("---")
    st.markdown("""<small>
    📊 Dataset: <a href='https://www.kaggle.com/code/mehvishsheikh31/crunchyroll-anime-recommender-eda-tf-idf-shap' target='_blank' style='color:#c77dff;'>Kaggle by Mehvish Sheikh</a><br>
    🛠️ Built by <a href='https://github.com/YonathanHH' target='_blank' style='color:#c77dff;'>Yonathan Hary</a>
    </small>""", unsafe_allow_html=True)

# ─── MAIN HEADER ──────────────────────────────────────────────────────────────
st.markdown("# 🎌 Anime Recommender")
st.markdown("*Discover your next obsession — powered by Weighted Cosine Similarity*")
st.markdown("---")

# ─── QUICK STATS ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Total Anime", f"{len(df):,}")
c2.metric("✅ Filtered Pool", f"{len(df_filtered):,}", "≥50 votes")
c3.metric("🎭 Genres", len(GENRE_COLS))
c4.metric("⭐ Avg Rating", f"{df_filtered['rate'].mean():.2f}")
st.markdown("---")

# ─── MODE: BY TITLE ───────────────────────────────────────────────────────────
if mode == "By Anime Title":
    st.markdown("## 🔎 Find Similar Anime")
    anime_list = sorted(df_filtered["anime"].tolist())
    selected = st.selectbox("Select an anime you enjoyed:", anime_list)

    if selected:
        row = df_filtered[df_filtered["anime"] == selected].iloc[0]
        st.markdown("### 📌 Selected Anime")
        render_card({**row.to_dict(), "similarity": 100.0}, 0)

        st.markdown("---")
        st.markdown(f"### 🎯 Top {top_n} Similar Anime")

        with st.spinner("Finding your recommendations..."):
            results = recommend_by_title(selected, top_n, genre_weight)

        if results.empty:
            st.warning("No recommendations found. Try a different title.")
        else:
            for rank, (_, row) in enumerate(results.iterrows(), 1):
                render_card(row.to_dict(), rank)

# ─── MODE: BY GENRES ──────────────────────────────────────────────────────────
else:
    st.markdown("## 🎭 Explore by Genre")
    selected_genres = st.multiselect(
        "Pick your favourite genres:",
        GENRE_LABELS,
        default=["Action", "Fantasy"]
    )

    if selected_genres:
        st.markdown(f"### 🎯 Top {top_n} Anime matching your genres")
        with st.spinner("Finding your recommendations..."):
            results = recommend_by_genres(selected_genres, top_n, genre_weight)

        if results.empty:
            st.warning("No results. Try different genres.")
        else:
            for rank, (_, row) in enumerate(results.iterrows(), 1):
                render_card(row.to_dict(), rank)
    else:
        st.info("👆 Select at least one genre from the list above to get started.")
