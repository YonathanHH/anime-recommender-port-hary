# 🎌 Anime Recommender — by Hary

A **genre & rating-based anime recommendation system** built with Python and deployed via Streamlit.  
Finds similar anime using **Weighted Cosine Similarity** across 28 binary genre features and weighted rating signals.

---

## 📁 Project Structure

```
anime-recommender-port-hary/
├── app.py                  # Streamlit application
├── animes.csv              # Dataset (Crunchyroll via Kaggle)
├── EDA_and_Recommender.ipynb  # EDA + model notebook
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

```bash
git clone https://github.com/YonathanHH/anime-recommender-port-hary
cd anime-recommender-port-hary
pip install -r requirements.txt
streamlit run app.py
```


## 🧠 How It Works

### Weighted Cosine Similarity

Each anime is represented as a vector:
- **28 binary genre dimensions** (one-hot encoded, already in dataset)
- Augmented with a **normalized rating score** (weight = 0.3) to bias results toward quality

The similarity between anime A and B:

```
sim(A, B) = cosine_similarity(genre_vector_A, genre_vector_B)
final_score = (0.7 × genre_sim) + (0.3 × rating_score_B)
```

This ensures genre match drives the result, while rating acts as a quality tiebreaker.

### Filtering
- Only anime with **≥ 50 votes** are shown in results to avoid obscure/low-quality entries.

## 📊 Dataset

**Source:** [Crunchyroll Anime Recommender — EDA, TF-IDF, SHAP by Mehvish Sheikh on Kaggle](https://www.kaggle.com/code/mehvishsheikh31/crunchyroll-anime-recommender-eda-tf-idf-shap)

The dataset contains 1,255 anime from Crunchyroll with:
- Title, URL, thumbnail image
- Episode count, votes, weighted score, rating (1–5)
- 28 binary genre columns

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| App framework | Streamlit |
| Similarity engine | scikit-learn (cosine_similarity) |
| Data processing | pandas, numpy |
| Visualization | plotly, seaborn, matplotlib |
| EDA | Jupyter Notebook |

## 👤 Author

**Yonathan Hary Hutagalung**  
Sustainable Energy Science MSc | Data Analyst  
[GitHub](https://github.com/YonathanHH)
