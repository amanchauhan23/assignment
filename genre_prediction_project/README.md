# TV Series and Movie Genre Prediction

This repository contains an end-to-end machine learning pipeline to predict the primary genre of a TV series or movie based on its title and description.

## High-Level Approach

Because the raw dataset contains highly fragmented and imbalanced target classes (over 60 unique sub-genres), this project approaches the problem through a multi-modal feature engineering pipeline:

1. **Target Consolidation**: Fragmented labels (e.g., "TV Comedies", "Stand-Up Comedy") are mapped to 7 broad "Super Genres" (Action, Comedy, Drama, Documentary, Family/Kids, Crime/Thriller, Horror) to provide the model with a statistically significant number of samples per class.
2. **Multi-Type Feature Engineering**:
   * **Text (NLP)**: The `title` and `description` are concatenated and vectorized using **TF-IDF** (Top 1000 features, 1-2 n-grams).
   * **Numeric**: Metadata like `duration` (extracted into integers) and `release_year` are scaled using `StandardScaler`.
   * **Categorical**: Binary flags for `has_director`/`has_cast` and nominal features like `type` (Movie/Show) are One-Hot Encoded.
3. **Modeling & Imbalance Handling**: The data is fed into a `ColumnTransformer` and trained on algorithms like Gradient Boosting. Class imbalance is strictly handled by dynamically computing and passing `sample_weights` during training.
4. **Evaluation**: The pipeline evaluates models using Accuracy, Weighted F1-Score, and produces global feature importance visualizations for explainability.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. **train:**
   ```bash
   python src/train.py --data_path data/tv-shows.csv --model gradient_boosting --seed 42
3. **eval:**
   ```bash
   python src/evaluate.py --model_path outputs/model.pkl --data_path data/tv-shows.csv
