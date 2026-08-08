import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_and_clean_data(data_path):
    """Loads dataset, consolidates genres, and engineers features."""
    df = pd.read_csv(data_path)
    
    df = df.dropna(subset=['listed_in'])
    df['primary_genre'] = df['listed_in'].apply(lambda x: str(x).split(',')[0].strip())
    
    genre_mapping = {
        'Dramas': 'Drama', 'Drama': 'Drama', 'TV Dramas': 'Drama', 
        'Comedies': 'Comedy', 'Comedy': 'Comedy', 'TV Comedies': 'Comedy', 'Stand-Up Comedy': 'Comedy',
        'Action & Adventure': 'Action', 'Action-Adventure': 'Action', 'TV Action & Adventure': 'Action',
        'Documentaries': 'Documentary', 'Documentary': 'Documentary', 'Docuseries': 'Documentary',
        'Children & Family Movies': 'Family/Kids', "Kids' TV": 'Family/Kids', 'Family': 'Family/Kids', 'Animation': 'Family/Kids',
        'Crime TV Shows': 'Crime/Thriller', 'Thrillers': 'Crime/Thriller', 'TV Thrillers': 'Crime/Thriller', 'Crime': 'Crime/Thriller',
        'Horror Movies': 'Horror', 'TV Horror': 'Horror'
    }
    df['grouped_genre'] = df['primary_genre'].map(genre_mapping)
    df = df.dropna(subset=['grouped_genre'])
    df['primary_genre'] = df['grouped_genre']
    
    df['text_feature'] = df['title'].fillna('') + " " + df['description'].fillna('')
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float).fillna(0)
    df['has_director'] = df['director'].notna().astype(int)
    df['has_cast'] = df['cast'].notna().astype(int)
    
    df['country_main'] = df['country'].str.split(',').str[0].fillna('Unknown')
    top_countries = df['country_main'].value_counts().nlargest(10).index
    df['country_main'] = df['country_main'].where(df['country_main'].isin(top_countries), 'Other')
    df['rating'] = df['rating'].fillna('Unknown')
    
    return df

def preprocess_data(df, seed=42, is_training=True, output_dir="outputs"):
    """Applies ColumnTransformer to handle mixed data types."""
    features = ['text_feature', 'type', 'country_main', 'rating', 'release_year', 'duration_num', 'has_director', 'has_cast']
    X_raw = df[features]
    y_raw = df['primary_genre']
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=seed, stratify=y_raw
    )
    
    if is_training:
        numeric_features = ['release_year', 'duration_num', 'has_director', 'has_cast']
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        categorical_features = ['type', 'country_main', 'rating']
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        text_feature = 'text_feature'
        # Reduced max_features to 1000 based on word count EDA
        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2)))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('text', text_transformer, text_feature)
            ])
            
        label_encoder = LabelEncoder()
        
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
        
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
        
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
        joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
        
        return X_train, X_test, y_train, y_test, y_train_raw
    else:
        preprocessor = joblib.load(os.path.join(output_dir, "preprocessor.pkl"))
        label_encoder = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))
        
        X_test = preprocessor.transform(X_test_raw)
        y_test = label_encoder.transform(y_test_raw)
        
        return None, X_test, None, y_test, None