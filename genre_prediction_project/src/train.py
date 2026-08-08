import argparse
import os
import joblib
from sklearn.utils.class_weight import compute_sample_weight
from data_preprocessing import load_and_clean_data, preprocess_data
from model import get_model

def main():
    parser = argparse.ArgumentParser(description="Train a genre prediction model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV")
    parser.add_argument("--model", type=str, required=True, help="Options: logistic_regression, random_forest, gradient_boosting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save artifacts")
    
    args = parser.parse_args()
    
    print(f"Loading and feature engineering data from {args.data_path}...")
    df = load_and_clean_data(args.data_path)
    
    print("Executing multi-type preprocessing pipeline...")
    # We now also capture y_train_raw to compute class weights
    X_train, X_test, y_train, y_test, y_train_raw = preprocess_data(df, seed=args.seed, is_training=True, output_dir=args.output_dir)
    
    print(f"Initializing {args.model}...")
    clf = get_model(args.model, args.seed)
    
    print("Computing sample weights to handle class imbalance...")
    weights = compute_sample_weight(class_weight='balanced', y=y_train_raw)
    
    print("Training the model...")
    # Pass the calculated weights during fitting
    clf.fit(X_train, y_train, sample_weight=weights)
    
    model_path = os.path.join(args.output_dir, "model.pkl")
    joblib.dump(clf, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()