import argparse
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_preprocessing import load_and_clean_data, preprocess_data

def plot_feature_importance(model, preprocessor, label_encoder, output_dir):
    """Generates explainability plots, dynamically pulling names from ColumnTransformer."""
    feature_names = preprocessor.get_feature_names_out()
    
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'coef_'):
        class_idx = 0
        target_class_name = label_encoder.inverse_transform([class_idx])[0]
        coefs = model.coef_[class_idx]
        
        if hasattr(coefs, "toarray"):
            coefs = coefs.toarray()[0]
            
        top_positive_indices = np.argsort(coefs)[-15:]
        sns.barplot(
            x=coefs[top_positive_indices], 
            y=feature_names[top_positive_indices], 
            hue=feature_names[top_positive_indices], 
            palette="viridis", 
            legend=False
        )
        plt.title(f"Top 15 Feature Coefficients for Genre: '{target_class_name}'")
        plt.xlabel("Coefficient Value")
        
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]
        
        sns.barplot(
            x=importances[indices], 
            y=feature_names[indices], 
            hue=feature_names[indices], 
            palette="mako", 
            legend=False
        )
        plt.title(f"Top 15 Global Feature Importances ({type(model).__name__})")
        plt.xlabel("Gini Importance")
        
    else:
        print("Model doesn't have recognizable feature importance attributes.")
        return
        
    plt.ylabel("Engineered Features")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path)
    print(f"Explainability plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained genre prediction model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model.pkl")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory containing artifacts")
    
    args = parser.parse_args()
    
    print("Loading test data and artifacts...")
    df = load_and_clean_data(args.data_path)
    
    # FIXED: Added a 5th variable unpack placeholder (_) to match preprocess_data's return signature
    _, X_test, _, y_test, _ = preprocess_data(df, seed=args.seed, is_training=False, output_dir=args.output_dir)
    
    print("Loading model and pipeline artifacts...")
    model = joblib.load(args.model_path)
    preprocessor = joblib.load(os.path.join(args.output_dir, "preprocessor.pkl"))
    label_encoder = joblib.load(os.path.join(args.output_dir, "label_encoder.pkl"))
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*40)
    print(f"MODEL METRICS: {type(model).__name__}")
    print("="*40)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print("\nDetailed Classification Report:")
    
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = label_encoder.inverse_transform(unique_labels)
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
    print("="*40 + "\n")
    
    print("Generating explainability visualizations...")
    plot_feature_importance(model, preprocessor, label_encoder, args.output_dir)

if __name__ == "__main__":
    main()