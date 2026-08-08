from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_model(model_name, seed):
    """Returns an uninitialized scikit-learn model based on the name."""
    if model_name.lower() == 'logistic_regression':
        return LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
    elif model_name.lower() == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
    elif model_name.lower() == 'gradient_boosting':
        # Fine-tuned parameters based on EDA
        return GradientBoostingClassifier(
            n_estimators=300,        # Increased trees for better learning
            learning_rate=0.05,      # Lowered learning rate for stability
            max_depth=5,             # Increased depth to combine text + duration
            subsample=0.8,           # Added stochastic sampling to prevent overfitting
            random_state=seed
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported.")