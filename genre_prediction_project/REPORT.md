## REPORT.md
```markdown
# Model Evaluation & Explainability Report

## Approach
Because the raw dataset relies heavily on text features to describe the content, we approached this genre prediction task as a Natural Language Processing (NLP) classification problem. 
1. **Target Engineering**: The `listed_in` column often contains multiple comma-separated genres. We extracted the first genre as the primary target class.
2. **Feature Engineering**: We concatenated the `title` and `description` columns into a single text feature to capture maximum contextual information.
3. **Vectorization**: We used TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text into numerical features (top 1000 words).
4. **Models**: We tested two traditional ML models: Logistic Regression and Random Forest Classifier.

## Recommended Model
**Logistic Regression** is the recommended model for this specific pipeline. 
* **Why?** Text data represented via TF-IDF results in a high-dimensional, sparse matrix. Linear models like Logistic Regression are highly robust and efficient for sparse text data compared to tree-based models like Random Forest, which can easily overfit and require significantly more time to train on high-dimensional text data. 
* **Explainability**: Logistic regression coefficients provide direct, easily interpretable feature importance (e.g., the word "murder" highly correlates with "Crime", or "laugh" with "Comedies").

*(Note: Run `evaluate.py` to generate the exact metric scores and the `feature_importance.png` chart to include in your final presentation!)*