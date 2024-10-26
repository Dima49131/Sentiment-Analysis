import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    'review': [
        "I love this product!",
        "This is the worst experience ever.",
        "Absolutely fantastic! Highly recommend!",
        "Not worth the money.",
        "I am very satisfied with my purchase.",
        "Deez Nuts"
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)
X = df['review']
y = df['sentiment']
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_bow = vectorizer.transform(X_train)
model = MultinomialNB()
model.fit(X_train_bow, y_train)
X_all_bow = vectorizer.transform(X)  # Transform all reviews to bag-of-words
y_pred_all = model.predict(X_all_bow)  # Predict sentiment for all reviews
results_df = pd.DataFrame({
    'Review': X,
    'Predicted Sentiment': y_pred_all,
    'Actual Sentiment': y
})
print("\nResults for all reviews:")
print(results_df)
