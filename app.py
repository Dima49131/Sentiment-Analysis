# app.py
from flask import Flask, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)
data = {
    'review': [
        "I love this product!",
        "This is the worst experience ever.",
        "Absolutely fantastic! Highly recommend!",
        "Not worth the money.",
        "I am very satisfied with my purchase.",
        "I'm such a big fan this is amazing!",
        "Terrible customer service.",
        "Exceeded my expectations!",
        "Would never buy this again.",
        "Great value for the price!",
        "Very poor quality, I'm disappointed.",
        "Amazing quality, very happy with this!",
        "Product broke within a week.",
        "Fast delivery and excellent service.",
        "Do not recommend this at all.",
        "My favorite purchase of the year!",
        "Complete waste of money.",
        "So useful and works like a charm!",
        "Regretting this purchase.",
        "Would buy again, loved it!",
        "Worst decision ever, avoid!",
        "My friends loved this gift.",
        "It does exactly what it says, very pleased!",
        "Save your money, this is junk.",
        "Perfect fit and just what I needed.",
        "It was a disappointment overall.",
        "Highly effective, changed my daily routine!",
        "Not worth the hype.",
        "Impressed by the quality and performance!",
        "Terrible experience, will not return.",
        "Totally worth every penny!",
        "It broke after just one use.",
        "Fantastic customer support!",
        "Poorly made and unsatisfactory.",
        "Delighted with my purchase!",
        "I feel cheated with this product.",
        "Amazing, will definitely recommend!",
        "One of the worst purchases I've made.",
        "Top-notch product, exceeded expectations!",
        "Horrible design, very inconvenient.",
        "Love it, exactly as described.",
        "Waste of money, low quality.",
        "Pleasantly surprised by this item!",
        "Cheap materials, fell apart quickly.",
        "Perfect gift for anyone!",
        "Doesn’t live up to the promises.",
        "Outstanding product, couldn't be happier!",
        "Extremely disappointing experience.",
        "Glad I bought it, makes life easier.",
        "Would never recommend to anyone.",
        "An essential addition to my collection!",
        "Fails to deliver on quality."
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative', 'positive', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative'
    ]
}
print(len(data['review']))
print(len(data['sentiment']))
df = pd.DataFrame(data)
X = df['review']
y = df['sentiment']
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(vectorizer.transform(X_train), y_train)
X_all_bow = vectorizer.transform(X)
y_pred_all = model.predict(X_all_bow)
results_df = pd.DataFrame({
    'Review': X,
    'Predicted Sentiment': y_pred_all,
    'Actual Sentiment': y
})
results_df = results_df.sort_values(by='Predicted Sentiment', ascending=False).reset_index(drop=True)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/results')
def results():
    return jsonify(results_df.to_dict(orient='records'))
if __name__ == '__main__':
    app.run(debug=True)
