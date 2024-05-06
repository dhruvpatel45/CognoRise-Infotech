import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report 
import nltk
from nltk.corpus import stopwords

file_path = "C:/Users/Dhruv Patel/OneDrive/Desktop/IMDB Dataset.csv" 
data = pd.read_csv(file_path)
features_column = 'review'
target_column = 'sentiment'

X = data[features_column]
y = data[target_column]
y = y.map({'negative': 0, 'positive': 1})
X.fillna('', inplace=True)  
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_tfidf = tfidf_vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

models = {
    "Naive Bayes Classifier": MultinomialNB(),
    "Random Forest Classifier": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "XGBoost Classifier": XGBClassifier(),
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  

    accuracy = accuracy_score(y_test, y_pred)  
    report = classification_report(y_test, y_pred) 

    print(f"\nResults for {model_name}:")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
