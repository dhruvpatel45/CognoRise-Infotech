import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

nltk.download('stopwords') 

file_path = 'c:/Users/Dhruv Patel/OneDrive/Desktop/spam.csv'  
data = pd.read_csv(file_path)
features_column = 'Message'  
target_column = 'Category'  

X = data[features_column]  
y = data[target_column]    
y = y.map({'ham': 0, 'spam': 1})
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

X_tfidf = tfidf_vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest Classifier": RandomForestClassifier(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

for model_name, model in models.items():
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)  

    accuracy = accuracy_score(y_test, y_pred) 
    report = classification_report(y_test, y_pred) 

    print(f"\nResults for {model_name}:")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
