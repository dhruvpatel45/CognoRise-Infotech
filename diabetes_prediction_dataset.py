import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  

file_path = 'c:/Users/Dhruv Patel/OneDrive/Desktop/diabetes_prediction_dataset.csv'  
data = pd.read_csv(file_path)

features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
target = 'diabetes'

X = data[features]
y = data[target]
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

models = {
    "Random Forest Classifier": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "AdaBoost Classifier": AdaBoostClassifier(algorithm='SAMME'),
    "Decision Tree Classifier": DecisionTreeClassifier(),
}

for model_name, model in models.items():
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred) 
    report = classification_report(y_test, y_pred)  
    print(f"\nResults for {model_name}:")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
