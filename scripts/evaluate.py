import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('iris.csv')
X = df.drop('target', axis=1)
y = df['target']

model = joblib.load('model.joblib')
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print("Accuracy:", acc)
