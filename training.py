import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load data
data = pd.read_csv('diabetes_latih.csv')

# pisah fitur dan target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

with open('knn_pickle', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved to knn_pickle")