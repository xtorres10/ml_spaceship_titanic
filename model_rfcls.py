import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("data/clean_with_all_cols.csv")

# Separate target and features
X = data.drop(columns="Transported")
y = data["Transported"]

# Split data: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(f"Size x_train: {X_train.shape[0]}\nSize x_test: {X_test.shape[0]}")

# Model
rf_cls = RandomForestClassifier(n_estimators=30)
rf_cls.fit(X_train)

err_train = np.mean(y_train != rf_cls.predict(X_train))
err_test  = np.mean(y_test  != rf_cls.predict(X_test))

print('Training sample error: ', err_train)
print('Error on the test sample: ', err_test)