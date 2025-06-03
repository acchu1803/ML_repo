import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv(r"C:\Users\raksh\OneDrive\Desktop\ml lab programs\ml\datasets\housing.csv")

# Stratified split based on income category
data["income_cat"] = pd.cut(data["median_income"], [0., 1.5, 3., 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(data, data["income_cat"]):
    train_set = data.loc[train_idx].drop("income_cat", axis=1)
    test_set = data.loc[test_idx].drop("income_cat", axis=1)

# Prepare features and labels
y_train = train_set["median_house_value"]
y_test = test_set["median_house_value"]
X_train = train_set.drop(["median_house_value", "ocean_proximity"], axis=1)
X_test = test_set.drop(["median_house_value", "ocean_proximity"], axis=1)

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train and evaluate model
model = LinearRegression()
model.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print("RMSE:", rmse)
