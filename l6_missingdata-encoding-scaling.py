import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

file_path= pd.read_csv(r"C:\Users\raksh\OneDrive\Desktop\ml lab programs\ml\datasets\housing.csv")
print(file_path.info())

num_data = file_path.drop("ocean_proximity", axis=1)
num_data = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(num_data), columns=num_data.columns)
print("\nNumeric data after imputation:\n", num_data.info())

cat_encoded = OneHotEncoder().fit_transform(file_path[["ocean_proximity"]]).toarray()
print("\nOne-hot encoded (first 10 rows):\n", cat_encoded[:10])

scaled = pd.DataFrame(StandardScaler().fit_transform(num_data), columns=num_data.columns)
print("\nStandardized data (first 5 rows):\n", scaled.head())
