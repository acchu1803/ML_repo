import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\raksh\OneDrive\Desktop\ml lab programs\ml\datasets\housing.csv")
print(data)

data["median_income"].hist()
data.plot.scatter(x="longitude", y="latitude", alpha=0.1)
data.plot.scatter(x="longitude", y="latitude", alpha=0.4,
                  s=data["population"]/100, c="median_house_value",
                  cmap="jet", colorbar=True)
plt.show()

print("\nBefore:\n", data.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False))

data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]

print("\nAfter:\n", data.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False))

data.plot.scatter(x="median_income", y="median_house_value", alpha=0.1)
plt.show()
