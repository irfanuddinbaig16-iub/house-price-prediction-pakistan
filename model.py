import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


def convert_to_sqft(row):
    if row["AreaType"] == "Marla":
        return row["AreaSize"] * 272.25
    if row["AreaType"] == "Kanal":
        return row["AreaSize"] * 5445
    return np.nan


df = pd.read_csv("data.csv")

df = df[
    ["price", "city", "location", "bedrooms", "baths", "Area Type", "Area Size"]
].copy()

df.rename(
    columns={
        "price": "Price",
        "city": "City",
        "location": "Location",
        "bedrooms": "Bedrooms",
        "baths": "Bathrooms",
        "Area Size": "AreaSize",
        "Area Type": "AreaType",
    },
    inplace=True,
)

df["Area_sqft"] = df.apply(convert_to_sqft, axis=1)

df = df[["Area_sqft", "Bedrooms", "Bathrooms", "City", "Location", "Price"]]
df.dropna(inplace=True)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), ["Area_sqft", "Bedrooms", "Bathrooms"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["City", "Location"]),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

model.fit(X_train, y_train)

joblib.dump(model, "house_model.pkl")
print("Model saved successfully: house_model.pkl")
