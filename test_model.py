import pickle

print("Loading model...")

with open("smartphone_price_model.pkl", "rb") as f:
    bundle = pickle.load(f)

print("MODEL LOADED SUCCESSFULLY!")
print("Model:", bundle["model_name"])
print("Numeric columns:", len(bundle["numeric_columns"]))
print("Categorical columns:", len(bundle["categorical_columns"]))