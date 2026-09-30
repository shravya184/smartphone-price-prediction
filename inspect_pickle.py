import pickletools

path = "smartphone_price_model.pkl"

print("=" * 70)
print("PICKLE INSPECTION")
print("=" * 70)

with open(path, "rb") as f:
    data = f.read()

print("Pickle size:", len(data), "bytes")
print()

# Print all sklearn-related strings found in the pickle
text = data.decode("latin1", errors="ignore")

keywords = [
    "sklearn",
    "GradientBoosting",
    "RandomForest",
    "XGB",
    "Ridge",
    "Lasso",
    "LinearRegression",
    "DecisionTree",
    "_loss",
]

print("IMPORTANT REFERENCES FOUND:")
print("-" * 70)

found = False

for keyword in keywords:
    if keyword in text:
        print("FOUND:", keyword)
        found = True

if not found:
    print("No known sklearn model names found.")

print()
print("=" * 70)
print("PICKLE GLOBAL REFERENCES")
print("=" * 70)

try:
    for opcode, arg, pos in pickletools.genops(data):
        if opcode.name in ("GLOBAL", "STACK_GLOBAL"):
            print(f"{opcode.name}: {arg}")
except Exception as e:
    print("Could not fully inspect pickle:", e)

print()
print("=" * 70)
print("DONE")
print("=" * 70)