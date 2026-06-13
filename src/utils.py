import os
import joblib

def save_model(model, filename):
    os.makedirs("../results", exist_ok=True)
    joblib.dump(model, f"../results/{filename}")
    print(f"Model saved: results/{filename}")

def load_model(filename):
    return joblib.load(f"../results/{filename}")