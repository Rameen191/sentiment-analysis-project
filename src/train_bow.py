import sys
sys.path.append('../src')
from sklearn.linear_model import LogisticRegression
from utils import save_model

def train_bow_model(X_train, y_train):
    print("Training BoW Model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    save_model(model, "bow_model.pkl")
    print("BoW Model Training Complete!")
    return model