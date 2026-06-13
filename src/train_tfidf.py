import sys
sys.path.append('../src')
from sklearn.linear_model import LogisticRegression
from utils import save_model

def train_tfidf_model(X_train, y_train):
    print("Training TF-IDF Model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    save_model(model, "tfidf_model.pkl")
    print("TF-IDF Model Training Complete!")
    return model