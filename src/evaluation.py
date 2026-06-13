import os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label='pos'),
        "Recall"   : recall_score(y_test, y_pred, pos_label='pos'),
        "F1 Score" : f1_score(y_test, y_pred, pos_label='pos'),
    }

    print(f"\n--- {name} Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs("../results", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['neg', 'pos'])
    disp.plot(cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"../results/{name.replace(' ', '_')}_confusion.png")
    plt.show()

    return metrics