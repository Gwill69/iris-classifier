import argparse
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris dataset")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split (default=0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility (default=42)")
    return parser.parse_args()

def main():
    args = parse_args()

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state)
    
    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

     os.makedirs("outputs", exist_ok=True)
    
     disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=iris.target_names, cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

     joblib.dump(model, "outputs/model.joblib")
    print("Model saved to outputs/model.joblib")
    print("Confusion matrix saved to outputs/confusion_matrix.png")

    if __name__ == "__main__":
    main()

    



