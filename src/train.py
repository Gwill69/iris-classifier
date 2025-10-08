from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
#!/usr/bin/env python3
"""
train.py — Train a Decision Tree on the Iris dataset
and save outputs (confusion matrix + model).
"""

import os
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def main():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # Save confusion matrix as PNG
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=iris.target_names, cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to outputs/confusion_matrix.png")

    # Save trained model
    joblib.dump(model, "outputs/model.joblib")
    print("Model saved to outputs/model.joblib")


if __name__ == "__main__":
    main()

import os
os.makedirs("outputs", exist_ok=True)

y_pred = model.predict(X_test)

import matplotlib.pyplot as plt

plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()


