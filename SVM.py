from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from vectorize_data import vectorize_data


def create_svm():
    """Create the SVM model."""

    return LinearSVC()


def train_svm(X_train, y_train):
    """Create and train the SVM model."""

    # Create model
    model = create_svm()

    # Train model
    model.fit(X_train, y_train)

    return model


def evaluate_svm(model, X_test, y_test, print_results=True):
    """Evaluate the trained SVM model."""

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="spam"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="spam"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="spam"
    )

    # Create confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=["ham", "spam"]
    )

    # Store results
    results = {
        "Model": "SVM",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    # Print results
    if print_results:
        print("\nSVM Results")
        print("-" * 30)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

    return results, matrix


def run_svm():
    """Run the complete SVM pipeline."""

    # Load and vectorize data
    X_train, X_test, y_train, y_test, vectorizer = vectorize_data(
        "data/cleaned.csv"
    )

    # Train model
    model = train_svm(
        X_train,
        y_train
    )

    # Evaluate model
    evaluate_svm(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    run_svm()