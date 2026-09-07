from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from vectorize_data import vectorize_data


def create_naive_bayes():
    """Create the Naive Bayes model."""

    return MultinomialNB()


def train_naive_bayes(X_train, y_train):
    """Create and train the Naive Bayes model."""

    # Create model
    model = create_naive_bayes()

    # Train model
    model.fit(X_train, y_train)

    return model


def evaluate_naive_bayes(model, X_test, y_test, print_results=True):
    """Evaluate the trained Naive Bayes model."""

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
        "Model": "Naive Bayes",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    # Print results
    if print_results:
        print("\nNaive Bayes Results")
        print("-" * 30)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

    return results, matrix


def run_naive_bayes():
    """Run the complete Naive Bayes pipeline."""

    # Load and vectorize data
    X_train, X_test, y_train, y_test, _ = vectorize_data(
        "data/cleaned.csv"
    )

    # Train model
    model = train_naive_bayes(
        X_train,
        y_train
    )

    # Evaluate model
    evaluate_naive_bayes(
        model,
        X_test,
        y_test
    )



if __name__ == "__main__":
    run_naive_bayes()