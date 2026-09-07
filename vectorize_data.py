import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_data(file_path: str):
    """Load, split, and vectorize the dataset."""

    # Load cleaned data
    df = pd.read_csv(file_path)

    # Separate messages and labels
    X = df["message"]
    y = df["label"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create vectorizer
    vectorizer = TfidfVectorizer()

    # Fit on training data
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform test data
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = vectorize_data(
        "data/cleaned.csv"
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")