import pandas as pd
from pathlib import Path


def clean_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Clean the raw spam dataset and save a cleaned version.
    """

    # Load the raw dataset
    try:
        df = pd.read_csv(input_path, encoding="cp1252")

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Input file is empty: {input_path}"
        )

    except pd.errors.ParserError as error:
        raise ValueError(
            f"Could not parse file: {input_path}"
        ) from error

    print(f"Original dataset shape: {df.shape}")

    # Check required columns
    required_columns = {"v1", "v2"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Expected columns v1 and v2. Found: {df.columns.tolist()}"
        )

    # Keep useful columns
    df = df[["v1", "v2"]].copy()

    # Rename columns
    df = df.rename(
        columns={
            "v1": "label",
            "v2": "message"
        }
    )

    # Remove missing values
    missing_rows = df.isnull().any(axis=1).sum()
    df = df.dropna(subset=["label", "message"])

    # Remove duplicate rows
    duplicate_rows = df.duplicated().sum()
    df = df.drop_duplicates()

    # Clean label values
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove message whitespace
    df["message"] = (
        df["message"]
        .astype(str)
        .str.strip()
    )

    # Remove empty messages
    empty_messages = (df["message"] == "").sum()
    df = df[df["message"] != ""]

    # Check label values
    valid_labels = {"ham", "spam"}
    invalid_labels = set(df["label"].unique()) - valid_labels

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels found: {invalid_labels}"
        )

    # Reset row index
    df = df.reset_index(drop=True)

    # Create output folder
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the cleaned dataset
    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    # Print cleaning results
    print(f"Missing rows removed: {missing_rows}")
    print(f"Duplicate rows removed: {duplicate_rows}")
    print(f"Empty messages removed: {empty_messages}")
    print(f"Final dataset shape: {df.shape}")

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print(f"\nSaved cleaned dataset to: {output_file}")

    return df


if __name__ == "__main__":
    clean_data(
        input_path="data/raw.csv",
        output_path="data/cleaned.csv"
    )