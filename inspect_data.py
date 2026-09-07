import pandas as pd


def inspect_data(file_path: str) -> None:
    """
    Inspect and print general information about a CSV dataset.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    """

    try:
        df = pd.read_csv(file_path, encoding="cp1252")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_path}")
        return

    except pd.errors.ParserError:
        print(f"Error parsing file: {file_path}")
        return

    # General dataset information

    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)

    print(f"Rows:    {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Column names
    print("\n" + "=" * 60)
    print("COLUMNS")
    print("=" * 60)

    print(df.columns.tolist())

    # DataFrame information

    print("\n" + "=" * 60)
    print("DATAFRAME INFO")
    print("=" * 60)

    df.info()

    # First rows


    print("\n" + "=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)

    print(df.head())

    # Missing values

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(df.isnull().sum())


    # Duplicate rows

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)

    print(f"Duplicates: {df.duplicated().sum()}")


    # Unique values per column
    print("\n" + "=" * 60)
    print("UNIQUE VALUES PER COLUMN")
    print("=" * 60)

    print(df.nunique())

    # Distribution of first column
    # Useful because v1 contains spam / ham in this dataset

    first_column = df.columns[0]

    print("\n" + "=" * 60)
    print(f"DISTRIBUTION OF '{first_column}'")
    print("=" * 60)

    print(df[first_column].value_counts(dropna=False))

    # Summary statistics

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(df.describe(include="all"))

    print("\nInspection complete.")


if __name__ == "__main__":
    inspect_data("data/raw.csv")