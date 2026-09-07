import pandas as pd
import joblib

from pathlib import Path

from SVM import train_svm, evaluate_svm

from logistic_regression import (
    train_logistic_regression,
    evaluate_logistic_regression
)

from naive_bayes import (
    train_naive_bayes,
    evaluate_naive_bayes
)

from vectorize_data import vectorize_data
from tune_model import tune_models


def save_model(models, vectorizer):
    """Let the user select and save a tuned model."""

    # Check available models
    if not models:
        print("\nNo models are available to save.")
        return

    print("\n" + "=" * 70)
    print("SAVE MODEL")
    print("=" * 70)

    # Get model names
    model_names = list(models.keys())

    # Print model options
    for number, name in enumerate(model_names, start=1):
        print(f"{number}. {name}")

    print("0. Do not save a model")

    try:
        # Get user choice
        choice = input("\nSelect a model to save: ").strip()

        # Check empty input
        if not choice:
            print("\nNo selection entered.")
            return

        # Convert input to number
        choice = int(choice)

        # Skip saving
        if choice == 0:
            print("\nNo model saved.")
            return

        # Check valid choice
        if choice < 1 or choice > len(model_names):
            print("\nInvalid model selection.")
            return

        # Get selected model
        selected_name = model_names[choice - 1]
        selected_model = models[selected_name]

        # Create safe file name
        file_name = (
            selected_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        # Create models folder
        model_folder = Path(f"models/{file_name}")
        model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Set file paths
        model_path = model_folder / f"{file_name}.joblib"
        vectorizer_path = model_folder / f"{file_name}_vectorizer.joblib"

        # Save model
        joblib.dump(
            selected_model,
            model_path
        )

        # Save vectorizer
        joblib.dump(
            vectorizer,
            vectorizer_path
        )

        print("\nModel saved successfully.")
        print(f"Selected Model: {selected_name}")
        print(f"Model File: {model_path}")
        print(f"Vectorizer File: {vectorizer_path}")

    except ValueError:
        print("\nInvalid input. Please enter a number.")

    except PermissionError:
        print("\nPermission denied while saving files.")

    except OSError as error:
        print(f"\nFile error: {error}")

    except Exception as error:
        print(f"\nUnexpected error while saving model: {error}")


def evaluate_models():
    """Evaluate models before and after tuning."""

    try:
        # Load and vectorize data
        X_train, X_test, y_train, y_test, vectorizer = vectorize_data(
            "data/cleaned.csv"
        )

    except FileNotFoundError:
        print("\nCleaned dataset was not found.")
        return

    except KeyError as error:
        print(f"\nMissing dataset column: {error}")
        return

    except Exception as error:
        print(f"\nError loading dataset: {error}")
        return

    print("\n" + "=" * 70)
    print("BEFORE TUNING")
    print("=" * 70)

    try:
        # Train SVM
        svm_model = train_svm(
            X_train,
            y_train
        )

        # Evaluate SVM
        svm_results, svm_matrix = evaluate_svm(
            svm_model,
            X_test,
            y_test,
            print_results=False
        )

        # Train Logistic Regression
        lr_model = train_logistic_regression(
            X_train,
            y_train
        )

        # Evaluate Logistic Regression
        lr_results, lr_matrix = evaluate_logistic_regression(
            lr_model,
            X_test,
            y_test,
            print_results=False
        )

        # Train Naive Bayes
        nb_model = train_naive_bayes(
            X_train,
            y_train
        )

        # Evaluate Naive Bayes
        nb_results, nb_matrix = evaluate_naive_bayes(
            nb_model,
            X_test,
            y_test,
            print_results=False
        )

    except Exception as error:
        print(f"\nError while training baseline models: {error}")
        return

    # Store baseline results
    before_results = pd.DataFrame([
        svm_results,
        lr_results,
        nb_results
    ])

    # Print baseline results
    print("\nPerformance:")

    print(
        before_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Print baseline confusion matrices
    print("\nConfusion Matrices:")

    print("\nSVM:")
    print(svm_matrix)

    print("\nLogistic Regression:")
    print(lr_matrix)

    print("\nNaive Bayes:")
    print(nb_matrix)

    print("\n" + "=" * 70)
    print("TUNING MODELS")
    print("=" * 70)

    try:
        # Tune all models
        tuned_models, tuning_results = tune_models(
            X_train,
            y_train
        )

    except Exception as error:
        print(f"\nError while tuning models: {error}")
        return

    # Print tuning results
    for model_name, info in tuning_results.items():

        print(f"\n{model_name}")

        print(
            f"Best Parameters: "
            f"{info['Best Parameters']}"
        )

        print(
            f"Cross-Validation F1: "
            f"{info['CV F1 Score']:.4f}"
        )

    print("\n" + "=" * 70)
    print("AFTER TUNING")
    print("=" * 70)

    try:
        # Evaluate tuned SVM
        tuned_svm_results, tuned_svm_matrix = evaluate_svm(
            tuned_models["SVM"],
            X_test,
            y_test,
            print_results=False
        )

        # Evaluate tuned Logistic Regression
        tuned_lr_results, tuned_lr_matrix = evaluate_logistic_regression(
            tuned_models["Logistic Regression"],
            X_test,
            y_test,
            print_results=False
        )

        # Evaluate tuned Naive Bayes
        tuned_nb_results, tuned_nb_matrix = evaluate_naive_bayes(
            tuned_models["Naive Bayes"],
            X_test,
            y_test,
            print_results=False
        )

    except KeyError as error:
        print(f"\nTuned model not found: {error}")
        return

    except Exception as error:
        print(f"\nError while evaluating tuned models: {error}")
        return

    # Store tuned results
    after_results = pd.DataFrame([
        tuned_svm_results,
        tuned_lr_results,
        tuned_nb_results
    ])

    # Print tuned results
    print("\nPerformance:")

    print(
        after_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Print tuned confusion matrices
    print("\nConfusion Matrices:")

    print("\nSVM:")
    print(tuned_svm_matrix)

    print("\nLogistic Regression:")
    print(tuned_lr_matrix)

    print("\nNaive Bayes:")
    print(tuned_nb_matrix)

    # Merge before and after results
    comparison = before_results.merge(
        after_results,
        on="Model",
        suffixes=(" Before", " After")
    )

    # Calculate accuracy change
    comparison["Accuracy Change"] = (
        comparison["Accuracy After"]
        - comparison["Accuracy Before"]
    )

    # Calculate precision change
    comparison["Precision Change"] = (
        comparison["Precision After"]
        - comparison["Precision Before"]
    )

    # Calculate recall change
    comparison["Recall Change"] = (
        comparison["Recall After"]
        - comparison["Recall Before"]
    )

    # Calculate F1 change
    comparison["F1 Change"] = (
        comparison["F1 Score After"]
        - comparison["F1 Score Before"]
    )

    print("\n" + "=" * 70)
    print("BEFORE VS AFTER TUNING")
    print("=" * 70)

    # Create performance table
    performance_table = comparison[
        [
            "Model",
            "Accuracy Before",
            "Accuracy After",
            "Precision Before",
            "Precision After",
            "Recall Before",
            "Recall After",
            "F1 Score Before",
            "F1 Score After"
        ]
    ].rename(
        columns={
            "Accuracy Before": "Acc Before",
            "Accuracy After": "Acc After",
            "Precision Before": "Prec Before",
            "Precision After": "Prec After",
            "Recall Before": "Rec Before",
            "Recall After": "Rec After",
            "F1 Score Before": "F1 Before",
            "F1 Score After": "F1 After"
        }
    )

    # Print performance table
    print("\nPerformance:")

    print(
        performance_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Create changes table
    change_table = comparison[
        [
            "Model",
            "Accuracy Change",
            "Precision Change",
            "Recall Change",
            "F1 Change"
        ]
    ].rename(
        columns={
            "Accuracy Change": "Acc Change",
            "Precision Change": "Prec Change",
            "Recall Change": "Rec Change"
        }
    )

    # Print changes table
    print("\nChanges:")

    print(
        change_table.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}"
        )
    )

    # Ask user to save a model
    save_model(
        tuned_models,
        vectorizer
    )

    return (
        before_results,
        after_results,
        comparison,
        tuned_models
    )


if __name__ == "__main__":
    evaluate_models()