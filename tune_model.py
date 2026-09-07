from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

from SVM import create_svm
from logistic_regression import create_logistic_regression
from naive_bayes import create_naive_bayes


# Create F1 scorer for spam
spam_f1 = make_scorer(
    f1_score,
    pos_label="spam"
)


def tune_svm(X_train, y_train):
    """Tune the SVM model."""

    # Create SVM model
    model = create_svm()

    # Define SVM parameters
    parameters = {
        "C": [0.1, 0.5, 1, 2, 5]
    }

    # Create grid search
    search = GridSearchCV(
        model,
        parameters,
        cv=5,
        scoring=spam_f1
    )

    # Run grid search
    search.fit(X_train, y_train)

    # Return best results
    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_
    )


def tune_logistic_regression(X_train, y_train):
    """Tune the Logistic Regression model."""

    # Create Logistic Regression model
    model = create_logistic_regression()

    # Define Logistic Regression parameters
    parameters = {
        "C": [0.1, 0.5, 1, 2, 5]
    }

    # Create grid search
    search = GridSearchCV(
        model,
        parameters,
        cv=5,
        scoring=spam_f1
    )

    # Run grid search
    search.fit(X_train, y_train)

    # Return best results
    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_
    )


def tune_naive_bayes(X_train, y_train):
    """Tune the Naive Bayes model."""

    # Create Naive Bayes model
    model = create_naive_bayes()

    # Define Naive Bayes parameters
    parameters = {
        "alpha": [0.1, 0.5, 1.0, 2.0]
    }

    # Create grid search
    search = GridSearchCV(
        model,
        parameters,
        cv=5,
        scoring=spam_f1
    )

    # Run grid search
    search.fit(X_train, y_train)

    # Return best results
    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_
    )


def tune_models(X_train, y_train):
    """Tune all models."""

    # Tune SVM
    svm_model, svm_params, svm_score = tune_svm(
        X_train,
        y_train
    )

    # Tune Logistic Regression
    lr_model, lr_params, lr_score = tune_logistic_regression(
        X_train,
        y_train
    )

    # Tune Naive Bayes
    nb_model, nb_params, nb_score = tune_naive_bayes(
        X_train,
        y_train
    )

    # Store tuned models
    models = {
        "SVM": svm_model,
        "Logistic Regression": lr_model,
        "Naive Bayes": nb_model
    }

    # Store tuning results
    tuning_results = {
        "SVM": {
            "Best Parameters": svm_params,
            "CV F1 Score": svm_score
        },
        "Logistic Regression": {
            "Best Parameters": lr_params,
            "CV F1 Score": lr_score
        },
        "Naive Bayes": {
            "Best Parameters": nb_params,
            "CV F1 Score": nb_score
        }
    }

    return models, tuning_results