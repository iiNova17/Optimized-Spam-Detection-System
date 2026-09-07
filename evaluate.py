import pandas as pd

from SVM import train_svm, evaluate_svm
from naive_bayes import train_naive_bayes, evaluate_naive_bayes
from logistic_regression import train_logistic_regression, evaluate_logistic_regression

from vectorize_data import vectorize_data


# Load and vectorize data
X_train, X_test, y_train, y_test, vectorizer = vectorize_data(
        "data/cleaned.csv"
    )

#SVM model
svm_model = train_svm(X_train,y_train)
svm_results, svm_matrix = evaluate_svm(svm_model,X_test,y_test)


#Naive Bayes model
naive_bayes_model = train_naive_bayes(X_train,y_train)
naive_bayes_results, naive_bayes_matrix = evaluate_naive_bayes(naive_bayes_model,X_test,y_test)


#Logistic Regression model
logistic_regression_model = train_logistic_regression(X_train,y_train)
logistic_regression_results, logistic_regression_matrix = evaluate_logistic_regression(logistic_regression_model,X_test,y_test)


results = pd.DataFrame([
    svm_results,
    naive_bayes_results,
    logistic_regression_results
])


print("\nModels Comparison: ")

print(results)


