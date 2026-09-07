# Optimized Spam Detection System

**TechMaster Academy - Phase 03: ML Fundamentals & Optimization**

A Python machine learning project that classifies SMS messages as **spam** or **ham** using three supervised learning algorithms:

* Multinomial Naive Bayes
* Logistic Regression
* Linear Support Vector Machine (SVM)

The project follows a complete machine learning workflow, including data inspection, data cleaning, TF-IDF feature extraction, baseline model training, hyperparameter tuning, model comparison, final evaluation, and deployment through a Streamlit application.

---

## Project Objective

The main objective of this project is to develop an effective SMS spam detection system capable of distinguishing unwanted spam messages from legitimate messages.

The project demonstrates the complete machine learning pipeline:

1. Inspect the raw dataset
2. Clean and prepare the data
3. Split the data into training and testing sets
4. Convert text messages into numerical features using TF-IDF
5. Train three baseline classification models
6. Evaluate and compare the baseline models
7. Optimize the model hyperparameters using GridSearchCV
8. Compare the models before and after tuning
9. Select the best-performing model
10. Evaluate the selected model on the held-out test set
11. Save the trained model and TF-IDF vectorizer
12. Use the final model in a Streamlit application

---

## Setup

Requires Python 3.9 or later.

Run these commands from the project folder:

```bash
python -m venv .venv
```

### Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Running the Project

### Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application provides an interactive interface for classifying messages and working with the trained models.

---

## Using the Application

The Streamlit application provides access to the main stages of the project.

| Tab             | Features                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Classify**    | Select a saved model, enter a message, and view its Spam/Ham prediction.                                                       |
| **Data**        | Inspect the raw and cleaned datasets, check missing values and duplicates, and explore the processed data and TF-IDF features. |
| **Experiments** | Train the three models, perform hyperparameter tuning, and compare their evaluation metrics and confusion matrices.            |
| **Models**      | Manage saved trained models and their corresponding TF-IDF vectorizers.                                                        |

The exact available controls may depend on the current implementation of the Streamlit application.

---

## Dataset

The SMS Spam Collection dataset is stored in:

```text
data/raw.csv
```

The cleaned dataset is stored in:

```text
data/cleaned.csv
```

The dataset contains two classes:

* `ham` — legitimate message
* `spam` — unwanted or spam message

### Data Cleaning

The cleaning process includes:

* Selecting the required label and message columns
* Renaming columns where necessary
* Handling missing values
* Removing empty messages
* Stripping unnecessary whitespace
* Removing duplicate records
* Validating the class labels
* Saving the cleaned dataset

---

## Data Splitting

The dataset is divided into training and testing sets using an **80/20 stratified split**.

The random seed is:

```text
42
```

Stratification is used to preserve the proportion of spam and ham messages in both subsets.

The test set is kept separate from model tuning and is used for final evaluation.

---

## TF-IDF Feature Extraction

Machine learning classifiers cannot directly process raw text messages. Therefore, the messages are converted into numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The vectorizer is fitted using the training data and then used to transform the test data.

The same fitted vectorizer is saved with the trained model so that new messages entered through the Streamlit application are transformed using the same feature representation used during training.

The project can also support alternative text representations depending on the application configuration.

---

## Machine Learning Models

Three classification algorithms are used.

### 1. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classifier commonly used for text classification.

The main hyperparameter optimized in this project is:

```text
alpha
```

The tested values are:

```text
0.1, 0.5, 1.0, 2.0
```

---

### 2. Logistic Regression

Logistic Regression is used as a linear classification model for comparison with Naive Bayes and SVM.

The main hyperparameter optimized is:

```text
C
```

The tested values are:

```text
0.1, 0.5, 1, 2, 5
```

---

### 3. Linear Support Vector Machine

A linear Support Vector Machine is implemented using `LinearSVC`.

The main hyperparameter optimized is:

```text
C
```

The tested values are:

```text
0.1, 0.5, 1, 2, 5
```

---

## Model Evaluation

Each model is evaluated using the same test set and the same evaluation procedure.

The following metrics are used:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**

Because spam detection is the main objective, the F1 score is calculated with **spam as the positive class**.

This provides a balanced measure of the model's ability to identify spam while limiting false positives.

---

## Baseline Evaluation

Before optimization, each classifier is trained using its default configuration.

The baseline models are evaluated on the held-out test set.

This provides a reference point for measuring whether hyperparameter tuning improves model performance.

---

## Hyperparameter Optimization

After the baseline evaluation, the three models are optimized using **GridSearchCV**.

A **5-fold cross-validation** strategy is used.

The hyperparameter search is performed using the training data only.

The optimization criterion is:

```text
Spam F1 Score
```

### SVM

```text
C = [0.1, 0.5, 1, 2, 5]
```

### Logistic Regression

```text
C = [0.1, 0.5, 1, 2, 5]
```

### Multinomial Naive Bayes

```text
alpha = [0.1, 0.5, 1.0, 2.0]
```

For each model, GridSearchCV evaluates the parameter combinations using five-fold cross-validation and selects the configuration with the highest mean spam F1 score.

The selected tuned models are then evaluated on the held-out test set.

---

## Model Selection

The final model is selected after comparing the tuned models.

The selection process is:

```text
Training Data
      ↓
5-Fold Cross-Validation
      ↓
Hyperparameter Search
      ↓
Best Parameters for Each Model
      ↓
Tuned Models
      ↓
Held-Out Test Set
      ↓
Final Performance Comparison
      ↓
Best Final Model
```

The test set is not used to choose hyperparameters. It is reserved for evaluating the final tuned models.

### Final Model

The current experiment selected:

**Linear SVM with C = 5**

It achieved the highest mean cross-validation spam F1 score:

**93.53%**

On the held-out test set, the tuned SVM achieved:

* **Accuracy:** 98.93%
* **Precision:** 99.16%
* **Recall:** 92.19%
* **F1 Score:** 95.55%

The test confusion matrix was:

```text
                 Predicted
                 ham    spam

Actual ham       903      1
Actual spam       10    118
```

This means the model correctly classified 903 ham messages and 118 spam messages, with 1 false positive and 10 false negatives.

Tuning improved the SVM by correctly identifying one additional spam message compared with the baseline configuration.

These results correspond to the current dataset and train/test split and may change if the dataset or experimental configuration is modified.

---

## Results

The current experiment produced the following results:

| Model               | Stage    | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | -------: | --------: | -----: | -------: |
| Naive Bayes         | Baseline |   95.45% |   100.00% | 63.28% |   77.51% |
| Naive Bayes         | Tuned    |   98.55% |    98.29% | 89.84% |   93.88% |
| Logistic Regression | Baseline |   96.41% |    98.92% | 71.88% |   83.26% |
| Logistic Regression | Tuned    |   98.06% |    99.09% | 85.16% |   91.60% |
| SVM                 | Baseline |   98.84% |    99.15% | 91.41% |   95.12% |
| SVM                 | Tuned    |   98.93% |    99.16% | 92.19% |   95.55% |

### Dataset Split

The current experiment used:

* **5,157 prepared messages**
* **4,125 training messages**
* **1,032 test messages**

---

## Streamlit Application

The project includes a Streamlit application that provides an interactive interface for the spam detection system.

The application allows users to enter a message and obtain a prediction from a saved trained model.

The prediction workflow is:

```text
User enters message
        ↓
Saved TF-IDF vectorizer
        ↓
TF-IDF feature vector
        ↓
Saved trained model
        ↓
Prediction
        ↓
SPAM / HAM
```

The application does not need to retrain the model for every prediction. Instead, it loads the previously trained model and vectorizer.

---

## Saving Models

The trained model and its TF-IDF vectorizer are saved using `joblib`.

The saved files are stored in the `models/` directory.

A model and its vectorizer must be kept together because the vectorizer defines how raw messages are converted into the numerical features expected by the trained classifier.

---

## Project Structure

```text
Spam-Detection/
│
├── data/
│   ├── raw.csv
│   └── cleaned.csv
│
├── models/
│   └── Saved trained models and TF-IDF vectorizers
│
├── inspect_data.py
│   └── Dataset inspection
│
├── clean_data.py
│   └── Dataset cleaning and preparation
│
├── vectorize_data.py
│   └── Train/test split and TF-IDF feature extraction
│
├── SVM.py
│   └── Linear SVM implementation
│
├── logistic_regression.py
│   └── Logistic Regression implementation
│
├── naive_bayes.py
│   └── Multinomial Naive Bayes implementation
│
├── tune_models.py
│   └── Hyperparameter tuning using GridSearchCV
│
├── evaluate.py
│   └── Baseline/tuned model evaluation and comparison
│
├── app.py
│   └── Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── .gitignore
    └── Ignored files and directories
```

---

## Main Python Files

| File                     | Purpose                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| `inspect_data.py`        | Inspects the raw dataset and provides information about its structure and contents.              |
| `clean_data.py`          | Cleans and prepares the dataset for machine learning.                                            |
| `vectorize_data.py`      | Performs the train/test split and TF-IDF feature extraction.                                     |
| `SVM.py`                 | Implements the Linear SVM classifier and its evaluation functions.                               |
| `logistic_regression.py` | Implements Logistic Regression and its evaluation functions.                                     |
| `naive_bayes.py`         | Implements Multinomial Naive Bayes and its evaluation functions.                                 |
| `tune_models.py`         | Performs hyperparameter optimization for all three models using GridSearchCV.                    |
| `evaluate.py`            | Trains baseline models, performs tuning, evaluates tuned models, and compares their performance. |
| `app.py`                 | Provides the Streamlit user interface.                                                           |

---

## Terminal Workflow

The main scripts can be executed from the project root.

### 1. Inspect the dataset

```bash
python inspect_data.py
```

### 2. Clean the dataset

```bash
python clean_data.py
```

### 3. Prepare TF-IDF features

```bash
python vectorize_data.py
```

### 4. Run individual baseline models

```bash
python naive_bayes.py
```

```bash
python logistic_regression.py
```

```bash
python SVM.py
```

### 5. Run the complete evaluation and tuning workflow

```bash
python evaluate.py
```

The evaluation script:

* Loads and vectorizes the cleaned dataset
* Trains the three baseline models
* Evaluates their baseline performance
* Tunes all three models using 5-fold cross-validation
* Evaluates the tuned models on the test set
* Compares performance before and after tuning
* Displays confusion matrices
* Allows a tuned model and its vectorizer to be saved

---

## Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Git**
* **GitHub**

---

## Conclusion

This project demonstrates an end-to-end machine learning solution for SMS spam detection.

The system covers the complete workflow from raw data preparation to deployment:

```text
Data
 ↓
Inspection
 ↓
Cleaning
 ↓
Train/Test Split
 ↓
TF-IDF
 ↓
Baseline Models
 ↓
Evaluation
 ↓
Hyperparameter Tuning
 ↓
Final Evaluation
 ↓
Model Selection
 ↓
Model Saving
 ↓
Streamlit Application
```

The final experiment showed that the tuned **Linear SVM** provided the best overall spam detection performance among the three evaluated models, achieving an F1 score of **95.55%** on the held-out test set.
