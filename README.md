# Optimized Spam Detection System

TechMaster Academy — Phase 03: ML Fundamentals & Optimization.

A Python project that classifies SMS messages as **spam** or **ham** using Naive Bayes, Logistic Regression, and a linear Support Vector Machine. A Streamlit interface provides access to data inspection, predictions, training, tuning, and model comparison.

## Setup

Requires Python 3.9 or later. Run these commands from the project folder:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Using the app

| Tab | Features |
|---|---|
| Classify | Select a saved model, enter a message, and view its prediction. |
| Data | Inspect raw or cleaned data, check missing values and duplicates, clean the dataset, and explore TF-IDF features. |
| Experiments | Train all three models, tune parameters, and compare metrics and confusion matrices. |
| Models | Save a trained model with its vectorizer, download it, or inspect saved model parameters. |

To train a model, open **Experiments** and select **Run experiment**. The app trains, selects, and evaluates the models in one run. Open **Models** to save a model, then use it in **Classify**.

## Dataset and method

The SMS Spam Collection dataset is included in `data/raw.csv`. Cleaning keeps the label and message columns, handles missing values, strips whitespace, and removes duplicates. Labels are `spam` and `ham`.

The app uses a stratified 80/20 train/test split with random seed 42. It removes repeated normalized messages before splitting. TF-IDF is the default feature representation; Bag of Words and word pairs are also available.

Each classifier is evaluated using accuracy, spam precision, recall, F1, and a confusion matrix. Five-fold cross-validation tunes `alpha` for Naive Bayes and `C` for Logistic Regression and SVM. Feature extraction runs inside each CV fold. The highest CV spam F1 selects the model before its held-out test evaluation.

## Results

Default app run: 5,157 prepared messages, with 4,125 training and 1,032 test messages.

| Model | Stage | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Naive Bayes | Baseline | 95.45% | 100.00% | 63.28% | 77.51% |
| Naive Bayes | Tuned | 98.55% | 98.29% | 89.84% | 93.88% |
| Logistic Regression | Baseline | 96.41% | 98.92% | 71.88% | 83.26% |
| Logistic Regression | Tuned | 98.06% | 99.09% | 85.16% | 91.60% |
| SVM | Baseline | 98.84% | 99.15% | 91.41% | 95.12% |
| SVM | Tuned | 98.93% | 99.16% | 92.19% | 95.55% |

**Selected model: SVM, C=5.** It achieved the highest mean CV spam F1 (93.53%). On the test set, it correctly classified 903 ham and 118 spam messages, with 1 false positive and 10 false negatives. Tuning caught one additional spam message compared with baseline SVM. These results describe this dataset and split.

Full metrics, configuration, and confusion matrices are saved in [`results/`](results/).

## Project files

```text
app.py                  Streamlit interface
spam_core.py            App training, evaluation, and model helpers
inspect_data.py         Dataset inspection
clean_data.py           Dataset cleaning
vectorize_data.py       Train/test split and TF-IDF extraction
naive_bayes.py          Multinomial Naive Bayes
logistic_regression.py  Logistic Regression
SVM.py                  Linear SVM
tune_model.py           Parameter searches for terminal evaluation
evaluate.py             Terminal model comparison and saving
data/                   Raw and cleaned datasets
models/                 Saved models and vectorizers
results/                Recorded experiment results
tests/                  Data, model, and app checks
.streamlit/config.toml  Light/crimson theme
requirements.txt        Python dependencies
```

## Terminal workflow

```bash
python inspect_data.py
python clean_data.py
python vectorize_data.py
python naive_bayes.py
python logistic_regression.py
python SVM.py
python evaluate.py
```

The terminal scripts use the original TF-IDF workflow. The app uses `spam_core.py` for CV with per-fold feature extraction and normalized deduplication, so its reported scores can differ from terminal results. Saved model/vectorizer pairs remain compatible with the app.

## Tests

```bash
python -m unittest discover -s tests -v
```
