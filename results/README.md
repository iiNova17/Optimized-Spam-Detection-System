# Default experiment

Default app experiment, 2026-09-08. Raw cleaned CSV: 5,169 rows. After excluding repeated normalized messages: 5,157 rows, split into 4,125 training and 1,032 test rows. Seed 42, stratified 80/20 split, TF-IDF single words, five-fold CV, alpha/C tuning.

Cross-validation selected **SVM with C=5** before the test results were revealed. Its mean training CV spam F1 was **93.53%**, versus **93.17%** for baseline SVM. Selection used spam F1 to balance precision and recall; the test set did not determine the winner.

| Model | Stage | Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---|---:|---:|---:|---:|
| Naive Bayes | Baseline | 95.45% | 100.00% | 63.28% | 77.51% |
| Naive Bayes | Tuned | 98.55% | 98.29% | 89.84% | 93.88% |
| Logistic Regression | Baseline | 96.41% | 98.92% | 71.88% | 83.26% |
| Logistic Regression | Tuned | 98.06% | 99.09% | 85.16% | 91.60% |
| SVM | Baseline | 98.84% | 99.15% | 91.41% | 95.12% |
| SVM | Tuned | 98.93% | 99.16% | 92.19% | 95.55% |

The selected model correctly kept 903 ham messages, incorrectly flagged 1 ham message, missed 10 spam messages, and caught 118 spam messages. Compared with baseline SVM, tuning caught one additional spam message, increasing test F1 by about 0.42 percentage points. Results apply to this SMS dataset and split.

`default_experiment.json` records the full scores, matrices, dataset SHA-256, configuration, selection rule, and actual package versions. `cross_validation.csv` and `test_metrics.csv` provide machine-readable comparison tables.
