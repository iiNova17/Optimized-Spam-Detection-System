"""Training, evaluation, and saved-model helpers for the Streamlit app."""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
import uuid

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from SVM import create_svm
from logistic_regression import create_logistic_regression
from naive_bayes import create_naive_bayes

ROOT = Path(__file__).resolve().parent
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
FACTORIES = {'Naive Bayes': create_naive_bayes, 'Logistic Regression': create_logistic_regression, 'SVM': create_svm}
GRIDS = {'Naive Bayes': {'model__alpha': [0.1, 0.5, 1., 2.]},
         'Logistic Regression': {'model__C': [0.1, 0.5, 1., 2., 5.]},
         'SVM': {'model__C': [0.1, 0.5, 1., 2., 5.]}}


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def prepare_frame(frame):
    """Normalize before deduplication; fail on conflicting labels for identical text."""
    frame = frame.rename(columns={'v1': 'label', 'v2': 'message'})
    if not {'label', 'message'}.issubset(frame.columns):
        raise ValueError('The dataset needs label and message columns (or v1 and v2).')
    data = frame[['label', 'message']].dropna().copy()
    data['label'] = data.label.astype(str).str.strip().str.lower()
    data['message'] = data.message.astype(str).str.strip()
    data = data[data.message.ne('')]
    if not set(data.label).issubset({'ham', 'spam'}):
        raise ValueError('Labels must be ham or spam. Correct invalid labels before training.')
    key = data.message.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    if data.groupby(key).label.nunique().gt(1).any():
        raise ValueError('Identical normalized messages have conflicting labels. Resolve them before training.')
    data = data.loc[~key.duplicated()].reset_index(drop=True)
    if set(data.label) != {'ham', 'spam'}:
        raise ValueError('Both ham and spam examples are required.')
    return data


def split_frame(data, test_size=0.2, seed=42):
    return train_test_split(data.message, data.label, test_size=test_size, random_state=seed, stratify=data.label)


def make_vectorizer(feature='TF-IDF', ngrams=1, min_df=1, max_features=None):
    factory = TfidfVectorizer if feature == 'TF-IDF' else CountVectorizer
    return factory(ngram_range=(1, ngrams), min_df=min_df, max_features=max_features)


def run_experiment(path, config, progress=None):
    """Select using training CV only. Test labels are not scored here."""
    data = prepare_frame(pd.read_csv(path))
    train_x, test_x, train_y, test_y = split_frame(data, config['test_size'], config['seed'])
    folds = config['folds']
    if train_y.value_counts().min() < folds:
        raise ValueError(f'Each training class needs at least {folds} messages for this cross-validation.')
    cv = StratifiedKFold(folds, shuffle=True, random_state=config['seed'])
    scorer = make_scorer(f1_score, pos_label='spam', zero_division=0)
    vectorizer = make_vectorizer(config['feature'], config['ngrams'], config['min_df'], config['max_features'])
    models, rows, searches = {}, [], {}
    for i, (name, factory) in enumerate(FACTORIES.items()):
        if progress:
            progress(i / 3, f'Training {name} with {folds}-fold cross-validation…')
        estimator = factory()
        if 'random_state' in estimator.get_params():
            estimator.set_params(random_state=config['seed'])
        pipeline = Pipeline([('features', clone(vectorizer)), ('model', estimator)])
        scores = cross_val_score(pipeline, train_x, train_y, cv=cv, scoring=scorer)
        pipeline.fit(train_x, train_y)
        models[(name, 'Baseline')] = pipeline
        rows.append({'Model': name, 'Stage': 'Baseline', 'CV F1': scores.mean(), 'CV std': scores.std(), 'Parameters': 'Default'})
        if config['tune']:
            search = GridSearchCV(clone(pipeline), GRIDS[name], cv=cv, scoring=scorer, error_score='raise')
            search.fit(train_x, train_y)
            models[(name, 'Tuned')] = search.best_estimator_
            rows.append({'Model': name, 'Stage': 'Tuned', 'CV F1': search.best_score_,
                         'CV std': search.cv_results_['std_test_score'][search.best_index_],
                         'Parameters': str({k.replace('model__', ''): v for k, v in search.best_params_.items()})})
            searches[name] = pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    table = pd.DataFrame(rows)
    # Stable ties favor the first simpler baseline, then Naive Bayes, LR, SVM.
    best = table.sort_values('CV F1', ascending=False, kind='stable').iloc[0]
    return {'models': models, 'cv': table, 'searches': searches,
            'winner': (best['Model'], best['Stage']), 'train_x': train_x, 'train_y': train_y,
            'test_x': test_x, 'test_y': test_y, 'config': config,
            'fingerprint': fingerprint(path), 'rows': len(data),
            'created': datetime.now(timezone.utc).isoformat(), 'validation': None}


def validate_experiment(run):
    """Reveal holdout results after the CV winner is fixed, without reselection."""
    rows, matrices, mistakes = [], {}, {}
    for (name, stage), model in run['models'].items():
        pred = model.predict(run['test_x'])
        y = run['test_y']
        rows.append({'Model': name, 'Stage': stage, 'Accuracy': accuracy_score(y, pred),
                     'Precision': precision_score(y, pred, pos_label='spam', zero_division=0),
                     'Recall': recall_score(y, pred, pos_label='spam', zero_division=0),
                     'F1 Score': f1_score(y, pred, pos_label='spam', zero_division=0)})
        matrices[(name, stage)] = confusion_matrix(y, pred, labels=['ham', 'spam'])
        examples = pd.DataFrame({'Message': run['test_x'], 'Actual': y, 'Predicted': pred})
        mistakes[(name, stage)] = examples[examples.Actual.ne(examples.Predicted)].reset_index(drop=True)
    return {'scores': pd.DataFrame(rows), 'matrices': matrices, 'mistakes': mistakes}


def predict_messages(model, vectorizer, messages):
    vectors = vectorizer.transform(messages)
    predictions = model.predict(vectors)
    result = pd.DataFrame({'message': list(messages), 'prediction': predictions,
                           'recognized_terms': vectors.getnnz(axis=1)})
    spam_index = list(model.classes_).index('spam')
    if hasattr(model, 'predict_proba'):
        result['spam_probability'] = model.predict_proba(vectors)[:, spam_index]
    elif hasattr(model, 'decision_function'):
        direction = 1 if spam_index == 1 else -1
        result['spam_margin'] = model.decision_function(vectors) * direction
    return result


def report_dict(run):
    report = {'created': run['created'], 'dataset_sha256': run['fingerprint'],
              'config': run['config'], 'training_rows': len(run['train_x']), 'test_rows': len(run['test_x']),
              'selected_model': list(run['winner']), 'selection_rule': 'Highest training CV spam F1; stable ties prefer simpler baselines.',
              'cross_validation': run['cv'].to_dict('records'),
              'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__}}
    if run['validation'] is not None:
        report['test_metrics'] = run['validation']['scores'].to_dict('records')
        report['confusion_matrices'] = {f'{k[0]} / {k[1]}': v.tolist() for k, v in run['validation']['matrices'].items()}
    return report


def model_bytes(run, choice):
    pipeline = run['models'][choice]
    output = BytesIO()
    joblib.dump({'model': pipeline.named_steps['model'], 'vectorizer': pipeline.named_steps['features'],
                 'metadata': {**report_dict(run), 'saved_model': list(choice)}}, output)
    return output.getvalue()


def save_model(run, choice, folder=None):
    """Save a model and vectorizer in a new folder."""
    folder = Path(folder) if folder is not None else ROOT / 'models'
    slug = choice[0].lower().replace(' ', '_')
    name = f'{slug}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    target = folder / name
    target.mkdir(parents=True, exist_ok=False)
    (target / 'bundle.joblib').write_bytes(model_bytes(run, choice))
    (target / 'metadata.json').write_text(json.dumps({**report_dict(run), 'saved_model': list(choice)}, indent=2), encoding='utf-8')
    return target


def load_saved_models(folder=None):
    folder = Path(folder) if folder is not None else ROOT / 'models'
    saved_models, errors = {}, []
    if not folder.exists():
        return saved_models, errors
    for directory in sorted(folder.iterdir()):
        if not directory.is_dir():
            continue
        try:
            if (directory / 'bundle.joblib').exists():
                item = joblib.load(directory / 'bundle.joblib')
                choice = item.get('metadata', {}).get('saved_model', [directory.name, 'Saved'])
                item['title'] = ' / '.join(choice) + ' · ' + directory.name[-6:]
            else:
                model_path = directory / f'{directory.name}.joblib'
                vec_path = directory / f'{directory.name}_vectorizer.joblib'
                if not model_path.exists() or not vec_path.exists():
                    errors.append(f'{directory.name}: missing model or vectorizer.')
                    continue
                item = {'model': joblib.load(model_path), 'vectorizer': joblib.load(vec_path), 'metadata': {},
                        'title': 'SVM' if directory.name == 'svm' else directory.name.replace('_', ' ').title()}
            if set(item['model'].classes_) != {'ham', 'spam'}:
                raise ValueError('Expected ham and spam classes.')
            item['model'].predict(item['vectorizer'].transform(['compatibility check']))
            item['path'] = directory
            saved_models[directory.name] = item
        except Exception as exc:
            errors.append(f'{directory.name}: {exc}')
    return saved_models, errors
