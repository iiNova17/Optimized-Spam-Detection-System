"""SpamGuard: a compact Streamlit interface for the complete course workflow."""
from contextlib import redirect_stdout
from html import escape
from io import StringIO
import json

import altair as alt
import pandas as pd
import streamlit as st

from clean_data import clean_data
from spam_core import (ROOT, METRICS, fingerprint, prepare_frame, split_frame,
                       make_vectorizer, run_experiment, validate_experiment,
                       predict_messages, report_dict, model_bytes, save_model,
                       load_saved_models)

DATA_PATH = ROOT / 'data' / 'cleaned.csv'
RAW_PATH = ROOT / 'data' / 'raw.csv'
CRIMSON = '#be123c'


def apply_styles():
    st.markdown('''<style>
    .stApp {background:#f8f9fb;color:#24232b}
    .block-container {max-width:1180px;padding:1.8rem 2rem 2rem}
    header[data-testid="stHeader"] {background:transparent;height:1.5rem}
    [data-testid="stToolbar"] {right:1rem}
    [data-testid="stVerticalBlock"] {gap:.7rem}
    h1 {font-size:1.8rem!important;letter-spacing:-.05rem;padding:0!important}
    h2 {font-size:1.35rem!important;padding:.2rem 0!important}
    h3 {font-size:1.05rem!important;padding:.15rem 0!important}
    p, [data-testid="stWidgetLabel"] p {font-size:.9rem}
    [data-testid="stCaptionContainer"] p {color:#656571;font-size:.8rem}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:10px}
    [data-testid="stVerticalBlockBorderWrapper"] {background:white}
    [data-baseweb="tab-list"] {gap:1.7rem;border-bottom:1px solid #dedee5}
    [data-baseweb="tab"] {height:42px;font-size:14px}
    [data-testid="stMetricValue"] {font-size:1.65rem}
    [data-testid="stMetricLabel"] {font-size:.8rem}
    .stButton button,.stDownloadButton button {border-radius:7px;min-height:38px;font-size:14px}
    .stButton button[kind="primary"] {background:#be123c;border-color:#be123c;color:white}
    .stButton button[kind="primary"]:hover {background:#9f1239;border-color:#9f1239}
    button:focus-visible,textarea:focus-visible {outline:2px solid #be123c!important;outline-offset:2px}
    .brand {display:flex;align-items:center;gap:12px;margin-bottom:12px}
    .brand svg {width:34px;height:40px;color:#be123c}
    .brand strong {font-size:27px;letter-spacing:-1px;line-height:1.2}
    .brand small {display:block;color:#656571;font-size:13px;margin-top:3px}
    .prediction {background:#fff6f8;border:1px solid #eadce1;border-radius:10px;padding:22px;min-height:280px}
    .prediction h3 {margin:0 0 28px}
    .prediction .center {text-align:center}
    .prediction .result {font-size:32px;color:#9f1239;font-weight:700;margin:15px 0 5px}
    .prediction .muted {font-size:13px;color:#656571;line-height:1.6}
    .score-bar {height:8px;background:#f3dbe3;border-radius:8px;margin:18px 0 8px;overflow:hidden}
    .score-bar span {display:block;height:100%;background:#be123c}
    .matrix {display:grid;grid-template-columns:100px 1fr 1fr;gap:6px;font-size:12px;align-items:center;margin:12px 0}
    .matrix .cell {padding:12px;background:#fce7ef;border-radius:7px;text-align:center}
    .matrix .error {background:#fff1f2;border:1px solid #fda4af}
    .matrix b {display:block;font-size:24px;color:#9f1239}
    .matrix small {font-size:11px}
    .stExpander {background:white}
    @media(max-width:640px) {
      .block-container {padding:1.6rem 1rem}
      [data-baseweb="tab-list"] {gap:.6rem}
      [data-baseweb="tab"] {font-size:13px}
      .prediction {min-height:240px}
    }
    </style>''', unsafe_allow_html=True)


def heading(title, caption):
    st.subheader(title)
    st.caption(caption)


def csv_download(label, frame, name):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8'), name, 'text/csv')


def table(frame, height=None):
    args = dict(hide_index=True, width='stretch')
    if height:
        args['height'] = height
    st.dataframe(frame, **args)


def sample_message(value):
    st.session_state['message'] = value
    st.session_state.pop('prediction', None)


def prediction_panel(result=None):
    if result is None:
        body = '<p class="muted">Enter a message and select Analyze message.</p>'
    else:
        label = 'Spam' if result['prediction'] == 'spam' else 'Ham · legitimate'
        body = f'<div class="result">{label}</div><p class="muted">{escape(result["model_name"])}</p>'
        if 'spam_probability' in result:
            score = result['spam_probability']
            body += f'<div class="score-bar"><span style="width:{score*100:.2f}%"></span></div><b>{score:.1%} spam probability</b><p class="muted">Ham probability: {1-score:.1%}</p>'
        else:
            body += f'<b>Decision margin: {result["spam_margin"]:+.3f}</b><p class="muted">Positive: spam · Negative: ham</p>'
        if result['recognized_terms'] == 0:
            body += '<p><b>No known terms.</b> Try a longer message.</p>'
    st.markdown(f'<section class="prediction"><h3>Prediction</h3>{body}</section>', unsafe_allow_html=True)


def classify_page(saved_models):
    heading('Classify a message', 'Choose a model and enter an SMS message.')
    if not saved_models:
        st.info('No usable saved models. Train models in Experiments, then save one in Models.')
        return
    left, right = st.columns([1.2, 1], gap='medium')
    with left, st.container(border=True):
        options = list(saved_models)
        selected = st.selectbox('Model', options, index=options.index('naive_bayes') if 'naive_bayes' in options else 0,
                                format_func=lambda key: saved_models[key]['title'], key='classify_model')
        a, b = st.columns(2)
        a.button('Spam example', on_click=sample_message, args=('Congratulations! You won a free prize. Call now to claim your cash reward!',), width='stretch')
        b.button('Ham example', on_click=sample_message, args=('Are we still meeting for lunch tomorrow? Let me know when you are free.',), width='stretch')
        message = st.text_area('Message', placeholder='Paste a message to analyze...', height=115, key='message', max_chars=20000)
        st.caption(f'{len(message):,} characters')
        if st.button('Analyze message', type='primary', width='stretch'):
            if not message.strip():
                st.warning('Enter a message first.')
            else:
                try:
                    item = saved_models[selected]
                    row = predict_messages(item['model'], item['vectorizer'], [message]).iloc[0].to_dict()
                    st.session_state['prediction'] = {'key': (selected, message), 'row': {**row, 'model_name': item['title']}}
                except Exception as exc:
                    st.error(f'Could not classify this message: {exc}')
    with right:
        result = st.session_state.get('prediction', {})
        prediction_panel(result.get('row') if result.get('key') == (selected, message) else None)



def feature_lab(data):
    with st.expander('Split & feature explorer'):
        st.caption('An 80/20 stratified split, seed 42. Vocabulary is learned from training messages only.')
        train_x, test_x, train_y, test_y = split_frame(data)
        vectorizer = make_vectorizer()
        vectors = vectorizer.fit_transform(train_x)
        cols = st.columns(4)
        for col, label, value in zip(cols, ['Training rows', 'Test rows', 'Features', 'Vocabulary'],
                                     [len(train_x), len(test_x), vectors.shape[1], len(vectorizer.vocabulary_)]):
            col.metric(label, f'{value:,}')
        st.caption(f'Sparse matrix density: {vectors.nnz / (vectors.shape[0] * vectors.shape[1]):.3%}')
        distribution = pd.DataFrame({'Train': train_y.value_counts(), 'Test': test_y.value_counts()}).fillna(0).astype(int)
        st.dataframe(distribution, width='stretch')
        query = st.text_input('Find a vocabulary term', placeholder='e.g. prize')
        words = pd.DataFrame({'Term': vectorizer.get_feature_names_out(), 'IDF': vectorizer.idf_})
        if query:
            words = words[words.Term.str.contains(query, case=False, regex=False)]
        table(words.head(100), 180)
        text = st.text_input('Inspect TF-IDF weights', 'Free prize call now')
        row = vectorizer.transform([text]).tocoo()
        if row.nnz:
            weights = pd.DataFrame({'Term': vectorizer.get_feature_names_out()[row.col], 'Weight': row.data})
            st.bar_chart(weights, x='Term', y='Weight', color=CRIMSON, height=180)
        else:
            st.info('No vocabulary terms found in this text.')


def data_page():
    heading('Dataset', 'Inspect and clean the SMS dataset.')
    a, b = st.columns([2, 1])
    source = a.radio('Dataset', ['Cleaned', 'Raw'], horizontal=True)
    if b.button('Clean raw dataset', width='stretch'):
        try:
            log = StringIO()
            with redirect_stdout(log):
                clean_data(str(RAW_PATH), str(DATA_PATH))
            st.session_state['cleaning_log'] = log.getvalue()
            st.session_state.pop('experiment', None)
            st.success('Cleaned dataset saved. Run a new experiment to use it.')
        except Exception as exc:
            st.error(f'Cleaning failed: {exc}')
    path = RAW_PATH if source == 'Raw' else DATA_PATH
    try:
        data = pd.read_csv(path, encoding='cp1252' if source == 'Raw' else 'utf-8')
    except Exception as exc:
        st.error(f'Could not load {path.name}: {exc}')
        return
    cols = st.columns(4)
    for col, label, value in zip(cols, ['Rows', 'Columns', 'Missing cells', 'Duplicate rows'],
                                 [len(data), len(data.columns), int(data.isna().sum().sum()), int(data.duplicated().sum())]):
        col.metric(label, f'{value:,}')
    label_col = 'v1' if source == 'Raw' else 'label'
    if label_col in data:
        counts = data[label_col].value_counts().rename_axis('Label').reset_index(name='Messages')
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('### Class balance')
            st.bar_chart(counts, x='Label', y='Messages', color=CRIMSON, height=185)
        with c2:
            st.markdown('### Records')
            filt = st.selectbox('Filter label', ['All'] + data[label_col].dropna().unique().tolist())
            shown = data if filt == 'All' else data[data[label_col] == filt]
            table(shown, 170)
    with st.expander('Full inspection report'):
        info = StringIO()
        data.info(buf=info)
        st.code(info.getvalue(), language=None)
        table(pd.DataFrame({'Column': data.columns, 'Type': data.dtypes.astype(str).values,
                            'Missing': data.isna().sum().values, 'Unique': data.nunique().values}))
        st.markdown('**First five records**')
        table(data.head())
        st.markdown('**Summary statistics**')
        st.dataframe(data.describe(include='all').astype(str), width='stretch')
    if 'cleaning_log' in st.session_state:
        with st.expander('Last cleaning log'):
            st.code(st.session_state['cleaning_log'], language=None)
    csv_download(f'Download {source.lower()} data', data, f'{source.lower()}.csv')
    if source == 'Cleaned':
        try:
            prepared = prepare_frame(data)
            st.caption(f'{len(data)-len(prepared):,} repeated normalized messages excluded from experiments to prevent overlap.')
            feature_lab(prepared)
        except Exception as exc:
            st.warning(f'Feature explorer unavailable: {exc}')


def matrix_view(matrix):
    tn, fp, fn, tp = [int(x) for x in matrix.ravel()]
    st.markdown(f'''<div class="matrix"><div>Actual ↓<br>Predicted →</div><div>Ham</div><div>Spam</div>
    <div>Ham</div><div class="cell"><b>{tn}</b><small>Correct ham</small></div><div class="cell error"><b>{fp}</b><small>False positives</small></div>
    <div>Spam</div><div class="cell error"><b>{fn}</b><small>Missed spam</small></div><div class="cell"><b>{tp}</b><small>Caught spam</small></div></div>''', unsafe_allow_html=True)


def experiment_results(run):
    cv = run['cv']
    winner = run['winner']
    st.success(f'Best CV F1: {winner[0]} / {winner[1]}')
    st.caption(f'{len(run["train_x"]):,} training rows · {len(run["test_x"]):,} held-out rows · {run["config"]["folds"]} folds · seed {run["config"]["seed"]}')
    chart = alt.Chart(cv).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X('CV F1:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
        y=alt.Y('Model:N', sort=None), yOffset='Stage:N',
        color=alt.Color('Stage:N', scale=alt.Scale(domain=['Baseline', 'Tuned'], range=['#d1a1af', CRIMSON])),
        tooltip=['Model', 'Stage', alt.Tooltip('CV F1:Q', format='.3%')]).properties(height=190)
    st.altair_chart(chart, use_container_width=True)
    with st.expander('Cross-validation scores & tuning search'):
        table(cv.style.format({'CV F1': '{:.2%}', 'CV std': '{:.2%}'}))
        for name, search in run['searches'].items():
            st.markdown(f'**{name}**')
            table(search.assign(params=search.params.astype(str)))
    if run['validation'] is not None:
        scores = run['validation']['scores']
        c1, c2 = st.columns(2)
        model_names = list(dict.fromkeys(scores.Model))
        name = c1.selectbox('Inspect model', model_names, index=model_names.index(winner[0]))
        stages = scores[scores.Model.eq(name)].Stage.tolist()
        stage = c2.radio('Stage', stages, index=len(stages)-1, horizontal=True)
        row = scores[scores.Model.eq(name) & scores.Stage.eq(stage)].iloc[0]
        baseline = scores[scores.Model.eq(name) & scores.Stage.eq('Baseline')].iloc[0]
        for col, metric in zip(st.columns(4), METRICS):
            delta = f'{(row[metric]-baseline[metric])*100:+.2f} pp' if stage == 'Tuned' else None
            col.metric(metric, f'{row[metric]:.2%}', delta)
        left, right = st.columns(2)
        with left:
            st.markdown('### Confusion matrix')
            matrix_view(run['validation']['matrices'][(name, stage)])
        with right:
            st.markdown('### What the errors mean')
            st.write('False positives block legitimate messages. False negatives let spam through.')
            st.caption('Precision, recall, and F1 measure the spam class. Changes are percentage points versus the same model’s baseline.')
        with st.expander('All test metrics & before/after changes'):
            table(scores.style.format({metric: '{:.2%}' for metric in METRICS}))
            if run['config']['tune']:
                before = scores[scores.Stage.eq('Baseline')].set_index('Model')[METRICS]
                after = scores[scores.Stage.eq('Tuned')].set_index('Model')[METRICS]
                st.markdown('**Change in percentage points**')
                st.dataframe(((after-before)*100).style.format('{:+.2f}'), width='stretch')
            csv_download('Download test metrics', scores, 'test_metrics.csv')
        with st.expander('Inspect misclassified messages'):
            errors = run['validation']['mistakes'][(name, stage)]
            error_filter = st.radio('Error type', ['All errors', 'False positives', 'False negatives'], horizontal=True)
            if error_filter != 'All errors':
                errors = errors[errors.Actual.eq('ham' if error_filter == 'False positives' else 'spam')]
            table(errors, 230)
            csv_download('Download errors', errors, 'misclassified_messages.csv')
    st.download_button('Download experiment report', json.dumps(report_dict(run), indent=2), 'experiment.json', 'application/json')


def experiments_page():
    heading('Model evaluation', 'Train and compare Naive Bayes, Logistic Regression, and SVM.')
    with st.form('experiment_config'):
        a, b, c = st.columns(3)
        feature = a.selectbox('Text features', ['TF-IDF', 'Bag of words'])
        ngrams = b.selectbox('Word groups', [1, 2], format_func=lambda x: 'Single words' if x == 1 else 'Single words + pairs')
        tune = c.checkbox('Tune model parameters', value=True, help='Search alpha for Naive Bayes and C for Logistic Regression / SVM.')
        with st.expander('Split & feature settings'):
            x, y, z = st.columns(3)
            test_size = x.slider('Test share (%)', 10, 40, 20, 5)
            seed = y.number_input('Random seed', min_value=0, max_value=100000, value=42)
            folds = z.selectbox('CV folds', [3, 5], index=1)
            x, y = st.columns(2)
            min_df = x.number_input('Minimum document frequency', min_value=1, max_value=20, value=1)
            limit = y.selectbox('Vocabulary limit', ['Unlimited', 3000, 5000, 10000])
        start = st.form_submit_button('Run experiment', type='primary')
    if start:
        config = dict(feature=feature, ngrams=ngrams, tune=tune, test_size=test_size/100,
                      seed=int(seed), folds=folds, min_df=int(min_df), max_features=None if limit == 'Unlimited' else limit)
        bar = st.progress(0, text='Preparing the dataset…')
        try:
            run = run_experiment(DATA_PATH, config, lambda value, text: bar.progress(value, text=text))
            run['validation'] = validate_experiment(run)
            st.session_state['experiment'] = run
            bar.progress(1., text='Experiment complete')
        except Exception as exc:
            st.error(f'Experiment failed: {exc}')
        finally:
            bar.empty()
    run = st.session_state.get('experiment')
    if run:
        if not DATA_PATH.exists() or fingerprint(DATA_PATH) != run['fingerprint']:
            st.warning('The dataset changed since this experiment. Run it again before saving or comparing results.')
        experiment_results(run)
    else:
        st.info('Run an experiment to see CV scores, test metrics, confusion matrices, and tuning changes.')


def models_page(saved_models, errors):
    heading('Saved models', 'Use existing models or save a model from your latest experiment.')
    run = st.session_state.get('experiment')
    if run:
        choices = list(run['models'])
        selected = st.selectbox('Model to save', choices, index=choices.index(run['winner']), format_func=lambda x: ' / '.join(x))
        stale = not DATA_PATH.exists() or fingerprint(DATA_PATH) != run['fingerprint']
        if stale:
            st.warning('The dataset changed. Run a fresh experiment before saving.')
        elif run['validation'] is None:
            st.info('Run model evaluation before saving.')
        a, b = st.columns(2)
        if a.button('Save model locally', type='primary', disabled=stale or run['validation'] is None):
            try:
                target = save_model(run, selected)
                st.session_state['save_notice'] = f'{selected[0]} saved. Available in Classify.'
                st.rerun()
            except Exception as exc:
                st.error(f'Could not save the model: {exc}')
        if not stale and run['validation'] is not None:
            b.download_button('Download model', model_bytes(run, selected), 'spamguard.joblib', 'application/octet-stream')
    else:
        st.caption('Run an experiment to train and save another model.')
    if 'save_notice' in st.session_state:
        st.success(st.session_state['save_notice'])
    for error in errors:
        st.warning(error)
    if saved_models:
        table(pd.DataFrame([{'Model': item['title'], 'Features': type(item['vectorizer']).__name__,
                             'Vocabulary': len(item['vectorizer'].vocabulary_)} for item in saved_models.values()]))
        with st.expander('Model details'):
            selected = st.selectbox('Saved model', list(saved_models), format_func=lambda k: saved_models[k]['title'])
            item = saved_models[selected]
            st.code(str(item['path']), language=None)
            st.json(item['model'].get_params())
    else:
        st.info('No saved model pairs found in the models folder.')


def main():
    st.set_page_config(page_title='SpamGuard', page_icon='✉', layout='wide', initial_sidebar_state='collapsed')
    apply_styles()
    st.markdown('''<div class="brand"><svg viewBox="0 0 32 38" fill="none" aria-hidden="true"><path d="M16 2 29 7v11c0 9-6 14-13 18C9 32 3 27 3 18V7Z" stroke="currentColor" stroke-width="2.5"/><path d="M9 16l5 5 9-10" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg><div><strong>SpamGuard</strong><small>SMS spam detection</small></div></div>''', unsafe_allow_html=True)
    saved_models, errors = load_saved_models()
    tabs = st.tabs(['Classify', 'Data', 'Experiments', 'Models'])
    with tabs[0]:
        classify_page(saved_models)
    with tabs[1]:
        data_page()
    with tabs[2]:
        experiments_page()
    with tabs[3]:
        models_page(saved_models, errors)


if __name__ == '__main__':
    main()
