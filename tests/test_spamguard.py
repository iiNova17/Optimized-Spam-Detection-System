import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from streamlit.testing.v1 import AppTest

from spam_core import (ROOT, prepare_frame, split_frame, make_vectorizer, run_experiment,
                       validate_experiment, predict_messages, save_model, load_saved_models, report_dict)


class DataTests(unittest.TestCase):
    def test_normalized_duplicates_and_missing(self):
        frame = pd.DataFrame({'label': [' HAM ', 'ham', 'spam', 'spam', None],
                              'message': ['Hello there', ' hello   THERE ', 'free prize', '  ', 'lost']})
        out = prepare_frame(frame)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out.label), {'ham', 'spam'})

    def test_conflicting_labels_rejected(self):
        frame = pd.DataFrame({'label': ['ham', 'spam'], 'message': ['Hello', ' hello ']})
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            prepare_frame(frame)

    def test_invalid_labels_and_schema_rejected(self):
        for frame in [pd.DataFrame({'x': [1]}), pd.DataFrame({'label': ['bad'], 'message': ['hello']})]:
            with self.assertRaises(ValueError):
                prepare_frame(frame)

    def test_split_reproducible_and_disjoint(self):
        data = prepare_frame(pd.read_csv(ROOT / 'data/cleaned.csv'))
        a = split_frame(data)
        b = split_frame(data)
        self.assertEqual(a[0].tolist(), b[0].tolist())
        normalize = lambda s: set(s.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip())
        self.assertFalse(normalize(a[0]) & normalize(a[1]))
        self.assertEqual(len(a[0]) + len(a[1]), len(data))
        self.assertEqual(set(a[2]), {'ham', 'spam'})

    def test_vectorizer_does_not_learn_test_terms(self):
        vec = make_vectorizer()
        vec.fit_transform(['meeting lunch', 'free prize'])
        out = vec.transform(['unseenuniquetoken'])
        self.assertNotIn('unseenuniquetoken', vec.vocabulary_)
        self.assertEqual(out.nnz, 0)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        data = prepare_frame(pd.read_csv(ROOT / 'data/cleaned.csv'))
        sample = data.groupby('label', group_keys=False).apply(lambda group: group.sample(100, random_state=42))
        cls.path = Path(cls.temp.name) / 'sample.csv'
        sample.to_csv(cls.path, index=False)
        cls.config = dict(feature='TF-IDF', ngrams=1, min_df=1, max_features=None,
                          test_size=.2, seed=42, folds=3, tune=True)
        cls.experiment = run_experiment(cls.path, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_cv_selection_precedes_test(self):
        self.assertIsNone(self.experiment['validation'])
        self.assertEqual(len(self.experiment['models']), 6)
        expected = self.experiment['cv'].sort_values('CV F1', ascending=False, kind='stable').iloc[0]
        self.assertEqual(self.experiment['winner'], (expected.Model, expected.Stage))
        for pipe in self.experiment['models'].values():
            self.assertEqual(list(pipe.named_steps), ['features', 'model'])

    def test_test_metrics_and_prediction_shape(self):
        validation = validate_experiment(self.experiment)
        for choice, pipe in self.experiment['models'].items():
            pred = pipe.predict(self.experiment['test_x'])
            self.assertEqual(pred.shape, (len(self.experiment['test_y']),))
            matrix = validation['matrices'][choice]
            self.assertEqual(int(matrix.sum()), len(pred))
            self.assertTrue((matrix == confusion_matrix(self.experiment['test_y'], pred, labels=['ham', 'spam'])).all())
            row = validation['scores']
            row = row[row.Model.eq(choice[0]) & row.Stage.eq(choice[1])].iloc[0]
            self.assertAlmostEqual(row['F1 Score'], f1_score(self.experiment['test_y'], pred, pos_label='spam'))

    def test_save_load_preserves_predictions_and_no_overwrite(self):
        run = dict(self.experiment, validation=validate_experiment(self.experiment))
        choice = run['winner']
        folder = Path(self.temp.name) / 'models'
        a = save_model(run, choice, folder)
        b = save_model(run, choice, folder)
        self.assertNotEqual(a, b)
        saved_models, errors = load_saved_models(folder)
        self.assertFalse(errors)
        item = saved_models[a.name]
        messages = ['free prize claim now', 'see you tomorrow', 'zzzxxyy']
        out = predict_messages(item['model'], item['vectorizer'], messages)
        self.assertEqual(out.prediction.tolist(), run['models'][choice].predict(messages).tolist())
        self.assertEqual(int(out.iloc[-1].recognized_terms), 0)
        json.dumps(report_dict(run))

    def test_svm_margin_is_not_probability(self):
        pipe = self.experiment['models'][('SVM', 'Baseline')]
        out = predict_messages(pipe.named_steps['model'], pipe.named_steps['features'], ['hello tomorrow'])
        self.assertIn('spam_margin', out)
        self.assertNotIn('spam_probability', out)

    def test_bag_of_words_baselines(self):
        run = run_experiment(self.path, dict(self.config, feature='Bag of words', tune=False))
        self.assertEqual(len(run['models']), 3)


class AppTests(unittest.TestCase):
    def test_experiment_and_validation_controls(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
        next(c for c in app.checkbox if c.label == 'Tune model parameters').uncheck()
        next(b for b in app.button if b.label == 'Run experiment').click().run()
        self.assertFalse(app.exception)
        self.assertIsNotNone(app.session_state['experiment']['validation'])
        self.assertTrue(any(b.label == 'Save model locally' and not b.disabled for b in app.button))

    def test_empty_sample_prediction_and_navigation(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
        self.assertFalse(app.exception)
        self.assertEqual([t.label for t in app.tabs], ['Classify', 'Data', 'Experiments', 'Models'])
        find_button = lambda label: next(b for b in app.button if b.label == label)
        find_button('Analyze message').click().run()
        self.assertTrue(any('Enter a message' in w.value for w in app.warning))
        find_button('Spam example').click().run()
        find_button('Analyze message').click().run()
        self.assertFalse(app.exception)
        self.assertIn('prediction', app.session_state)
        next(x for x in app.text_area if x.label == 'Message').set_value('new text').run()
        panels = [x.value for x in app.markdown if 'class="prediction"' in x.value]
        self.assertTrue(any('Enter a message and select Analyze message.' in p for p in panels))


if __name__ == '__main__':
    unittest.main()

