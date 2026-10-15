"""Small offline regression suite; no Streamlit, LM Studio or project data needed."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from risk_service import RiskService, assess_risk, extract_applicant_id
from explanations import build_expected, generate_explanation, parse_explanation, validate_explanation


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            'predicted_risk': 'High', 'missing_features': ['MonthlyIncome'],
            'shap_factors': [
                {'feature': 'DebtRatio', 'contribution': 0.1},
                {'feature': 'NumberOfTimes90DaysLate', 'contribution': -0.02}
            ]
        }
        self.expected = build_expected(self.data)

    def test_valid_json_variants(self):
        raw = json.dumps(self.expected)
        variants = [raw, f'```json\n{raw}\n```',
                    '```json\n' + json.dumps({'evidence': self.expected}) + '\n```']
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(validate_explanation(parse_explanation(variant), self.expected), self.expected)
        with self.assertRaises(ValueError):
            parse_explanation('{"risk_label":"High","risk_label":"Low"}')

    def test_rejects_six_incorrect_payloads(self):
        cases = []
        bad = copy.deepcopy(self.expected); bad['risk_label'] = 'Low'; cases.append(bad)
        bad = copy.deepcopy(self.expected); bad['factors'][0]['direction'] = 'lowers'; cases.append(bad)
        bad = copy.deepcopy(self.expected); bad['factors'].append(bad['factors'][0]); cases.append(bad)
        bad = copy.deepcopy(self.expected); bad['factors'].pop(); cases.append(bad)
        bad = copy.deepcopy(self.expected); bad['missing_features'] = []; cases.append(bad)
        bad = copy.deepcopy(self.expected); bad['explanation'] = 'Guaranteed approval'; cases.append(bad)
        for index, payload in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_explanation(payload, self.expected)

    def test_valid_client_and_failure_fallback(self):
        expected = self.expected
        class GoodClient:
            def invoke(self, prompt):
                return SimpleNamespace(content=json.dumps(expected))
        class UnavailableClient:
            def invoke(self, prompt):
                raise ConnectionError('Offline test')
        good = generate_explanation(self.data, client=GoodClient())
        fallback = generate_explanation(self.data, client=UnavailableClient())
        self.assertEqual(good.source, 'validated_llm')
        self.assertEqual(fallback.source, 'template')
        self.assertEqual(good.text, fallback.text)
        self.assertIn('Monthly income', good.text)
        self.assertIn('90+ day delinquency count lowers', good.text)


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        folder = Path(cls.temp.name)
        rng = np.random.default_rng(42)
        count = 80
        df = pd.DataFrame({
            'applicant_id': np.arange(1001, 1001 + count),
            'RevolvingUtilizationOfUnsecuredLines': rng.random(count),
            'DebtRatio': rng.random(count),
            'MonthlyIncome': rng.uniform(1000, 8000, count),
            'NumberOfDependents': rng.integers(0, 4, count),
            'NumberOfTime30-59DaysPastDueNotWorse': rng.integers(0, 3, count),
            'NumberOfTime60-89DaysPastDueNotWorse': rng.integers(0, 2, count),
            'NumberOfTimes90DaysLate': rng.integers(0, 2, count)
        })
        df.loc[[0, 5], 'MonthlyIncome'] = np.nan
        df.loc[[2, 7], 'NumberOfDependents'] = np.nan
        columns = df.columns.drop('applicant_id').tolist()
        y = (df['DebtRatio'] + df['RevolvingUtilizationOfUnsecuredLines'] > 1).astype(int)
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('classifier', RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42))
        ]).fit(df[columns], y)
        # Deliberately shuffle rows to test stable ID lookup.
        df = df.sample(frac=1, random_state=7)
        df.to_csv(folder / 'data.csv', index=False)
        joblib.dump({'pipeline': pipeline, 'threshold': 0.25, 'feature_columns': columns}, folder / 'model.joblib')
        cls.service = RiskService(folder / 'data.csv', folder / 'model.joblib')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_search_lookup_and_shap_agree(self):
        for label in ['High', 'Low']:
            rows = self.service.search_applicants(f'Show {label.lower()} risk applicants')
            self.assertGreater(len(rows), 0)
            for _, row in rows.iterrows():
                result = self.service.lookup_applicant(int(row['applicant_id']))
                self.assertEqual(result['predicted_risk'], label)
                self.assertAlmostEqual(result['predicted_probability'], row['predicted_probability'])
                self.assertAlmostEqual(result['shap_baseline'] + sum(f['contribution'] for f in result['shap_factors']), result['predicted_probability'], places=6)

    def test_missing_income_and_invalid_id(self):
        result = self.service.lookup_applicant(1001)
        self.assertIsNone(result['features']['MonthlyIncome'])
        self.assertIn('MonthlyIncome', result['missing_features'])
        self.assertAlmostEqual(result['shap_baseline'] + sum(f['contribution'] for f in result['shap_factors']), result['predicted_probability'], places=6)
        self.assertIn('error', self.service.lookup_applicant(999999))

    def test_routing_and_missing_income_rule(self):
        self.assertEqual(extract_applicant_id(' 7 '), 7)
        self.assertEqual(extract_applicant_id('Why applicant ID: 7?'), 7)
        self.assertIsNone(extract_applicant_id('Show applicants with 90 days late payments'))
        factors, recommendations = assess_risk({'DebtRatio': 5710})
        self.assertEqual(factors, [])
        self.assertEqual(recommendations, [])


if __name__ == '__main__':
    unittest.main()
