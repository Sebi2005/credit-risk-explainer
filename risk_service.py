"""CSV lookup, predictions, manual indicators and SHAP; no Streamlit dependency."""
from functools import cached_property
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

def assess_risk(features):
    """Supplementary manual indicators and suggestions"""
    factors = []
    recommendations = []
    if features.get('RevolvingUtilizationOfUnsecuredLines', 0) > 0.35:
        factors.append(f"High Credit Utilization ({features['RevolvingUtilizationOfUnsecuredLines']:.2%})")
        recommendations.append('Reduce credit card balances to below 30% of your total limit.')
    income = features.get('MonthlyIncome')
    if income is not None and income > 0:
        if features.get('DebtRatio', 0) > 0.4:
            factors.append(f"Elevated recorded debt ratio ({features['DebtRatio']:.2f})")
            recommendations.append('Review existing debt payments and verify the debt-ratio inputs.')
    if features.get('NumberOfTime30-59DaysPastDueNotWorse', 0) > 0:
        factors.append('History of 30-59 day delinquencies')
        recommendations.append('Set up automatic payments.')
    if features.get('NumberOfTimes90DaysLate', 0) > 0:
        factors.append('History of severe (90+ days) delinquency')
        recommendations.append('Maintain on-time payments and review overdue accounts.')
    return (factors, recommendations)


def extract_applicant_id(query):
    match = re.fullmatch(r"\s*(\d+)\s*", query)
    if match is None:
        match = re.search(r"\bapplicant\s+(?:id\s*[:#]?\s*)?(\d+)\b", query, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


class RiskService:
    def __init__(self, csv_path=None, model_path=None):
        self.csv_path = Path(csv_path) if csv_path is not None else PROJECT_ROOT / "dataset/cs-training.csv"
        self.model_path = Path(model_path) if model_path is not None else PROJECT_ROOT / "models/credit_risk_model.joblib"

    @cached_property
    def bundle(self):
        return joblib.load(self.model_path)

    @cached_property
    def positive_class_index(self):
        classes = self.bundle["pipeline"].named_steps["classifier"].classes_
        matches = np.flatnonzero(classes == 1)
        if len(matches) != 1:
            raise ValueError("Model must contain positive class 1.")
        return int(matches[0])

    @cached_property
    def explainer(self):
        import shap
        forest = self.bundle["pipeline"].named_steps["classifier"]
        return shap.TreeExplainer(forest, feature_perturbation="tree_path_dependent", model_output="raw")

    @cached_property
    def _applicants(self):
        df = pd.read_csv(self.csv_path)
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.rename(columns={df.columns[0]: "applicant_id"})
        if "applicant_id" not in df.columns:
            raise ValueError("Dataset must contain applicant_id.")
        if df["applicant_id"].isna().any() or df["applicant_id"].duplicated().any():
            raise ValueError("Applicant IDs must be present and unique.")
        missing = set(self.bundle["feature_columns"]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        return df

    def load_applicants(self):
        return self._applicants.copy()

    @cached_property
    def _scored_applicants(self):
        df = self.load_applicants()
        probabilities = self.bundle["pipeline"].predict_proba(
            df[self.bundle["feature_columns"]]
        )[:, self.positive_class_index]
        df["predicted_probability"] = probabilities
        df["predicted_risk"] = np.where(probabilities >= self.bundle["threshold"], "High", "Low")
        return df

    def load_scored_applicants(self):
        return self._scored_applicants.copy()

    def lookup_applicant(self, applicant_id):
        df = self.load_applicants()

        matches = df[df['applicant_id'] == applicant_id]
        if matches.empty:
            return {'error': f'Applicant ID {applicant_id} not found.'}
        bundle = self.bundle
        pipeline = bundle['pipeline']
        threshold = bundle['threshold']
        feature_columns = bundle['feature_columns']
        model_input = matches[feature_columns].iloc[[0]]
        positive_class_index = self.positive_class_index
        probability = float(pipeline.predict_proba(model_input)[0, positive_class_index])
        predicted_risk = 'High' if probability >= threshold else 'Low'
        raw_features = model_input.iloc[0]
        missing_features = raw_features[raw_features.isna()].index.tolist()
        features = {name: None if pd.isna(value) else float(value) for name, value in raw_features.items()}
        observed_features = {name: value for name, value in features.items() if value is not None}
        rule_factors, recommendations = assess_risk(observed_features)
        imputer = pipeline.named_steps['imputer']
        forest = pipeline.named_steps['classifier']
        prepared_input = imputer.transform(model_input)
        feature_names = imputer.get_feature_names_out(feature_columns)
        positive_class_index = int(np.flatnonzero(forest.classes_ == 1)[0])
        shap_result = self.explainer(prepared_input)
        contributions = shap_result.values[0, :, positive_class_index]
        baseline = float(shap_result.base_values[0, positive_class_index])
        np.testing.assert_allclose(baseline + contributions.sum(), probability, atol=1e-06, rtol=1e-06)
        shap_factors = [{'feature': str(name), 'model_input_value': float(value), 'contribution': float(contribution)} for name, value, contribution in zip(feature_names, prepared_input[0], contributions)]
        shap_factors.sort(key=lambda factor: abs(factor['contribution']), reverse=True)
        return {'applicant_id': int(applicant_id), 'features': features, 'predicted_probability': probability, 'predicted_risk': predicted_risk, 'classification_threshold': float(threshold), 'missing_features': missing_features, 'rule_based_factors': rule_factors, 'recommendations': recommendations, 'shap_baseline': baseline, 'shap_factors': shap_factors}

    def search_applicants(self, query, limit=3):
        df = self.load_scored_applicants()
        q = query.lower()
        if 'high credit utilization' in q:
            results = df[df['RevolvingUtilizationOfUnsecuredLines'] > 0.35]
        elif 'low risk' in q:
            results = df[df['predicted_risk'] == 'Low']
        elif 'high risk' in q:
            results = df[df['predicted_risk'] == 'High']
        elif 'late' in q or 'delinquency' in q:
            results = df[(df['NumberOfTime30-59DaysPastDueNotWorse'] > 0) | (df['NumberOfTime60-89DaysPastDueNotWorse'] > 0) | (df['NumberOfTimes90DaysLate'] > 0)]
        else:
            return 'Supported searches: high risk, low risk, high credit utilization, or delinquency.'
        display_columns = ['applicant_id', 'predicted_risk', 'predicted_probability', 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
        return results[display_columns].head(limit)
