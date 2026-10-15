"""Evidence validation and optional local LLM integration; no Streamlit calls."""
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FEATURE_LABELS = {'RevolvingUtilizationOfUnsecuredLines': 'Credit utilization', 'age': 'Age', 'NumberOfTime30-59DaysPastDueNotWorse': '30–59 day delinquency count', 'DebtRatio': 'Debt ratio', 'MonthlyIncome': 'Monthly income', 'NumberOfOpenCreditLinesAndLoans': 'Open credit lines and loans', 'NumberOfTimes90DaysLate': '90+ day delinquency count', 'NumberRealEstateLoansOrLines': 'Real estate loans or lines', 'NumberOfTime60-89DaysPastDueNotWorse': '60–89 day delinquency count', 'NumberOfDependents': 'Dependents'}

def format_value(name, value):
    if value is None:
        return 'Unknown'
    if name == 'RevolvingUtilizationOfUnsecuredLines':
        return f'{value:.2%}'
    if name == 'DebtRatio':
        return f'{value:.3f}'
    if name == 'MonthlyIncome':
        return f'${value:,.2f}'
    return str(int(value))

def contribution_label(name):
    if name.startswith('missingindicator_'):
        original_name = name.removeprefix('missingindicator_')
        label = FEATURE_LABELS.get(original_name, original_name)
        return f'Missing indicator: {label}'
    return FEATURE_LABELS.get(name, name)

def validate_explanation(payload, expected):
    evidence = expected['factors']
    if not isinstance(payload, dict):
        raise ValueError('Response must be a JSON object.')
    required_keys = {'risk_label', 'factors', 'missing_features'}
    if set(payload) != required_keys:
        raise ValueError('Unexpected or missing response fields.')
    if payload['risk_label'] != expected['risk_label']:
        raise ValueError('Incorrect risk label.')
    factors = payload['factors']
    if not isinstance(factors, list):
        raise ValueError('Factors must be a list.')
    expected_directions = {item['feature']: item['direction'] for item in evidence}
    seen = set()
    for factor in factors:
        if not isinstance(factor, dict):
            raise ValueError('Each factor must be an object.')
        if set(factor) != {'feature', 'direction'}:
            raise ValueError('Unexpected factor fields.')
        feature = factor['feature']
        direction = factor['direction']
        if not isinstance(feature, str):
            raise ValueError('Feature must be a string.')
        if not isinstance(direction, str):
            raise ValueError('Direction must be a string.')
        if feature not in expected_directions:
            raise ValueError(f'Unsupported feature: {feature}')
        if feature in seen:
            raise ValueError(f'Duplicate feature: {feature}')
        if direction != expected_directions[feature]:
            raise ValueError(f'Incorrect direction for: {feature}')
        seen.add(feature)
    if seen != set(expected_directions):
        raise ValueError('Required features are missing.')
    missing = payload['missing_features']
    if not isinstance(missing, list):
        raise ValueError('Missing features must be a list.')
    if not all((isinstance(name, str) for name in missing)):
        raise ValueError('Missing feature names must be strings.')
    if len(missing) != len(set(missing)):
        raise ValueError('Duplicate missing-feature names.')
    if set(missing) != set(expected['missing_features']):
        raise ValueError('Incorrect missing-feature information.')
    return payload

def render_explanation(payload, expected):
    evidence = expected['factors']
    directions = {item['feature']: item['direction'] for item in payload['factors']}
    sentences = [f"The model assigns a {payload['risk_label']} label because its estimated probability of serious delinquency within two years is " + ('at or above' if payload['risk_label'] == 'High' else 'below') + ' the configured classification threshold.']
    if evidence:
        sentences.append('Among the largest SHAP contributions:')
        for item in evidence:
            feature = item['feature']
            direction = directions[feature]
            sentences.append(f'The contribution from {feature} {direction} the prediction relative to the SHAP baseline.')
    if payload['missing_features']:
        labels = [FEATURE_LABELS.get(name, name) for name in expected['missing_features']]
        sentences.append(f"Unknown observed values: {', '.join(labels)}. The model uses training-set medians for these values.")
    sentences.append('SHAP contributions describe model behavior, not causal effects.')
    return ' '.join(sentences)

def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result

def build_expected(data):
    ranked = sorted(data["shap_factors"], key=lambda f: abs(f["contribution"]), reverse=True)
    top = [f for f in ranked if abs(f["contribution"]) > 1e-6][:5]
    return {
        "risk_label": data["predicted_risk"],
        "factors": [
            {"feature": contribution_label(f["feature"]),
             "direction": "raises" if f["contribution"] > 0 else "lowers"}
            for f in top
        ],
        "missing_features": list(data["missing_features"])
    }


def parse_explanation(raw_response):
    if not isinstance(raw_response, str):
        raise ValueError("Expected a text response.")
    json_text = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text,
                          flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        json_text = fenced.group(1)
    payload = json.loads(json_text, object_pairs_hook=reject_duplicate_keys)
    if isinstance(payload, dict) and set(payload) == {"evidence"}:
        payload = payload["evidence"]
    return payload


@dataclass(frozen=True)
class ExplanationResult:
    text: str
    source: str
    fallback_reason: str | None = None


def generate_explanation(data, *, use_llm=True, client=None):
    """Return validated/template text and status. Inject a client for offline tests."""
    expected = build_expected(data)
    if not use_llm:
        return ExplanationResult(render_explanation(expected, expected), "template")
    prompt = f"""Return only a JSON object representing the supplied evidence.
Required fields:
- risk_label: copy the supplied label exactly.
- factors: include every supplied feature exactly once, with its exact
  feature name and direction.
- missing_features: copy the supplied list exactly.
Do not add fields, explanations, Markdown fences, or text outside JSON.
Do not change directions or infer additional facts.
Supplied evidence:
{json.dumps(expected, indent=2)}
"""
    try:
        if client is None:
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(
                model="deepseek-r1-distill-qwen-1.5b",
                base_url="http://127.0.0.1:1234/v1",
                api_key="lm-studio", temperature=0, timeout=60, max_retries=0
            )
        raw = client.invoke(prompt).content
        payload = validate_explanation(parse_explanation(raw), expected)
    except Exception as exc:
        logger.warning("LLM fallback reason: %s", exc)
        return ExplanationResult(render_explanation(expected, expected), "template", str(exc))
    return ExplanationResult(render_explanation(payload, expected), "validated_llm")
